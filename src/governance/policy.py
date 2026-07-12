"""Domain-agnostic policy gate.

Loads a YAML policy and evaluates an :class:`ActionIntent` against it. The
constraint check is a small predicate evaluator (``gt``/``ge``/``lt``/``le``/
``in``/``regex_match``) over a ``constraints`` mapping, so domain-specific
notions like a spend cap and a risk-band threshold are checked by the *same*
mechanism. That is the linchpin that lets a domain swap in its own policy
without rewriting the reviewer.

This module imports nothing from ``src.domain`` and contains no domain string
literals — see ``tests/test_governance_no_domain_imports.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

# --- constraint predicates ---------------------------------------------------
# Each predicate takes (field_value, operand) -> bool. Field names in the YAML
# ``constraints`` mapping are arbitrary domain keys (spend_cap_usd, risk_band,
# ...); the predicate is what makes them enforceable. Reused verbatim across domains.

_PREDICATES: dict[str, Callable[[Any, Any], bool]] = {
    "gt": lambda v, o: v is not None and v > o,
    "ge": lambda v, o: v is not None and v >= o,
    "lt": lambda v, o: v is not None and v < o,
    "le": lambda v, o: v is not None and v <= o,
    "eq": lambda v, o: v == o,
    "ne": lambda v, o: v != o,
    "in": lambda v, o: v in o,
    "regex_match": lambda v, o: bool(re.search(o, str(v))) if v is not None else False,
}


def evaluate_constraints(args: dict[str, Any], constraints: dict[str, dict]) -> list[str]:
    """Return a list of violation messages for any constraint that fails.

    ``constraints`` maps a field name to a dict of predicate->operand, e.g.
    ``{"spend_cap_usd": {"le": 500}}`` reads "args['spend_cap_usd'] must be
    <= 500". A missing field that has constraints is itself a violation — a
    side-effect action with no cost estimate cannot be cleared (fail closed).
    """
    violations: list[str] = []
    for field_name, preds in constraints.items():
        value = args.get(field_name) if args else None
        for pred_name, operand in preds.items():
            pred = _PREDICATES.get(pred_name)
            if pred is None:
                violations.append(f"unknown_predicate:{pred_name}")
                continue
            if not pred(value, operand):
                violations.append(
                    f"constraint_failed:{field_name}:{pred_name}:{operand}"
                )
    return violations


# --- policy data structures --------------------------------------------------


@dataclass
class ActionRule:
    """One row of the policy's ``actions`` list."""

    name: str
    side_effect: bool
    allow: bool
    require_human: bool = False
    constraints: dict[str, dict] = field(default_factory=dict)
    require_fields: list[str] = field(default_factory=list)
    block_reason: str = ""


@dataclass
class ReviewDecision:
    """Output of :meth:`Policy.evaluate`. Consumed by the reviewer node."""

    allow: bool
    require_human: bool
    violations: list[str] = field(default_factory=list)
    reason: str = ""

    @property
    def route(self) -> str:
        """Where the graph should go next: allow | require_human | block."""
        if not self.allow:
            return "block"
        if self.require_human:
            return "require_human"
        return "allow"


@dataclass
class RedactionPattern:
    name: str
    regex: str
    replacement: str


@dataclass
class Policy:
    """Loaded policy. ``evaluate`` is the only method the reviewer calls."""

    domain: str
    actions: dict[str, ActionRule]
    redaction_patterns: list[RedactionPattern] = field(default_factory=list)
    redaction_enabled: bool = True
    intake: dict[str, Any] = field(default_factory=dict)
    retention: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Policy":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Policy":
        actions: dict[str, ActionRule] = {}
        for a in data.get("actions", []):
            actions[a["name"]] = ActionRule(
                name=a["name"],
                side_effect=bool(a.get("side_effect", False)),
                allow=bool(a.get("allow", False)),
                require_human=bool(a.get("require_human", False)),
                constraints=a.get("constraints", {}) or {},
                require_fields=list(a.get("require_fields", []) or []),
                block_reason=a.get("block_reason", "") or "",
            )
        pii = data.get("pii_redaction", {}) or {}
        patterns = [
            RedactionPattern(p["name"], p["regex"], p["replacement"])
            for p in pii.get("patterns", []) or []
        ]
        return cls(
            domain=data.get("domain", "unspecified"),
            actions=actions,
            redaction_patterns=patterns,
            redaction_enabled=bool(pii.get("enabled", True)),
            intake=data.get("intake", {}) or {},
            retention=data.get("retention", {}) or {},
            raw=data,
        )

    def get_rule(self, action: str) -> ActionRule | None:
        return self.actions.get(action)

    def evaluate(self, action: str, args: dict[str, Any] | None) -> ReviewDecision:
        """Evaluate an action+args against the policy. Pure, no I/O, no LLM."""
        rule = self.actions.get(action)
        if rule is None:
            return ReviewDecision(
                allow=False,
                require_human=False,
                violations=["unknown_action"],
                reason=f"Action '{action}' is not in the policy.",
            )
        if not rule.allow:
            # Hard block (e.g. an eviction/lockout or a SOC case-close that
            # must never be auto-executed). Still surfaces require_human so
            # the summarizer can explain the path to a human decision.
            return ReviewDecision(
                allow=False,
                require_human=rule.require_human,
                violations=["action_not_allowed"],
                reason=rule.block_reason or f"Action '{action}' is not permitted.",
            )

        violations: list[str] = []
        # Required fields first — a missing required field is a violation.
        for rf in rule.require_fields:
            if not args or args.get(rf) in (None, ""):
                violations.append(f"missing_required_field:{rf}")
        # Then predicate constraints (spend cap, risk band, ...).
        violations.extend(evaluate_constraints(args or {}, rule.constraints))

        if violations:
            # Fail closed: a side-effect action whose constraints/required
            # fields are not satisfied cannot be auto-executed. Route to a
            # human rather than silently dropping it.
            return ReviewDecision(
                allow=False,
                require_human=rule.require_human or rule.side_effect,
                violations=violations,
                reason=f"Action '{action}' failed policy checks: {violations}",
            )
        return ReviewDecision(
            allow=True,
            require_human=rule.require_human,
            violations=[],
            reason=f"Action '{action}' permitted by policy.",
        )


# one runnable check — evaluate the canonical spend-cap case and
# a hard-block case, so this file is self-verifying without the test suite.
if __name__ == "__main__":
    p = Policy.from_dict(
        {
            "domain": "self_test",
            "actions": [
                {
                    "name": "demo.spend",
                    "side_effect": True,
                    "allow": True,
                    "constraints": {"amount": {"le": 500}},
                    "require_fields": ["amount"],
                },
                {"name": "demo.forbidden", "side_effect": True, "allow": False,
                 "block_reason": "outside authority"},
            ],
        }
    )
    ok = p.evaluate("demo.spend", {"amount": 300})
    over = p.evaluate("demo.spend", {"amount": 600})
    missing = p.evaluate("demo.spend", {})
    blocked = p.evaluate("demo.forbidden", {})
    assert ok.allow and ok.route == "allow", ok
    assert not over.allow and "constraint_failed:amount:le:500" in over.violations, over
    assert not missing.allow and "missing_required_field:amount" in missing.violations, missing
    assert not blocked.allow and blocked.violations == ["action_not_allowed"], blocked
    print("policy.py self-check OK")