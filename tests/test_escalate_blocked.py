"""SOC equivalent of the property-management eviction-blocked test.

Asserts the non-bypassable policy gate blocks a low-risk escalation:
incident.escalate with risk_band_score < 75 must be blocked (fail closed),
and the tool must NOT run. This is the human-in-the-loop safety rule —
critical-band escalation is the only path that can reach dispatch, and only
after human approval. A high-band incident asking to escalate is refused.

Mirrors the structure of a property-management eviction-blocked test so the
two domains' safety guarantees are demonstrably the same shape.
"""
import sys
import tempfile
from pathlib import Path

# Allow ``pytest tests/`` from the project root without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from governance.audit import AuditLogger
from governance.graph_builder import build_graph
from governance.graph_state import PlanStep
from governance.memory import SessionScratchpad
from governance.policy import Policy

from domain import tools as domain_tools
from domain.prompts import DEFAULT_PROMPTS


def _policy():
    return Policy.from_yaml(Path(__file__).resolve().parents[1] / "src" / "domain" / "policy.yaml")


def _intake_with_plan(plan):
    def intake(state):
        state["redacted_text"] = "escalate a high-band incident"
        state["plan"] = plan
        return state
    return intake


def test_escalate_high_band_is_blocked():
    """An escalation with risk_band_score=64 (< 75) must NOT reach the tool.

    The reviewer's constraint check (risk_band_score: {ge: 75}) fails, the
    route is 'block', and the tool never runs. The dispatched tool would
    record escalated=True only if it ran; we assert it did not.
    """
    escalated = {"ran": False, "value": None}
    # Sentinel tool that records if it ever executes. If the gate holds, it
    # never runs and escalated['ran'] stays False.
    def fake_escalate(**args):
        escalated["ran"] = True
        escalated["value"] = args
        return {"escalated": True}

    registry = dict(domain_tools.TOOL_REGISTRY)
    registry["incident.escalate"] = fake_escalate

    plan = [PlanStep(action="incident.escalate", reason="try to escalate high-band",
                     expected_side_effect=True,
                     args={"incident_id": "INC-000002", "risk_band_score": 64})]

    # ignore_cleanup_errors handles the Windows SQLite file-handle lingering on
    # temp-dir teardown (same pattern as the eviction-blocked test). Snapshot
    # audit records inside the with block, before cleanup.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        audit = AuditLogger(Path(d) / "audit.jsonl")
        memory = SessionScratchpad(Path(d) / "scratch.db")
        graph = build_graph(
            policy=_policy(), tool_registry=registry, tool_specs=[],
            audit=audit, llm=None, memory=memory,
            intake_fn=_intake_with_plan(plan), prompts=DEFAULT_PROMPTS,
        )
        result = graph.invoke(
            {"user_id": "analyst-1", "turn_id": "t-escalate-block", "messages": []},
            config={"recursion_limit": 25},
        )
        blocks = [r for r in audit.read_all() if r.get("kind") == "block"]

    # The gate must have blocked it.
    review = result.get("review")
    assert review is not None, "reviewer never produced a verdict"
    assert not review.allow, f"gate wrongly allowed low-risk escalation: {review}"
    assert "constraint_failed:risk_band_score:ge:80" in review.violations, review.violations
    # And the tool must NOT have run.
    assert escalated["ran"] is False, "incident.escalate ran despite the policy block"
    # The audit log must record the block.
    assert blocks and blocks[0]["action"] == "incident.escalate", blocks


def test_escalate_critical_band_requires_human():
    """A critical-band escalation (risk_band_score=82 >= 75) is allowed but
    routed through human_approval (require_human). In demo mode human_approval
    auto-approves, so the tool runs — but only after the human node, never
    straight from the reviewer. This is the human-in-the-loop gate.
    """
    escalated = {"ran": False}
    def fake_escalate(**args):
        escalated["ran"] = True
        return {"escalated": True, "risk_band_score": args.get("risk_band_score")}

    registry = dict(domain_tools.TOOL_REGISTRY)
    registry["incident.escalate"] = fake_escalate

    plan = [PlanStep(action="incident.escalate", reason="critical band -> escalate",
                     expected_side_effect=True,
                     args={"incident_id": "INC-000001", "risk_band_score": 82})]

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        audit = AuditLogger(Path(d) / "audit.jsonl")
        memory = SessionScratchpad(Path(d) / "scratch.db")
        graph = build_graph(
            policy=_policy(), tool_registry=registry, tool_specs=[],
            audit=audit, llm=None, memory=memory,
            intake_fn=_intake_with_plan(plan), prompts=DEFAULT_PROMPTS,
        )
        result = graph.invoke(
            {"user_id": "analyst-1", "turn_id": "t-escalate-human", "messages": []},
            config={"recursion_limit": 25},
        )
        kinds = [r.get("kind") for r in audit.read_all()]

    review = result.get("review")
    assert review is not None
    assert review.allow, "critical-band escalation should be allowed (then human-gated)"
    assert review.require_human, "critical escalation must require human approval"
    # The tool ran only after human_approval auto-approved in demo mode.
    assert escalated["ran"] is True, "critical escalate should run after human approval"
    # And the audit log shows a human-approval record before the call.
    assert "human_approval" in kinds, "no human-approval record for critical escalation"
    assert "call" in kinds