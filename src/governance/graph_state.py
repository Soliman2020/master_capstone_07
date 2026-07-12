"""Shared state that flows through every node in the agent graph.

Domain-agnostic. The only domain-specific bit is ``domain_state``, a plain
dict that the domain layer fills with its own context (e.g. incident/zone
ids for the SOC copilot) — so the governance nodes never need to know what's
inside it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict


@dataclass
class PlanStep:
    """One step the planner decided to take. ``action`` is a domain action name.

    ``args`` is optional pre-filled arguments (used by scripted scenarios in
    stub mode); in LLM mode the worker fills args from the model's output.
    """

    action: str
    reason: str = ""
    expected_side_effect: bool = False
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionIntent:
    """What the worker wants to do, parsed from the LLM's tool-call output."""

    action: str
    args: dict[str, Any] = field(default_factory=dict)
    side_effect: bool = False
    cost_estimate: float | None = None


@dataclass
class ToolResult:
    """Result of actually calling a tool after the reviewer approved it."""

    tool: str
    ok: bool
    summary: str
    payload: Any = None


@dataclass
class ReviewDecision:
    """The reviewer's verdict on an ActionIntent. Mirrors policy.ReviewDecision
    but lives in state so the graph routers can read it without re-importing policy."""

    allow: bool
    require_human: bool
    violations: list[str] = field(default_factory=list)
    reason: str = ""

    @property
    def route(self) -> str:
        # The graph's conditional edge after the reviewer reads this to pick the
        # next node: block -> summarizer, require_human -> human_approval, allow
        # -> worker_dispatch. Putting it here keeps the routing logic with the
        # verdict instead of scattered in the graph builder.
        if not self.allow:
            return "block"
        if self.require_human:
            return "require_human"
        return "allow"


class AgentState(TypedDict, total=False):
    # Who/what this turn is for.
    user_id: str
    turn_id: str
    # The conversation so far (langchain BaseMessages) + the redacted user text
    # for this turn (PII already scrubbed by the intake node).
    messages: list
    redacted_text: str
    # The plan and where we are in it.
    plan: list[PlanStep]
    step_index: int
    # The current action the worker extracted, the reviewer's verdict on it, and
    # the result of executing it (only set once the reviewer allowed + dispatch
    # ran). These three fields are the per-step state the loop reuses.
    current_action: ActionIntent | None
    review: ReviewDecision | None
    tool_result: ToolResult | None
    # Memory cross-link (user_id:turn_id) so an audit line can point at the
    # scratchpad row for the same turn.
    scratchpad_ref: str
    # Domain context — opaque to governance. The domain layer puts its own ids
    # here (e.g. incident/zone for the SOC copilot). Governance nodes never read inside.
    domain_state: dict[str, Any]
    # Loop guard + lifecycle status. status lets the summarizer mark a turn done.
    iteration: int
    status: Literal["planning", "executing", "blocked", "done"]