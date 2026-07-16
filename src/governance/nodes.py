"""The governance graph nodes and routers.

Each node is a small function ``node(state) -> state_update``. Dependencies
(LLM, policy, audit logger, memory, tool registry, prompts) are injected via a
factory so the graph is wired up in one place (graph_builder.build_graph) and
the nodes themselves stay simple.

The LLM-touching nodes (planner, worker, summarizer) call the LLM through one
helper, ``call_llm``. When ``llm is None`` (stub mode, e.g. tests without
Ollama Cloud credentials), the helper falls back to a deterministic string
return so the graph can still be exercised end-to-end. That keeps the graph
testable before the model is wired in.

Nothing in this file knows about property management — no "tenant"/"lease"/
"evict" strings. The domain layer supplies the action names and tools.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .audit import AuditLogger
from .graph_state import ActionIntent, AgentState, PlanStep, ReviewDecision, ToolResult
from .memory import SessionScratchpad
from .policy import Policy

# --- LLM helper -------------------------------------------------------------


def call_llm(llm, system: str, user: str) -> str:
    """One LLM call returning text. Falls back to a stub when llm is None.

    The stub returns a JSON-shaped string so the downstream parsers (parse_plan,
    parse_action_intent) have something to chew on. This is what lets the whole
    graph run in tests / CI without Ollama Cloud credentials.
    """
    if llm is None:
        return json.dumps({"action": "noop", "args": {}, "summary": "[stub] no LLM configured"})
    msgs = [SystemMessage(content=system), HumanMessage(content=user)]
    resp = llm.invoke(msgs)
    # langchain chat models return an AIMessage; normalize to text.
    return resp.content if isinstance(resp, AIMessage) else str(resp)


# --- Node factories ---------------------------------------------------------


def make_planner_node(llm, prompts, memory: SessionScratchpad, audit: AuditLogger):
    def planner(state: AgentState) -> dict:
        # If the intake node already injected a plan (tests / scripted runs),
        # keep it instead of calling the LLM. This lets stub-mode tests drive
        # the graph without an LLM while real runs still plan via the model.
        if state.get("plan"):
            audit.log_decision(turn_id=state["turn_id"], node="planner",
                               decision="plan_pre_injected", rationale="plan supplied by intake")
            return {"step_index": 0, "iteration": 0, "status": "executing"}
        user_text = state.get("redacted_text", "")
        # Inject the last few scratchpad entries so a follow-up turn can refer
        # to an earlier one ("yes, like we just discussed"). This is the whole
        # memory story — no vector store, just recent context in the prompt.
        prior = memory.recent(state["user_id"], n=3)
        prior_block = "\n".join(f"- {p['kind']}: {p['content']}" for p in prior) or "(none)"
        sys_prompt = prompts.planner_system + "\n\n<prior_turns>\n" + prior_block + "\n</prior_turns>"
        raw = call_llm(llm, sys_prompt, user_text)
        plan = parse_plan(raw)
        # Log the plan and stash a trimmed copy in memory for future turns.
        audit.log_decision(turn_id=state["turn_id"], node="planner",
                           decision="plan_issued", rationale=raw[:200])
        memory.append(state["user_id"], state["turn_id"], "plan", raw[:500])
        return {"plan": plan, "step_index": 0, "iteration": 0, "status": "executing",
                "messages": state.get("messages", []) + [AIMessage(content=raw)]}
    return planner


def make_worker_node(llm, prompts, tool_specs):
    def worker(state: AgentState) -> dict:
        # If the planner produced no steps, there is nothing to do; go straight
        # to the summarizer rather than indexing an empty plan.
        if not state.get("plan"):
            return {"current_action": None, "status": "done"}
        step: PlanStep = state["plan"][state["step_index"]]
        if llm is None:
            # Stub mode: no LLM to fill in args, so build the intent straight
            # from the plan step (which may carry pre-filled args in scripted
            # scenarios). Keeps the worker generic — no peeking at domain_state.
            return {"current_action": ActionIntent(action=step.action, args=dict(step.args),
                                                   side_effect=step.expected_side_effect)}
        sys_prompt = prompts.worker_system
        tool_names = ", ".join(t.name for t in tool_specs)
        user_msg = f"Action to perform: {step.action}\nReason: {step.reason}\nAvailable tools: {tool_names}"
        raw = call_llm(llm, sys_prompt, user_msg)
        intent = parse_action_intent(raw, step)
        return {"current_action": intent}
    return worker


def make_reviewer_node(policy: Policy, audit: AuditLogger):
    def reviewer(state: AgentState) -> dict:
        intent: ActionIntent = state.get("current_action")
        # If there is no action to review (e.g. the planner produced no plan),
        # there is nothing to gate — short-circuit to the summarizer.
        if intent is None:
            return {"review": ReviewDecision(allow=False, require_human=False,
                                             violations=["no_action"], reason="no plan produced")}
        # The gate itself. policy.evaluate is pure — no LLM, no tools — so the
        # reviewer cannot be "talked into" approving something. That's the
        # non-bypassable property the eviction case depends on.
        decision = policy.evaluate(intent.action, intent.args)
        # Translate policy.ReviewDecision -> graph_state.ReviewDecision (same shape).
        rd = ReviewDecision(allow=decision.allow, require_human=decision.require_human,
                            violations=decision.violations, reason=decision.reason)
        if rd.allow:
            audit.log_decision(turn_id=state["turn_id"], node="reviewer",
                               decision="allow", rationale=rd.reason)
        else:
            # Blocks get their own audit kind so a post-incident review can find
            # every refusal immediately.
            audit.log_block(turn_id=state["turn_id"], action=intent.action, args=intent.args,
                            violations=rd.violations, block_reason=rd.reason)
        return {"review": rd}
    return reviewer


def make_worker_dispatch_node(tool_registry: dict[str, Callable], audit: AuditLogger):
    def worker_dispatch(state: AgentState) -> dict:
        # This node is only reached AFTER the reviewer allowed the action, so
        # actually running the tool here is safe. The tool is looked up by the
        # action name (e.g. "tenant.query") in the registry the app injected.
        intent: ActionIntent = state["current_action"]
        tool_fn = tool_registry.get(intent.action)
        if tool_fn is None:
            result = ToolResult(tool=intent.action, ok=False,
                                summary=f"no tool registered for {intent.action}")
        else:
            try:
                payload = tool_fn(**intent.args)
                # Cap the summary length so the audit log doesn't balloon.
                result = ToolResult(tool=intent.action, ok=True, summary=str(payload)[:300],
                                    payload=payload)
            except Exception as e:  # student-style: catch broadly, record the failure
                result = ToolResult(tool=intent.action, ok=False, summary=f"tool error: {e}")
        # Even read-only calls are logged — the trail must show what was
        # inspected, not just what was changed (a post-incident review needs it).
        audit.log_call(turn_id=state["turn_id"], action=intent.action, args=intent.args,
                       tool=result.tool, result_summary=result.summary)
        # Advance step_index so route_after_dispatch knows whether to loop or finish.
        return {"tool_result": result, "step_index": state["step_index"] + 1}
    return worker_dispatch


def make_summarizer_node(llm, prompts, memory: SessionScratchpad, audit: AuditLogger):
    def summarizer(state: AgentState) -> dict:
        # Build a short recap of what happened this turn for the user-facing summary.
        review = state.get("review")
        tool_result = state.get("tool_result")
        if review and not review.allow:
            action_name = state["current_action"].action if state.get("current_action") else "(no action)"
            recap = f"Action '{action_name}' was blocked: {review.reason}"
            summary_text = call_llm(llm, prompts.summarizer_system, recap) if llm else recap
            memory.append(state["user_id"], state["turn_id"], "block", recap)
        elif tool_result:
            recap = f"Action '{tool_result.tool}' executed: {tool_result.summary}"
            summary_text = call_llm(llm, prompts.summarizer_system, recap) if llm else recap
            memory.append(state["user_id"], state["turn_id"], "tool", recap)
        else:
            recap = "turn complete"
            summary_text = recap
        memory.append(state["user_id"], state["turn_id"], "summary", summary_text[:500])
        audit.log_decision(turn_id=state["turn_id"], node="summarizer",
                           decision="turn_complete", rationale=summary_text[:200])
        return {"status": "done"}
    return summarizer


def make_human_approval_node(audit: AuditLogger):
    def human_approval(state: AgentState) -> dict:
        intent: ActionIntent = state["current_action"]
        review = state.get("review")

        # When COPILOT_HUMAN_GATE=1, suspend via langgraph.interrupt and
        # wait for a human to resume with Command(resume=granted_bool).
        # Otherwise auto-approve so CLI / notebook runs stay headless.
        import os as _os
        if _os.environ.get("COPILOT_HUMAN_GATE") == "1":
            from langgraph.types import interrupt
            decision = interrupt({
                "action": intent.action,
                "args": intent.args,
                "reason": getattr(review, "reason", "") if review else "",
                "turn_id": state["turn_id"],
            })
            granted = bool(decision)
            audit.log_human_approval(turn_id=state["turn_id"], action=intent.action,
                                     approver="human_via_interrupt", granted=granted,
                                     note="via langgraph interrupt (COPILOT_HUMAN_GATE=1)")
            if review:
                review.allow = granted
                review.require_human = False
            return {"review": review}

        # Default: auto-approve and log it so the graph runs end-to-end in
        # the notebook/CLI. The report documents that a production deployment
        # would block here until a human responds.
        audit.log_human_approval(turn_id=state["turn_id"], action=intent.action,
                                 approver="notebook_operator", granted=True,
                                 note="auto-approved in demo mode")
        if review:
            review.allow = True
        return {"review": review}
    return human_approval


# --- Routers (module-level functions, not lambdas, so the graph serializes) --


def route_after_review(state: AgentState) -> str:
    return state["review"].route


def route_after_dispatch(state: AgentState) -> str:
    # After a tool ran, decide: more plan steps left -> back to the worker to
    # load the next plan step's action (then reviewer -> dispatch); otherwise
    # -> summarizer to wrap up the turn.
    #
    # FIX: an earlier version returned "reviewer" here. That routed dispatch ->
    # reviewer, skipping the worker, so current_action was never refreshed and
    # step 0's action re-ran every iteration (see the matching FIX note in
    # graph_builder.py). Now returns "worker" so each loop iteration re-loads
    # plan[step_index] before the gate. (Single-step plans never exposed this.)
    if state["step_index"] >= len(state["plan"]):
        return "summarizer"
    return "worker"


# --- Parsers (kept forgiving — LLM output is messy) -------------------------


def parse_plan(raw: str) -> list[PlanStep]:
    """Parse the planner's LLM output into PlanSteps.

    Expects a JSON list of {"action", "reason", "expected_side_effect"} or,
    if the LLM didn't return JSON, a single-step plan with the raw text as the
    reason. Keeping this forgiving is intentional — a brittle parser would
    fail constantly against a real model. We also strip markdown code fences
    (```json ... ```) since most chat models wrap JSON in them.
    """
    text = raw.strip()
    # Strip ```json ... ``` or ``` ... ``` fences that models commonly add.
    if text.startswith("```"):
        # Take the content between the first and last ``` fence.
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1]
            # Drop an optional leading language tag like 'json' on its own line.
            if text.lstrip().startswith(("json", "JSON")):
                text = text.split("\n", 1)[1] if "\n" in text else text
            text = text.strip()
    try:
        items = json.loads(text)
        if isinstance(items, list):
            return [PlanStep(action=i["action"], reason=i.get("reason", ""),
                             expected_side_effect=i.get("expected_side_effect", False))
                    for i in items if isinstance(i, dict) and "action" in i]
        if isinstance(items, dict) and "action" in items:
            return [PlanStep(action=items["action"], reason=items.get("reason", ""),
                             expected_side_effect=items.get("expected_side_effect", False))]
    except (json.JSONDecodeError, TypeError, KeyError):
        pass
    # Fallback: treat the whole user request as one opaque step.
    return [PlanStep(action="unknown", reason=raw[:200])]


def parse_action_intent(raw: str, step: PlanStep) -> ActionIntent:
    """Parse the worker's LLM output into an ActionIntent.

    The LLM is asked to return JSON like {"action": "...", "args": {...}}.
    If parsing fails, we use the step's action and empty args — the reviewer
    will then block unknown/missing-field cases (fail closed).
    """
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "action" in data:
            return ActionIntent(
                action=data["action"],
                args=data.get("args", {}) or {},
                side_effect=bool(data.get("side_effect", step.expected_side_effect)),
                cost_estimate=data.get("cost_estimate"),
            )
    except (json.JSONDecodeError, TypeError):
        pass
    # Fallback: keep the plan step's action but send empty args. The reviewer
    # then fails closed (missing required fields / unknown action) rather than
    # silently letting a malformed LLM output through.
    return ActionIntent(action=step.action, side_effect=step.expected_side_effect)