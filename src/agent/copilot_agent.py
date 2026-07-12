"""Wires the governance graph to the SOC domain + runs the copilot on an incident.

Lifts P6's build_graph call verbatim (the governance spine) and injects P7's
SOC domain: policy.yaml (security_operations), TOOL_REGISTRY (fusion/RAG/
summarizer wrappers), the event-driven intake node, and SOC prompts.

LLM: Groq llama-3.1-8b-instant (free cloud) via a tiny LangChain-compatible
adapter (GroqChat below). P6 used ChatOllama; we don't have the openai SDK
installed and don't want to add it, so GroqChat wraps `requests` and exposes
the one method the governance nodes actually call: ``.invoke(messages)`` ->
object with a ``.content`` attribute (mirrors langchain AIMessage).

Stub mode (no GROQ_API_KEY): llm=None, the graph still runs end-to-end via
the deterministic stubs in governance/nodes.call_llm. That keeps the agent
testable without network — the same property P6 had.

CLI:
    python -m project_07_final_synthesis.src.agent.copilot_agent --incident INC-000001
    python src/agent/copilot_agent.py --incident INC-000001
    python src/agent/copilot_agent.py --incident INC-000001 --llm   # use Groq
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass

_PROJECT = Path(__file__).resolve().parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[3]
# Put BOTH the project dir (so `from src...` resolves) AND src/ itself on the
# path. The src/ entry is what lets us import `governance.*` and `domain.*` the
# P6 way — keeping the lifted governance package byte-identical (it uses
# `from governance...` / relative imports, not `from src.governance...`).
_SRC = _PROJECT / "src"
sys.path.insert(0, str(_PROJECT))
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_REPO_ROOT))
os.chdir(_REPO_ROOT)

from dotenv import load_dotenv
load_dotenv(_PROJECT / ".env")

from governance.audit import AuditLogger
from governance.graph_builder import build_graph
from governance.memory import SessionScratchpad
from governance.policy import Policy
from governance.graph_state import PlanStep  # noqa: F401  (used by scripted plans)

from domain import tools as domain_tools
from domain.intake_node import make_intake_node
from domain.prompts import DEFAULT_PROMPTS

DATA_DIR = _PROJECT / "data"


@dataclass
class BuiltSystem:
    graph: object
    audit: AuditLogger
    memory: SessionScratchpad
    policy: Policy
    llm: object


class GroqChat:
    """Minimal LangChain-compatible chat model over Groq's OpenAI-compatible
    endpoint. Exposes only .invoke(messages) -> AIMessage-like, which is all
    the governance nodes (call_llm) call. No openai SDK, no langchain wrapper.

    messages: list of langchain BaseMessage (SystemMessage/HumanMessage) or
    dicts with role/content. We normalize both to Groq's {role, content} form.
    """

    def __init__(self, model: str | None = None, temperature: float = 0.2,
                 max_retries: int = 3, timeout: int = 60):
        from src.agent.summarizer import _api_key  # reuse the same key loader
        self.api_key = _api_key()
        self.model = model or os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
        self.url = (os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
                    + "/chat/completions")
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout

    def _to_role(self, m) -> dict:
        # langchain BaseMessage exposes .type ('system'/'human'/'ai') + .content.
        if hasattr(m, "type") and hasattr(m, "content"):
            role = "assistant" if m.type == "ai" else m.type  # 'system'/'user'
            if role == "human":
                role = "user"
            return {"role": role, "content": m.content}
        if isinstance(m, dict):
            return {"role": m["role"], "content": m["content"]}
        return {"role": "user", "content": str(m)}

    def invoke(self, messages, **_kwargs):
        import requests
        # Real langchain AIMessage so governance/nodes.call_llm's
        # isinstance(resp, AIMessage) check passes and returns .content.
        # (A duck-typed stand-in fails that check -> str(resp) -> object repr.)
        from langchain_core.messages import AIMessage
        body = {
            "model": self.model,
            "messages": [self._to_role(m) for m in messages],
            "temperature": self.temperature,
        }
        headers = {"Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"}
        last = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(self.url, headers=headers, json=body,
                                      timeout=self.timeout)
            except requests.RequestException as e:
                last = e
                time.sleep(2 ** attempt)
                continue
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                return AIMessage(content=content)
            if resp.status_code in (429, 500, 502, 503, 504):
                last = RuntimeError(f"Groq {resp.status_code}: {resp.text[:200]}")
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"Groq {resp.status_code}: {resp.text[:300]}")
        raise RuntimeError(f"Groq call failed after {self.max_retries} retries: {last}")


def build_system(use_llm: bool = False) -> BuiltSystem:
    """Assemble policy, audit, memory, tools, LLM (optional), and the graph."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    policy = Policy.from_yaml(_PROJECT / "src" / "domain" / "policy.yaml")
    audit = AuditLogger(DATA_DIR / "audit.jsonl")
    memory = SessionScratchpad(DATA_DIR / "scratchpad.db")
    llm = GroqChat() if use_llm else None
    intake = make_intake_node(policy, audit, llm=llm)
    graph = build_graph(
        policy=policy,
        tool_registry=domain_tools.TOOL_REGISTRY,
        tool_specs=[],  # real tool-calling is wired through the registry
        audit=audit, llm=llm, memory=memory,
        intake_fn=intake, prompts=DEFAULT_PROMPTS,
    )
    return BuiltSystem(graph=graph, audit=audit, memory=memory, policy=policy, llm=llm)


# --- scripted plans (deterministic; work in stub mode without Groq) ---------
# Each scenario injects a plan via the intake node so the agent run is
# deterministic even with no LLM. With --llm, the planner generates the plan
# from the staged incident text instead.

def _plan_for(incident: dict) -> list[PlanStep]:
    """A sensible default plan for any incident: fuse -> score -> retrieve ->
    summarize, then escalate if the band is critical."""
    steps = [
        PlanStep(action="incident.fuse", reason="read the fused incident",
                 expected_side_effect=False, args={"incident_id": incident["incident_id"]}),
        PlanStep(action="incident.score", reason="confirm risk band",
                 expected_side_effect=False, args={"incident_id": incident["incident_id"]}),
        PlanStep(action="sop.retrieve", reason="fetch relevant policy",
                 expected_side_effect=False, args={"incident_id": incident["incident_id"]}),
        PlanStep(action="incident.summarize", reason="analyst-facing summary",
                 expected_side_effect=False, args={"incident_id": incident["incident_id"]}),
    ]
    if incident["risk_band"] == "critical":
        steps.append(PlanStep(
            action="incident.escalate", reason="critical band -> escalate",
            expected_side_effect=True,
            args={"incident_id": incident["incident_id"],
                  "risk_band_score": int(incident["risk_score"])},
        ))
    return steps


def _intake_with_plan(incident: dict, base_intake):
    """Wrap the base intake so the staged plan is injected into state (the
    planner keeps a pre-injected plan instead of calling the LLM)."""
    plan = _plan_for(incident)

    def intake(state):
        out = base_intake(state)
        out["plan"] = plan
        return out

    return intake


def run_incident(incident_id: str, use_llm: bool = False) -> dict:
    """Run the copilot on one incident. Returns the final state snapshot."""
    from src.utils.io import read_parquet
    df = read_parquet("incidents")
    row = df[df["incident_id"] == incident_id]
    if row.empty:
        raise ValueError(f"incident {incident_id} not found")
    incident = row.iloc[0].to_dict()
    incident["risk_score"] = int(incident["risk_score"])

    sys_ = build_system(use_llm=use_llm)
    # In stub mode: inject a scripted plan so the run is deterministic without
    # an LLM. With --llm: use the plain intake and let the planner generate the
    # plan from the staged incident text via Groq (the real agent path).
    if use_llm:
        intake = make_intake_node(sys_.policy, sys_.audit, llm=sys_.llm)
    else:
        intake = _intake_with_plan(
            incident, make_intake_node(sys_.policy, sys_.audit, llm=sys_.llm))
    graph = build_graph(
        policy=sys_.policy, tool_registry=domain_tools.TOOL_REGISTRY, tool_specs=[],
        audit=sys_.audit, llm=sys_.llm, memory=sys_.memory,
        intake_fn=intake, prompts=DEFAULT_PROMPTS,
    )
    init_state = {
        "user_id": "analyst-1",
        "turn_id": f"turn-{incident_id}",
        "messages": [],
        "domain_state": {"incident_id": incident_id},
    }
    final = graph.invoke(init_state, config={"recursion_limit": 25})
    return final


def _main() -> None:
    ap = argparse.ArgumentParser(description="P7 SOC copilot agent (LangGraph + Groq).")
    ap.add_argument("--incident", required=True, help="incident_id to triage (e.g. INC-000001)")
    ap.add_argument("--llm", action="store_true", help="use Groq LLM (needs GROQ_API_KEY)")
    args = ap.parse_args()

    state = run_incident(args.incident, use_llm=args.llm)
    print(f"=== {args.incident} | status={state.get('status')} ===")
    tr = state.get("tool_result")
    if tr:
        print(f"last tool: {tr.tool} ok={tr.ok}")
        print(f"  summary: {tr.summary}")
    rev = state.get("review")
    if rev and not rev.allow:
        print(f"BLOCKED: {rev.reason}")
    print(f"audit log: {DATA_DIR / 'audit.jsonl'}")


if __name__ == "__main__":
    _main()