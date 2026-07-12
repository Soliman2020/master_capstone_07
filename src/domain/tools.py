"""SOC tools the worker can call, registered by action name.

Each tool is a plain function in ``TOOL_REGISTRY`` keyed by the dotted action
name (matching domain/policy.yaml). The worker dispatch node calls these
*after* the reviewer approves: ``tool_fn(**intent.args)`` -> payload dict.

This mirrors P6's domain/tools.py contract exactly — only the tools changed:
P6 read tenants/ledgers from SQLite; P7 reads incidents from Parquet and
calls the fusion / RAG / summarizer layers already built in src/.

Tools are thin: they delegate to the existing src/fusion, src/rag, and
src/agent modules. No business logic duplicated here.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Bootstrap: project dir (so `from src...` resolves) AND src/ itself (so
# `governance.*` / `domain.*` resolve P6-style, keeping governance verbatim)
# AND repo root (cwd-relative data paths).
_PROJECT = Path(__file__).resolve().parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _PROJECT / "src"
sys.path.insert(0, str(_PROJECT))
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_REPO_ROOT))
os.chdir(_REPO_ROOT)

import pandas as pd

from src.utils.io import read_parquet
from src.rag.retriever import retrieve_for_incident


# --- helpers ----------------------------------------------------------------

def _load_incidents() -> pd.DataFrame:
    """Read the fused incidents parquet (fusion output)."""
    return read_parquet("incidents")


def _incident_row(incident_id: str) -> dict:
    """Fetch one incident as a plain dict. Raises ValueError if not found."""
    df = _load_incidents()
    row = df[df["incident_id"] == incident_id]
    if row.empty:
        raise ValueError(f"incident {incident_id} not found in incidents.parquet")
    return row.iloc[0].to_dict()


# --- tools (names match domain/policy.yaml) ---------------------------------

def incident_fuse(incident_id: str = "") -> dict:
    """Return the fused incident row (the fusion layer's output for one id).

    Read-only: fusion itself is run out-of-band (src/fusion/incidents.py);
    this tool surfaces a specific incident to the agent. The reviewer gate is
    the point — the agent cannot fabricate an incident, it can only read one
    that the deterministic fusion layer already produced.
    """
    if not incident_id:
        df = _load_incidents()
        return {"count": len(df), "incident_ids": df["incident_id"].tolist()}
    row = _incident_row(incident_id)
    # Keep the payload small + auditable: the fields the planner/summarizer need.
    return {
        "incident_id": row["incident_id"],
        "incident_type": row["incident_type"],
        "risk_score": int(row["risk_score"]),
        "risk_band": row["risk_band"],
        "zone_id": row["zone_id"],
        "linked_event_ids": row["linked_event_ids"],
        "linked_log_ids": row["linked_log_ids"],
    }


def incident_score(incident_id: str) -> dict:
    """Return the risk_score + risk_band for an incident (rule layer output)."""
    row = _incident_row(incident_id)
    return {
        "incident_id": row["incident_id"],
        "risk_score": int(row["risk_score"]),
        "risk_band": row["risk_band"],
        "human_review_required": bool(row["human_review_required"]),
    }


def sop_retrieve(incident_id: str, k: int = 3) -> dict:
    """Retrieve the relevant policy docs for an incident, routed on its type.

    Uses retrieve_for_incident (category-routed MMR) so the matching policy
    ranks first. Returns doc_ids + titles + scores for the agent to cite.
    """
    row = _incident_row(incident_id)
    docs = retrieve_for_incident(row["incident_type"], _query_text(row), k=k)
    return {
        "incident_id": incident_id,
        "incident_type": row["incident_type"],
        "docs": [
            {"doc_id": d["doc_id"], "title": d["title"], "category": d["category"],
             "score": round(d["score"], 3)}
            for d in docs
        ],
    }


def incident_summarize(incident_id: str) -> dict:
    """Generate (or read) the analyst-facing summary + recommended_action
    with doc_id citations.

    Delegates to src.agent.summarizer.summarize_incident, which calls Groq
    with the citation guard (validate + one retry). If the summarizer already
    ran over the incidents parquet, this returns the stored summary; otherwise
    it computes one on the fly.
    """
    from src.agent.summarizer import summarize_incident
    row = _incident_row(incident_id)
    out = summarize_incident({
        "incident_id": row["incident_id"],
        "incident_type": row["incident_type"],
        "risk_band": row["risk_band"],
        "risk_score": int(row["risk_score"]),
        "zone_id": row["zone_id"],
        "linked_event_ids": row["linked_event_ids"],
        "linked_log_ids": row["linked_log_ids"],
    })
    return {
        "incident_id": out["incident_id"],
        "summary_text": out["summary_text"],
        "recommended_action": out["recommended_action"],
        "citation_doc_ids": out["citation_doc_ids"],
        "status": out.get("_summary_status", "ok"),
    }


def incident_escalate(incident_id: str, risk_band_score: int) -> dict:
    """Escalate a critical incident for human response.

    Side-effect tool: the policy requires risk_band_score >= 75 AND a human
    approval. This tool records the escalation intent; a production deployment
    would page the duty manager here. In the slice it returns the escalation
    record so the audit trail + summarizer can cite it.
    """
    row = _incident_row(incident_id)
    return {
        "incident_id": incident_id,
        "risk_band_score": risk_band_score,
        "risk_band": row["risk_band"],
        "escalated": risk_band_score >= 75,
        "action": "page_duty_manager",
        "note": "Escalation recorded; awaiting human approval (demo auto-approves).",
    }


def case_close(incident_id: str) -> dict:
    """Close an incident case.

    HARD-BLOCKED by policy (allow: false) — the agent can request it but the
    reviewer never lets the tool run. If this function ever executes, the
    policy gate has failed. The tool is here only so the registry has the
    callable for the action name; the reviewer blocks it before dispatch.
    """
    # If we ever reach here the policy gate is broken — surface it loudly.
    raise RuntimeError(
        "case.close reached dispatch — the policy gate failed to block it. "
        "This is a safety violation; case closure must be a human act."
    )


# --- registry (the worker dispatches against this) ---------------------------

import inspect as _inspect


def _filter_kwargs(fn, kwargs: dict) -> dict:
    """Drop kwargs the tool function doesn't accept.

    The LLM worker can invent argument names (e.g. 'sop_id' for sop.retrieve)
    that don't match the tool's signature -> TypeError at dispatch. Rather
    than crash the turn, keep only the kwargs the function actually takes
    (plus anything if the function accepts **kwargs). Robust against a
    free-model's unreliable arg naming.
    """
    sig = _inspect.signature(fn)
    params = sig.parameters
    if any(p.kind == _inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs  # fn accepts **kwargs -> pass everything through
    return {k: v for k, v in kwargs.items() if k in params}


def _wrap(fn):
    """Wrap a tool so dispatch never crashes on an unknown kwarg."""
    def wrapped(**kwargs):
        return fn(**_filter_kwargs(fn, kwargs))
    wrapped.__name__ = fn.__name__
    return wrapped


TOOL_REGISTRY = {
    "incident.fuse": _wrap(incident_fuse),
    "incident.score": _wrap(incident_score),
    "sop.retrieve": _wrap(sop_retrieve),
    "incident.summarize": _wrap(incident_summarize),
    "incident.escalate": _wrap(incident_escalate),
    "case.close": _wrap(case_close),
}


# --- helper ----------------------------------------------------------------

def _query_text(row: dict) -> str:
    """Build the retrieval query from an incident row."""
    return (
        f"{row['incident_type']} in zone {row['zone_id']}; "
        f"events {row.get('linked_event_ids', '')}; "
        f"logs {row.get('linked_log_ids', '')}"
    )


if __name__ == "__main__":
    # Self-check: the read-only tools work against the fused incidents.
    df = _load_incidents()
    assert not df.empty, "no incidents — run src/fusion/incidents.py first"
    iid = df.iloc[0]["incident_id"]
    f = incident_fuse(iid)
    assert f["incident_id"] == iid, f
    s = incident_score(iid)
    assert "risk_band" in s, s
    r = sop_retrieve(iid, k=1)
    assert r["docs"], "sop.retrieve returned no docs (vector store built?)"
    print("domain/tools.py self-check OK:", iid, s["risk_band"], r["docs"][0]["doc_id"])