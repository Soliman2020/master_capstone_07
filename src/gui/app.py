"""Streamlit GUI for the SOC Security Operations Copilot.

One-file app. Reuses `src.agent.copilot_agent.run_incident` directly — no
new abstraction, no separate API. Analysts pick an incident, run the agent,
and see the summary / recommended action / KB citations / escalation verdict
plus the hash-chained audit trail for that turn.

Run from repo root:
    project_07_final_synthesis\\final_venv\\Scripts\\python.exe -m \\
        streamlit run project_07_final_synthesis/src/gui/app.py

Default is deterministic stub mode (no Groq, reproducible). Flip "Use LLM"
to call Groq llama-3.1-8b-instant — that path needs GROQ_API_KEY in .env and
is NOT reproducible.

Streamlit instead of FastAPI+frontend = one file reusing the
existing entry point. No REST surface; add one only if another consumer
(Slack, external UI) actually needs to call the copilot remotely.
"""
from __future__ import annotations
import sys
from pathlib import Path

_PROJECT = Path(__file__).resolve().parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _PROJECT / "src"
for _p in (_PROJECT, _SRC, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd
import streamlit as st

from src.utils.io import read_parquet
from src.agent.copilot_agent import run_incident

AUDIT_PATH = _PROJECT / "data" / "audit.jsonl"

# risk_band -> display color for the status badge.
BAND_COLOR = {"critical": "red", "high": "orange",
              "medium": "blue", "low": "green"}


@st.cache_data(show_spinner=False)
def load_incidents() -> pd.DataFrame:
    return read_parquet("incidents")


def _audit_tail_for_turn(turn_id: str, limit: int = 50) -> list[dict]:
    if not AUDIT_PATH.exists():
        return []
    import json
    out = []
    with open(AUDIT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("turn_id") == turn_id:
                out.append(rec)
    # return out[-limit:]
    return out


def main() -> None:
    st.set_page_config(page_title="SOC Security Operations Copilot",
                       page_icon="🛡️", layout="wide")
    st.title("🛡️ SOC Security Operations Copilot")
    st.caption("Agent-led triage: fuse → score → retrieve policy → summarize → escalate")

    try:
        df = load_incidents()
    except FileNotFoundError as e:
        st.error(str(e))
        st.info("Run the pipeline first: `python -m project_07_final_synthesis.src.fusion.incidents`")
        return

    with st.sidebar:
        st.header("Triage controls")
        ids = sorted(df["incident_id"].tolist())
        incident_id = st.selectbox("Incident", ids, index=0)
        use_llm = st.checkbox("Use LLM (Groq llama-3.1-8b-instant)",
                              value=False, help="Off = deterministic stub mode "
                              "(reproducible, no API key). On = calls Groq "
                              "using GROQ_API_KEY from .env (non-reproducible).")
        if use_llm:
            import os
            from dotenv import load_dotenv
            load_dotenv(_PROJECT / ".env")
            if not os.environ.get("GROQ_API_KEY"):
                st.warning("GROQ_API_KEY not set in .env — LLM run will fail.")
            st.caption("⚠️ LLM mode is non-reproducible and uses Groq quota.")
        run = st.button("▶ Run copilot", type="primary")

    row = df[df["incident_id"] == incident_id].iloc[0]
    band = str(row["risk_band"])
    col1, col2, col3 = st.columns(3)
    col1.metric("Incident", incident_id)
    col2.metric("Risk band", band)
    col3.metric("Risk score", int(row["risk_score"]))
    st.caption(f"Type: **{row['incident_type']}** · "
               f"Site/zone: {row['site_id']} / {row['zone_id']}")

    with st.expander("Fused incident record (raw row)"):
        # The row mixes str / Timestamp / int / bool; Arrow/Streamlit's
        # st.dataframe chokes on the mixed object column. Build a tidy
        # two-column frame where every value is a string.
        raw = (
            row.to_frame(name="value")
            .reset_index()
            .rename(columns={"index": "field"})
        )
        raw["value"] = raw["value"].apply(
            lambda v: v.isoformat() if hasattr(v, "isoformat") else str(v)
        )
        st.dataframe(raw, width="stretch", hide_index=True)

    if not run:
        st.info("Pick an incident and press **▶ Run copilot** in the sidebar.")
        return

    with st.spinner(f"Running copilot on {incident_id} "
                    f"({'LLM' if use_llm else 'stub'} mode)…"):
        try:
            state = run_incident(incident_id, use_llm=use_llm)
        except Exception as e:
            st.error(f"Copilot run failed: {e}")
            return

    # --- verdict banner -----------------------------------------------------
    rev = state.get("review")
    if rev and not rev.allow:
        st.error(f"🚫 BLOCKED: {rev.reason}")
    elif rev and rev.require_human:
        st.warning("✋ Escalation requires human approval.")
    else:
        st.success(f"✅ Status: {state.get('status', 'done')}")

    tr = state.get("tool_result")
    if tr:
        st.subheader("Last tool result")
        st.markdown(f"`{tr.tool}` · ok={tr.ok}")
        st.write(tr.summary)

    # The agent's in-loop incident.summarize returns the enrichment dict in
    # tool_result.payload but does NOT persist it back to incidents.parquet
    # (only the standalone `summarizer` module does that). Prefer the live
    # payload when the last step was summarize, fall back to the parquet row
    # for runs that didn't reach it.
    summary_text = row["summary_text"]
    recommended_action = row["recommended_action"]
    citation_doc_ids = row["citation_doc_ids"]
    if (tr and tr.tool == "incident.summarize"
            and isinstance(getattr(tr, "payload", None), dict)):
        p = tr.payload
        summary_text = p.get("summary_text", summary_text)
        recommended_action = p.get("recommended_action", recommended_action)
        citation_doc_ids = p.get("citation_doc_ids", citation_doc_ids)

    # --- enrichment fields (filled by RAG+LLM; empty until summarizer ran) --
    st.subheader("Analyst summary")
    st.write(summary_text or "_(empty — run the summarizer step)_")

    left, right = st.columns(2)
    with left:
        st.markdown("**Recommended action**")
        st.write(recommended_action or "_(empty)_")
    with right:
        st.markdown("**Citations**")
        cids = citation_doc_ids
        if isinstance(cids, str):
            import json as _json
            try:
                cids = _json.loads(cids)
            except _json.JSONDecodeError:
                cids = [cids] if cids else []
        cids = list(cids or [])
        if cids:
            for cid in cids:
                st.markdown(f"`{cid}`")
        else:
            st.write("_(no citations)_")
        st.markdown(f"**Human review required:** {bool(row['human_review_required'])}")

    # --- audit trail --------------------------------------------------------
    st.subheader("Audit trail (hash-chained)")
    turn_id = f"turn-{incident_id}"
    records = _audit_tail_for_turn(turn_id)
    if records:
        st.dataframe(pd.DataFrame(records)[["seq", "node", "kind", "actor",
                                            "decision", "rationale", "ts"]],
                     width="stretch", hide_index=True)
        st.caption(f"{len(records)} record(s) for {turn_id} · "
                   f"verify with `AuditLogger.verify_chain()`")
    else:
        st.caption(f"No audit records yet for {turn_id}.")


if __name__ == "__main__":
    main()