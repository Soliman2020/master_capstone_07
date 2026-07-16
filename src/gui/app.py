"""Streamlit GUI for the SOC Security Operations Copilot.

One-file app. Reuses `src.agent.copilot_agent` directly — no new abstraction,
no separate API. Analysts pick an incident, run the agent, and see the
summary / recommended action / KB citations / escalation verdict plus the
hash-chained audit trail for that turn.

Human-in-the-loop: when a step requires human approval (critical-band
escalation), the agent pauses, the GUI shows the pending action and
a Grant / Deny choice, and the analyst's decision is logged to the audit
chain. This replaces the "demo auto-approves" path in the CLI/notebook
while keeping that path intact for headless use.

Run from repo root:
    project_07_final_synthesis\\final_venv\\Scripts\\python.exe -m \\
        streamlit run project_07_final_synthesis/src/gui/app.py

Default is deterministic stub mode (no Groq, reproducible). Flip "Use LLM"
to call Groq llama-3.1-8b-instant — that path needs GROQ_API_KEY in .env and
is NOT reproducible.

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
from src.agent.copilot_agent import run_incident_streaming

AUDIT_PATH = _PROJECT / "data" / "audit.jsonl"

# risk_band -> display color for the status badge.
BAND_COLOR = {"critical": "red", "high": "orange",
              "medium": "blue", "low": "green"}

# Per-run session-state keys. Reset whenever a new run starts.
_SS_RUN = "_gui_run_state"           # dict: {phase, last_state, pending, granted, summarize_payload}
_SS_HUMAN_DECIDED = "_gui_human_decided"  # bool
_SS_HUMAN_GRANT = "_gui_human_grant"      # bool
_SS_GEN = "_gui_gen"                      # the long-lived streaming generator
_SS_GEN_KEY = "_gui_gen_key"              # (incident_id, use_llm) the gen was created for


@st.cache_data(show_spinner=False)
def load_incidents() -> pd.DataFrame:
    return read_parquet("incidents")


def _audit_tail_for_turn(turn_id: str) -> list[dict]:
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
    return out


def _render_summary_panel(row: pd.Series, summarize_payload: dict | None,
                          last_state: dict | None) -> None:
    """Show the analyst enrichment fields. Prefer the most recent
    incident.summarize payload observed during streaming, fall back to
    the parquet row, and surface a useful message when neither exists."""
    summary_text = row["summary_text"]
    recommended_action = row["recommended_action"]
    citation_doc_ids = row["citation_doc_ids"]
    if summarize_payload:
        summary_text = summarize_payload.get("summary_text", summary_text) or summary_text
        recommended_action = (summarize_payload.get("recommended_action", recommended_action)
                              or recommended_action)
        citation_doc_ids = (summarize_payload.get("citation_doc_ids", citation_doc_ids)
                            or citation_doc_ids)

    st.subheader("Analyst summary")
    if summary_text:
        st.write(summary_text)
    else:
        msg = "_(no summarization step ran in this turn)_"
        if last_state and last_state.get("status") == "done":
            # The run completed without ever calling incident.summarize.
            tr = last_state.get("tool_result")
            if tr and tr.tool != "incident.summarize":
                msg = (f"_(no summarization step ran — last tool was "
                       f"`{tr.tool}`)_")
        st.write(msg)

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
        incident_id = st.selectbox("Incident", ids, index=0,
                                   key="incident_id")
        use_llm = st.checkbox(
            "Use LLM (Groq llama-3.1-8b-instant)", value=False,
            help="Off = deterministic stub mode (reproducible, no API key). "
                 "On = calls Groq using GROQ_API_KEY from .env (non-reproducible).",
            key="use_llm",
        )
        if use_llm:
            import os
            from dotenv import load_dotenv
            load_dotenv(_PROJECT / ".env")
            if not os.environ.get("GROQ_API_KEY"):
                st.warning("GROQ_API_KEY not set in .env — LLM run will fail.")
            st.caption("⚠️ LLM mode is non-reproducible and uses Groq quota.")
        run = st.button("▶ Run copilot", type="primary", key="run_btn")

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
        st.session_state.pop(_SS_RUN, None)
        st.session_state.pop(_SS_HUMAN_DECIDED, None)
        st.session_state.pop(_SS_HUMAN_GRANT, None)
        st.session_state.pop(_SS_GEN, None)
        st.session_state.pop(_SS_GEN_KEY, None)
        st.info("Pick an incident and press **▶ Run copilot** in the sidebar.")
        return

    # New run: clear any prior per-turn state. The generator is keyed on
    # (incident_id, use_llm) so a run for a different incident can't pick
    # up a leftover pending event from a previous suspended run.
    gen_key = (incident_id, use_llm)
    if st.session_state.get(_SS_GEN_KEY) != gen_key:
        st.session_state.pop(_SS_GEN, None)
        st.session_state[_SS_GEN_KEY] = gen_key
    st.session_state[_SS_RUN] = {
        "phase": "running",
        "last_state": None,
        "pending": None,
        "granted": None,
        "summarize_payload": None,
    }
    st.session_state[_SS_HUMAN_DECIDED] = False
    st.session_state[_SS_HUMAN_GRANT] = False

    if _SS_GEN not in st.session_state:
        def _on_human_approval(pending: dict) -> bool:
            # The generator calls this when the graph is suspended at
            # human_approval. The GUI shows the gate, captures the
            # decision in session_state, and yields control back. When
            # the user clicks Grant/Deny, st.rerun() brings us back
            # here with the decision already set.
            st.session_state[_SS_RUN]["phase"] = "human_decision"
            st.session_state[_SS_RUN]["pending"] = pending
            granted = st.session_state.get(_SS_HUMAN_GRANT)
            return bool(granted)

        gen = run_incident_streaming(
            incident_id, use_llm=use_llm,
            on_human_approval=_on_human_approval,
        )
        st.session_state[_SS_GEN] = gen

    # Drive the generator forward. On a normal "Run" click this loop
    # runs until the first `pending` event (or `done`). On a Grant/Deny
    # rerun, _on_human_approval returns the decision immediately and the
    # loop continues through the rest of the run.
    with st.spinner(f"Running copilot on {incident_id} "
                    f"({'LLM' if use_llm else 'stub'} mode)…"):
        try:
            gen = st.session_state[_SS_GEN]
            while True:
                event = next(gen)
                kind = event[0]
                run_state = st.session_state[_SS_RUN]
                if kind == "step":
                    state = event[1]
                    run_state["last_state"] = state
                    tr = state.get("tool_result")
                    if (tr and tr.tool == "incident.summarize"
                            and isinstance(getattr(tr, "payload", None), dict)):
                        run_state["summarize_payload"] = tr.payload
                elif kind == "pending":
                    # The agent paused at human_approval. _on_human_approval
                    # already ran and recorded the decision. We update the
                    # last_state from the snapshot and break out so the GUI
                    # can render the verdict + the gate.
                    run_state["phase"] = "human_decision"
                    run_state["pending"] = event[1]
                    break
                elif kind == "done":
                    run_state["phase"] = "done"
                    run_state["last_state"] = event[1]
                    break
        except StopIteration:
            st.session_state[_SS_RUN]["phase"] = "done"
        except Exception as e:
            st.error(f"Copilot run failed: {e}")
            st.session_state.pop(_SS_GEN, None)
            return

    # --- verdict banner ----------------------------------------------------
    run_state = st.session_state[_SS_RUN]
    state = run_state.get("last_state") or {}

    if run_state["phase"] == "human_decision":
        pending = run_state["pending"] or {}
        action = pending.get("action", "unknown")
        args = pending.get("args", {})
        reason = pending.get("review_reason", "")
        st.warning("✋ Escalation requires human approval.")
        st.markdown(f"**Pending action:** `{action}`")
        st.markdown(f"**Args:** `{args}`")
        if reason:
            st.markdown(f"**Reviewer reason:** {reason}")
        st.markdown("**Choose:**")
        c1, c2, _ = st.columns([1, 1, 4])
        with c1:
            if st.button("✅ Grant", type="primary", key="grant_btn"):
                st.session_state[_SS_HUMAN_GRANT] = True
                st.rerun()
        with c2:
            if st.button("🚫 Deny", key="deny_btn"):
                st.session_state[_SS_HUMAN_GRANT] = False
                st.rerun()
        st.stop()

    rev = state.get("review")
    if rev and not rev.allow:
        st.error(f"🚫 BLOCKED: {rev.reason}")
    elif rev and getattr(rev, "require_human", False) and getattr(rev, "allow", False):
        # Critical path that was auto-approved (shouldn't happen with
        # COPILOT_HUMAN_GATE=1, but reported faithfully if it does).
        st.warning("✋ Escalation auto-approved (no human gate).")
    else:
        st.success(f"✅ Status: {state.get('status', 'done')}")

    tr = state.get("tool_result")
    if tr:
        st.subheader("Last tool result")
        st.markdown(f"`{tr.tool}` · ok={tr.ok}")
        st.write(tr.summary)

    _render_summary_panel(row, run_state.get("summarize_payload"), state)

    # --- audit trail --------------------------------------------------------
    st.subheader("Audit trail (hash-chained)")
    turn_id = f"turn-{incident_id}"
    records = _audit_tail_for_turn(turn_id)
    if records:
        st.dataframe(pd.DataFrame(records)[["seq", "node", "kind", "actor",
                                            "decision", "rationale", "ts"]],
                     width="stretch", hide_index=True)
        # Verify the chain on the live log so the caption reflects reality.
        # Pre-existing breaks in earlier rows (e.g. from before the GUI
        # existed) show up here; the chain grows forward from the last
        # valid hash as new rows are appended.
        from src.governance.audit import AuditLogger, ChainBrokenError
        verifier = AuditLogger(AUDIT_PATH)
        try:
            verifier.verify_chain()
            chain_msg = "hash chain verified"
        except ChainBrokenError as e:
            chain_msg = f"⚠️ chain broken: {e}"
        st.caption(f"{len(records)} record(s) for {turn_id} · {chain_msg}")
    else:
        st.caption(f"No audit records yet for {turn_id}.")


if __name__ == "__main__":
    main()