"""Streamlit GUI for the SOC Security Operations Copilot.

One-file app. Reuses `src.agent.copilot_agent` directly — no new abstraction,
no separate API. Analysts pick an incident, run the agent, and see the
summary / recommended action / KB citations / escalation verdict plus the
hash-chained audit trail for that turn.

Two data sources, selectable per session:
  1. Default: the project's baked `data/synthetic/incidents.parquet` (226
     incidents from the P1 corpus). Reproducible, no upload needed.
  2. Optional: user-uploaded surveillance + access CSVs, adapted into
     the project schema via `src.utils.upload_adapter`, run through the
     same fusion layer, and the resulting per-session incidents are
     triaged through the same copilot. The project parquet is never
     touched; uploads live in `data/uploads/{uuid}/`.

Human-in-the-loop: when a step requires human approval (critical-band
escalation), the agent pauses, the GUI shows the pending action and a
Grant / Deny choice, and the analyst's decision is logged to the audit
chain. This replaces the "demo auto-approves" path in the CLI/notebook
while keeping that path intact for headless use.

Run from repo root:
    project_07_final_synthesis\\final_venv\\Scripts\\python.exe -m \\
        streamlit run project_07_final_synthesis/src/gui/app.py

Default is deterministic stub mode (no Groq, reproducible). Tick "Use LLM"
to call Groq llama-3.1-8b-instant — the typed key is the only source;
the GUI never reads .env.
"""
from __future__ import annotations
import sys
import uuid
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
from src.utils.upload_adapter import (
    P7_ACCESS_REASONS,
    P7_ACCESS_RESULTS,
    P7_EVENT_TYPES,
    adapt_uploaded_access,
    adapt_uploaded_surveillance,
)
from src.agent.copilot_agent import run_incident_streaming

AUDIT_PATH = _PROJECT / "data" / "audit.jsonl"
UPLOADS_DIR = _PROJECT / "data" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# risk_band -> display color for the status badge.
BAND_COLOR = {"critical": "red", "high": "orange",
              "medium": "blue", "low": "green"}

# Per-run session-state keys. Reset whenever a new run starts.
_SS_RUN = "_gui_run_state"           # dict: {phase, last_state, pending, granted, summarize_payload}
_SS_HUMAN_DECIDED = "_gui_human_decided"  # bool
_SS_HUMAN_GRANT = "_gui_human_grant"      # bool
_SS_GEN = "_gui_gen"                      # the long-lived streaming generator
_SS_GEN_KEY = "_gui_gen_key"              # (incident_id, use_llm) the gen was created for
# Per-session upload state.
_SS_UPLOAD_DIR = "_gui_upload_dir"        # Path to data/uploads/{uuid}/ when an upload is active
_SS_UPLOAD_INCIDENTS = "_gui_upload_incidents_parquet"  # Path to the per-session incidents.parquet
_SS_UPLOAD_SV_PATH = "_gui_upload_sv_path"  # Path to the saved surveillance CSV
_SS_UPLOAD_AC_PATH = "_gui_upload_ac_path"  # Path to the saved access-log CSV
_SS_UPLOAD_STATS = "_gui_upload_stats"    # dict with row counts + drop reasons
_SS_UPLOAD_ACTIVE = "_gui_upload_active"  # bool: is the GUI currently driving on uploaded data?


@st.cache_data(show_spinner=False)
def load_incidents_default() -> pd.DataFrame:
    return read_parquet("incidents")


def load_incidents() -> pd.DataFrame:
    """Load incidents from the active session's upload dir if present,
    else from the project's synthetic/ directory. The upload path is
    bypassed (and a fresh read happens) when the upload parquet is
    regenerated, so the dropdown always reflects the latest build."""
    upload_path = st.session_state.get(_SS_UPLOAD_INCIDENTS)
    if upload_path and Path(upload_path).exists():
        return pd.read_parquet(upload_path)
    return load_incidents_default()


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


# --- Upload helpers --------------------------------------------------------

def _column_mapping_editor(
    user_cols: list[str],
    choices: list[str],
    session_key: str,
) -> dict[str, str]:
    """Render a 2-column mapping editor: each user column maps to a
    project field (or "" to drop it). Returns the mapping dict.

    `choices` is the full list of project-side fields the user can map
    to (required + optional)."""
    rows = [{"user column": c, "project field": ""} for c in user_cols]
    df = pd.DataFrame(rows)
    if session_key in st.session_state:
        prev = st.session_state[session_key]
        for i, c in enumerate(user_cols):
            if c in prev:
                df.at[i, "project field"] = prev[c] or ""
    edited = st.data_editor(
        df,
        column_config={
            "user column": st.column_config.TextColumn("Your column", disabled=True),
            "project field": st.column_config.SelectboxColumn(
                "Project field",
                options=[""] + choices,
                required=False,
                help="Pick a project field for this column, or leave blank to drop it.",
            ),
        },
        hide_index=True,
        key=session_key,
        width="stretch",
    )
    out: dict[str, str] = {}
    for _, r in edited.iterrows():
        if r["project field"]:
            out[r["user column"]] = r["project field"]
    return out


def _value_mapping_editor(
    user_values: list[str],
    legal_targets: tuple[str, ...],
    session_key: str,
    label: str,
) -> dict[str, str]:
    """For each user-side enum value, the user picks a project-side value.
    Returns {user_value: project_value}."""
    if not user_values:
        st.caption(f"(no {label} values to map)")
        return {}
    rows = [{"user value": v, "project value": ""} for v in user_values]
    df = pd.DataFrame(rows)
    if session_key in st.session_state:
        prev = st.session_state[session_key]
        for i, v in enumerate(user_values):
            if v in prev:
                df.at[i, "project value"] = prev[v] or ""
    edited = st.data_editor(
        df,
        column_config={
            "user value": st.column_config.TextColumn("User value", disabled=True),
            "project value": st.column_config.SelectboxColumn(
                f"{label} → project value",
                options=[""] + list(legal_targets),
                required=False,
                help=f"Map to one of {list(legal_targets)}, or leave blank "
                     f"to drop rows with this value.",
            ),
        },
        hide_index=True,
        key=session_key,
        width="stretch",
    )
    out: dict[str, str] = {}
    for _, r in edited.iterrows():
        if r["project value"]:
            out[r["user value"]] = r["project value"]
    return out


def _save_uploaded_csv(uploaded_file, dest_path: Path) -> None:
    """Read an UploadedFile via pandas and write it to disk at dest_path.
    Using a file on disk lets the fusion layer's read paths work
    without changes."""
    df = pd.read_csv(uploaded_file)
    df.to_csv(dest_path, index=False, encoding="utf-8")


def _adapt_and_build(
    sv_path: Path,
    ac_path: Path,
    column_map_sv: dict[str, str],
    column_map_ac: dict[str, str],
    value_map_sv: dict[str, str],
    value_map_ac: dict[str, str],
) -> tuple[pd.DataFrame, dict, list[str]]:
    """Run the upload adapter on the two saved CSVs, then call
    build_incidents to materialize the incidents. Returns
    (incidents_df, combined_stats, combined_reasons)."""
    from src.fusion.incidents import build_incidents
    sv_raw = pd.read_csv(sv_path)
    ac_raw = pd.read_csv(ac_path)
    sv_df, sv_counts, sv_reasons = adapt_uploaded_surveillance(
        sv_raw, column_map_sv, {"event_type": value_map_sv})
    ac_df, ac_counts, ac_reasons = adapt_uploaded_access(
        ac_raw, column_map_ac,
        {"access_result": value_map_ac, "reason": {}})
    # Reference data the fusion rules need.
    # Reference data the fusion rules need. Lives at the **repo root**
    # data/reference/ (not inside the project subdir); the
    # `src/generators/reference_data.py` writer is anchored to whatever
    # cwd it runs from, which is the repo root in the canonical
    # `python -m project_07_final_synthesis.src.generators.run_all` flow.
    sites = pd.read_csv(_REPO_ROOT / "data" / "reference" / "sites.csv")
    zones = pd.read_csv(_REPO_ROOT / "data" / "reference" / "zones.csv")
    devices = pd.read_csv(_REPO_ROOT / "data" / "reference" / "devices.csv")
    reasons = list(sv_reasons) + list(ac_reasons)
    if sv_df.empty:
        return (pd.DataFrame(),
                {"surveillance": sv_counts, "access": ac_counts,
                 "incidents": 0},
                reasons + ["no surveillance rows survived validation"])
    incidents = build_incidents(sv_df, ac_df, zones, devices)
    stats = {"surveillance": sv_counts, "access": ac_counts,
             "incidents": int(len(incidents))}
    return incidents, stats, reasons


def _render_upload_section() -> bool:
    """Render the upload section in the sidebar. Returns True if an
    upload is active (so the main view can pass incidents_dir through
    to the streaming call)."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Upload your data (optional)")
    sv_file = st.sidebar.file_uploader(
        "Surveillance events CSV", type="csv",
        key="upload_sv", help="Camera / sensor events. Any schema; "
        "you'll map the columns below.")
    ac_file = st.sidebar.file_uploader(
        "Access logs CSV", type="csv",
        key="upload_ac", help="Badge reader events. Any schema; "
        "you'll map the columns below.")
    if not (sv_file and ac_file):
        # No upload yet — clear any stale state so the dropdown reverts
        # to the project data.
        for k in (_SS_UPLOAD_DIR, _SS_UPLOAD_INCIDENTS,
                  _SS_UPLOAD_SV_PATH, _SS_UPLOAD_AC_PATH,
                  _SS_UPLOAD_STATS, _SS_UPLOAD_ACTIVE):
            st.session_state.pop(k, None)
        st.sidebar.caption("Upload both files to run the copilot on "
                           "your own data. The project's baked parquet "
                           "is preserved.")
        return False

    # Persist the uploads to disk in a per-session subdir.
    upload_dir = st.session_state.get(_SS_UPLOAD_DIR)
    if not upload_dir:
        upload_dir = UPLOADS_DIR / f"session-{uuid.uuid4().hex[:8]}"
        upload_dir.mkdir(parents=True, exist_ok=True)
        st.session_state[_SS_UPLOAD_DIR] = str(upload_dir)
    upload_dir = Path(upload_dir)  # coerce str -> Path before any / operations
    sv_path = upload_dir / "surveillance.csv"
    ac_path = upload_dir / "access_logs.csv"
    # Always re-write on a fresh upload so the on-disk file matches
    # what the user just picked (Streamlit's UploadedFile carries the
    # fresh bytes each rerun).
    _save_uploaded_csv(sv_file, sv_path)
    _save_uploaded_csv(ac_file, ac_path)
    st.session_state[_SS_UPLOAD_SV_PATH] = str(sv_path)
    st.session_state[_SS_UPLOAD_AC_PATH] = str(ac_path)
    sv_df_raw = pd.read_csv(sv_path)
    ac_df_raw = pd.read_csv(ac_path)

    st.sidebar.caption(
        f"Surveillance: **{len(sv_df_raw)}** rows · "
        f"Access: **{len(ac_df_raw)}** rows.")

    # --- Column mapping step ---------------------------------------------
    sv_required = ["site_id", "zone_id", "event_timestamp",
                   "event_type", "confidence_score"]
    sv_optional = ["event_id", "device_id", "description"]
    ac_required = ["site_id", "zone_id", "log_timestamp",
                   "access_result", "user_id"]
    ac_optional = ["log_id", "device_id", "reason"]
    with st.sidebar.expander("1️⃣ Map columns", expanded=True):
        st.caption("Map your CSV columns to project fields. Required fields "
                   "are listed first; optional fields can be left blank.")
        st.markdown("**Surveillance**")
        sv_map = _column_mapping_editor(
            list(sv_df_raw.columns),
            sv_required + sv_optional,
            session_key="_col_map_sv",
        )
        st.markdown("**Access logs**")
        ac_map = _column_mapping_editor(
            list(ac_df_raw.columns),
            ac_required + ac_optional,
            session_key="_col_map_ac",
        )

    # --- Value mapping step ---------------------------------------------
    with st.sidebar.expander("2️⃣ Map enum values", expanded=False):
        st.caption("For each user-side value, pick the project-side value. "
                   "Leave blank to drop rows with that value.")
        # event_type: only meaningful if the user mapped a column to it.
        sv_event_col = next(
            (u for u, p in sv_map.items() if p == "event_type"), None)
        sv_event_values = sorted(
            sv_df_raw[sv_event_col].dropna().astype(str).unique().tolist()
        ) if sv_event_col else []
        st.markdown("**event_type values**")
        sv_val_map = _value_mapping_editor(
            sv_event_values, P7_EVENT_TYPES,
            session_key="_val_map_sv_evt", label="event_type")
        # access_result
        ac_ar_col = next(
            (u for u, p in ac_map.items() if p == "access_result"), None)
        ac_ar_values = sorted(
            ac_df_raw[ac_ar_col].dropna().astype(str).unique().tolist()
        ) if ac_ar_col else []
        st.markdown("**access_result values**")
        ac_val_map = _value_mapping_editor(
            ac_ar_values, P7_ACCESS_RESULTS,
            session_key="_val_map_ac_ar", label="access_result")

    # --- Build button + result summary -----------------------------------
    if st.sidebar.button("🔨 Build incidents from upload", key="build_btn"):
        try:
            incidents, stats, reasons = _adapt_and_build(
                sv_path, ac_path, sv_map, ac_map, sv_val_map, ac_val_map)
        except Exception as e:
            st.sidebar.error(f"Build failed: {e}")
            return False
        if incidents.empty:
            st.sidebar.warning(
                "No incidents survived validation. "
                "Check the column + value mappings below.")
            st.session_state[_SS_UPLOAD_STATS] = {
                **stats, "reasons": reasons}
            st.session_state[_SS_UPLOAD_ACTIVE] = False
            return False
        incidents_path = Path(st.session_state[_SS_UPLOAD_DIR]) / "incidents.parquet"
        incidents.to_parquet(incidents_path, index=False)
        st.session_state[_SS_UPLOAD_INCIDENTS] = str(incidents_path)
        st.session_state[_SS_UPLOAD_STATS] = {
            **stats, "reasons": reasons}
        st.session_state[_SS_UPLOAD_ACTIVE] = True
        st.sidebar.success(
            f"Built {stats['incidents']} incident(s) from upload. "
            "Switch incident in the dropdown to triage.")
    elif _SS_UPLOAD_INCIDENTS in st.session_state and Path(
            st.session_state[_SS_UPLOAD_INCIDENTS]).exists():
        st.session_state[_SS_UPLOAD_ACTIVE] = True
        st.sidebar.caption("✅ Upload active. "
                           f"Incidents: {Path(st.session_state[_SS_UPLOAD_INCIDENTS]).name}")

    # Surface drop reasons (if any) from the last build.
    stats = st.session_state.get(_SS_UPLOAD_STATS) or {}
    for r in stats.get("reasons", []) or []:
        st.sidebar.caption(f"· {r}")

    return bool(st.session_state.get(_SS_UPLOAD_ACTIVE))


# --- Main -----------------------------------------------------------------

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

    # Sidebar — upload section first (so the user can switch data sources
    # before picking the incident), then the triage controls.
    upload_active = _render_upload_section()
    # If the active upload just changed, re-read incidents before the
    # dropdown renders so the list is in sync.
    if upload_active and _SS_UPLOAD_INCIDENTS in st.session_state:
        df = load_incidents()

    with st.sidebar:
        st.header("Triage controls")
        ids = sorted(df["incident_id"].tolist())
        if not ids:
            st.error("No incidents to triage. Upload a CSV or run the "
                     "project pipeline to populate incidents.")
            return
        incident_id = st.selectbox("Incident", ids, index=0,
                                   key="incident_id")
        use_llm = st.checkbox(
            "Use LLM (Groq llama-3.1-8b-instant)", value=False,
            help="Off = deterministic stub mode (reproducible, no API key). "
                 "On = calls Groq using the key you type below. The .env "
                 "file is NOT consulted in the GUI; the typed key is the "
                 "only source.",
            key="use_llm",
        )
        # When Use LLM is on, show a password field for the Groq API key.
        # The typed key is the ONLY source — the GUI never reads .env.
        # The key is held in session state, never written to disk, and
        # never logged. If the field is empty, we refuse the LLM run.
        user_api_key: str | None = None
        if use_llm:
            user_api_key = st.text_input(
                "Groq API key",
                value="",
                type="password",
                key="user_api_key",
                placeholder="paste gsk_...",
                help="Free key at https://console.groq.com/keys. "
                     "Stored only in this session; not written to .env.",
            )
            if user_api_key:
                st.caption("✅ Using the key you typed.")
            else:
                st.warning("No key typed — LLM run will not start. "
                           "Tick the checkbox off for stub mode, or type a key.")
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
    if upload_active:
        st.caption("📁 Data source: your upload "
                   f"({st.session_state.get(_SS_UPLOAD_DIR)})")

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
    # (incident_id, use_llm, incidents_dir) so a run for a different
    # incident / data source / mode can't pick up a leftover pending
    # event from a previous suspended run. `incidents_dir` is the
    # directory containing `incidents.parquet` (what read_parquet
    # expects as `in_dir`); the full file path is in
    # _SS_UPLOAD_INCIDENTS so we take its parent.
    incidents_file = (st.session_state.get(_SS_UPLOAD_INCIDENTS)
                      if st.session_state.get(_SS_UPLOAD_ACTIVE) else None)
    incidents_dir = (str(Path(incidents_file).parent)
                     if incidents_file else None)
    gen_key = (incident_id, use_llm, incidents_dir)
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

        # Refuse the LLM run if the user ticked Use LLM but didn't type
        # a key. The GUI never reads .env, so an empty field means we
        # have no key at all — better to fail loudly here than to send
        # an empty Authorization header to Groq.
        if use_llm and not (user_api_key and user_api_key.strip()):
            st.error("Use LLM is on but no Groq API key was typed. "
                     "Tick 'Use LLM' off for stub mode, or type a key.")
            st.session_state.pop(_SS_GEN, None)
            return

        gen = run_incident_streaming(
            incident_id, use_llm=use_llm,
            on_human_approval=_on_human_approval,
            api_key=user_api_key,
            incidents_dir=incidents_dir,
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