"""LLM summarizer: fill summary_text / recommended_action / citation_doc_ids
on each incident, grounded in retrieved policy docs.

Provider: Groq Cloud (free tier), OpenAI-compatible chat completions endpoint.
Default model: llama-3.1-8b-instant. No local runtime, no API cost.

Citation guard (validate + one retry, per build decision 2026-07-12):
  1. Parse KB-XXXXX ids out of the model's summary + recommended_action.
  2. Keep only ids that are in the retrieved set; drop any not retrieved.
  3. If zero valid ids survive: re-prompt ONCE with a stricter instruction
     that lists the allowed ids explicitly.
  4. If still zero valid ids: mark the incident summary with a
     'needs_review: no valid citation' note so the operator sees it.
  A smaller free model can invent a KB-XXXXX; the guard catches it.

Auth: reads GROQ_API_KEY from the environment. Set it in your shell:
  PowerShell:  $env:GROQ_API_KEY = "gsk_..."
  (or put it in a local .env you source before running)

CLI:
  python -m project_07_final_synthesis.src.agent.summarizer
  python src/agent/summarizer.py            # from project_07_final_synthesis/
  python src/agent/summarizer.py --incident INC-000001   # one incident
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

_PROJECT = Path(__file__).resolve().parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT))
sys.path.insert(0, str(_REPO_ROOT))
os.chdir(_REPO_ROOT)

import pandas as pd
import requests

from src.utils.io import read_parquet, write_parquet
from src.rag.retriever import retrieve_for_incident

# --- Groq config -------------------------------------------------------------

load_dotenv(_PROJECT / ".env")  # load GROQ_API_KEY from .env if present

# Groq endpoint. Hardcoded default (no stray env var pickup); overridable
# via GROQ_BASE_URL for testing against a different OpenAI-compatible host.
GROQ_URL = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1") + "/chat/completions"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
# Conservative for the free tier. One summary call is well under this.
GROQ_TIMEOUT = 60
# On a 503 / 429 we retry up to this many times with backoff.
GROQ_MAX_RETRIES = 3

# Citation regex: KB- followed by digits. Used to parse + validate model output.
KB_ID_RE = re.compile(r"KB-\d{5}")


def _api_key() -> str:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Get a free key at https://console.groq.com/keys "
            "then:  $env:GROQ_API_KEY = \"gsk_...\""
        )
    return key


def _chat(messages: list[dict], temperature: float = 0.2) -> str:
    """Call Groq chat completions. Returns the assistant text.

    ponytail: plain requests.post. No openai SDK. Retries on 429/5xx with
    exponential backoff; raises on 4xx (auth, bad model name).
    """
    headers = {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
    }
    body = {"model": GROQ_MODEL, "messages": messages, "temperature": temperature}
    last_exc: Exception | None = None
    for attempt in range(GROQ_MAX_RETRIES):
        try:
            resp = requests.post(GROQ_URL, headers=headers, json=body, timeout=GROQ_TIMEOUT)
        except requests.RequestException as e:
            last_exc = e
            time.sleep(2 ** attempt)
            continue
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        if resp.status_code in (429, 500, 502, 503, 504):
            # Transient: back off and retry.
            time.sleep(2 ** attempt)
            last_exc = RuntimeError(f"Groq {resp.status_code}: {resp.text[:200]}")
            continue
        # 4xx (auth, bad request): no point retrying.
        raise RuntimeError(f"Groq {resp.status_code}: {resp.text[:300]}")
    raise RuntimeError(f"Groq call failed after {GROQ_MAX_RETRIES} retries: {last_exc}")


def _build_prompt(incident: dict, docs: list[dict]) -> list[dict]:
    """Build the chat messages: system (role + citation rule) + user (the case).

    The allowed doc_ids are listed EXPLICITLY so the model has no reason to
    invent one. The citation guard still validates the output.
    """
    allowed = ", ".join(d["doc_id"] for d in docs)
    policy_block = "\n\n".join(
        f"[{d['doc_id']}] {d['title']} ({d['category']})\n{d['body']}" for d in docs
    )
    system = (
        "You are a Security Operations Center copilot. You write concise, "
        "analyst-facing incident summaries grounded in the provided policy docs. "
        "Every claim about procedure MUST cite a doc_id from the allowed set "
        f"({allowed}). Do NOT invent or cite any doc_id not in that set. "
        "Output STRICTLY as JSON with keys: summary (2-3 sentences), "
        "recommended_action (one concrete next step), citations (list of doc_id strings)."
    )
    user = (
        f"Incident {incident['incident_id']} ({incident['incident_type']}), "
        f"risk_band={incident['risk_band']}, risk_score={incident['risk_score']}, "
        f"zone={incident['zone_id']}.\n"
        f"Linked events: {incident['linked_event_ids'] or 'none'}; "
        f"linked logs: {incident['linked_log_ids'] or 'none'}.\n\n"
        f"Relevant policies:\n{policy_block}\n\n"
        "Write the JSON now."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _extract_citations(text: str) -> list[str]:
    """Pull KB-XXXXX ids from model output, deduped, order-preserving."""
    return list(dict.fromkeys(KB_ID_RE.findall(text)))


def _parse_json_response(text: str) -> dict:
    """Best-effort JSON parse of the model output. Groq/Llama sometimes
    wraps JSON in markdown fences or adds prose. We grab the first {...}
    block and json.loads it; on failure we fall back to treating the whole
    text as the summary with empty action/citations (the guard catches it).
    """
    # Strip markdown code fences if present.
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    # Fallback: raw text becomes the summary; citations may still parse.
    return {"summary": text.strip(), "recommended_action": "", "citations": _extract_citations(text)}


def _summarize_once(incident: dict, docs: list[dict]) -> dict:
    """One Groq call -> parsed {summary, recommended_action, citations}."""
    msgs = _build_prompt(incident, docs)
    raw = _chat(msgs)
    parsed = _parse_json_response(raw)
    # Always reconcile citations against the full text too (model may put
    # ids in the summary/action but leave the citations field empty).
    cited = set(_extract_citations(parsed.get("summary", "")))
    cited |= set(_extract_citations(parsed.get("recommended_action", "")))
    cited |= set(parsed.get("citations", []) or [])
    parsed["citations"] = list(dict.fromkeys(cited))
    return parsed


def summarize_incident(incident: dict) -> dict:
    """Fill an incident row with summary_text, recommended_action,
    citation_doc_ids. Returns a copy of the incident dict with those
    fields set (and a status field for the audit trail).

    Flow: retrieve -> call -> validate citations -> retry once if empty
    -> fall back to needs_review if still empty.
    """
    docs = retrieve_for_incident(
        incident["incident_type"],
        # Query text: the incident_type + linked ids give the model context.
        f"{incident['incident_type']} in zone {incident['zone_id']}; "
        f"events {incident['linked_event_ids']}; logs {incident['linked_log_ids']}",
        k=3,
    )
    valid_ids = {d["doc_id"] for d in docs}

    def _validate(parsed: dict) -> list[str]:
        return [c for c in parsed.get("citations", []) if c in valid_ids]

    parsed = _summarize_once(incident, docs)
    valid = _validate(parsed)

    if not valid:
        # One retry with a stricter prompt that lists allowed ids up front.
        stricter = dict(incident)
        stricter["_strict_retry"] = True
        # Rebuild with an extra instruction injected into the user message.
        msgs = _build_prompt(incident, docs)
        msgs[1]["content"] += (
            "\n\nIMPORTANT: You cited no valid policy. You MUST cite at least "
            f"one of these ids in the citations list: {', '.join(sorted(valid_ids))}. "
            "Do not output any other doc_id."
        )
        raw = _chat(msgs)
        parsed = _parse_json_response(raw)
        parsed["citations"] = list(dict.fromkeys(
            _extract_citations(parsed.get("summary", ""))
            | set(_extract_citations(parsed.get("recommended_action", "")))
            | set(parsed.get("citations", []) or [])
        ))
        valid = _validate(parsed)

    if not valid:
        # Give up gracefully: surface the failure to the operator.
        return {
            **incident,
            "summary_text": "[needs_review: no valid citation] "
                            + parsed.get("summary", "")[:200],
            "recommended_action": parsed.get("recommended_action", ""),
            "citation_doc_ids": "",
            "_summary_status": "needs_review",
        }

    return {
        **incident,
        "summary_text": parsed.get("summary", "").strip(),
        "recommended_action": parsed.get("recommended_action", "").strip(),
        "citation_doc_ids": ",".join(valid),
        "_summary_status": "ok",
    }


def _load_incidents() -> pd.DataFrame:
    return read_parquet("incidents")


def _row_to_incident(row: pd.Series) -> dict:
    return {
        "incident_id": row["incident_id"],
        "incident_type": row["incident_type"],
        "risk_band": row["risk_band"],
        "risk_score": row["risk_score"],
        "zone_id": row["zone_id"],
        "linked_event_ids": row["linked_event_ids"],
        "linked_log_ids": row["linked_log_ids"],
    }


def main(incident_id: str | None = None, max_n: int | None = None) -> None:
    """Summarize one or all incidents and write back to incidents.parquet.

    ``incident_id`` None -> all incidents. ``max_n`` caps the run to the
    first N rows of the working set (after the incident_id filter);
    used by the notebook to bound Groq latency on the ~226-row scaled
    slice (free-tier cold model ~= a few sec per call). Kept
    notebook-callable: argparse lives only in the __main__ guard so a
    kernel's injected ``-f`` arg doesn't trip SystemExit here.
    """
    full = _load_incidents()
    df = full
    if incident_id:
        df = df[df["incident_id"] == incident_id]
        if df.empty:
            print(f"incident {incident_id} not found in incidents.parquet")
            return
    if max_n is not None and len(df) > max_n:
        df = df.head(max_n)

    rows = []
    for _, r in df.iterrows():
        inc = _row_to_incident(r)
        out = summarize_incident(inc)
        rows.append(out)
        print(f"{out['incident_id']} [{out['_summary_status']}] "
              f"risk={out['risk_band']} \nsummary: {out['summary_text'][:100]} cites={out['citation_doc_ids'] or '-'}")

    # Write back: update ONLY the targeted rows in the full incidents frame.
    # Writing just the filtered df would drop the other 225 rows on the
    # scaled slice; preserve them.
    upd = full.copy()
    for out in rows:
        idx = upd.index[upd["incident_id"] == out["incident_id"]]
        if len(idx) == 0:
            continue
        upd.at[idx[0], "summary_text"] = out["summary_text"]
        upd.at[idx[0], "recommended_action"] = out["recommended_action"]
        upd.at[idx[0], "citation_doc_ids"] = out["citation_doc_ids"]
    write_parquet(upd, "incidents")
    n_ok = sum(1 for o in rows if o["_summary_status"] == "ok")
    n_rev = len(rows) - n_ok
    print(f"wrote incidents with summaries: {n_ok} ok, {n_rev} needs_review")


if __name__ == "__main__":
    import argparse as _ap
    _ap_parser = _ap.ArgumentParser(description="LLM summarizer (Groq, citation-guarded).")
    _ap_parser.add_argument("--incident", default=None, help="Summarize one incident_id; else all.")
    _args = _ap_parser.parse_args()
    main(incident_id=_args.incident)