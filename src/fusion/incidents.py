"""Incident generator: turns rule candidates into final incident rows.

Pipeline:
  1. Run `rules.all_candidates` over the events + logs.
  2. Deduplicate (same (rule, sorted linked_ids) tuple).
  3. Score each candidate via `risk_scorer.score_candidate`.
  4. Materialize to the INCIDENTS_COLS schema and write Parquet.

The summary_text / recommended_action / citation_doc_ids columns are
left empty here. The agent's `summarize_incident` tool fills them
later (RAG + LLM step). The fusion layer is intentionally
language-agnostic so it stays testable without an LLM.
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from project_07_final_synthesis.src.schema import (
    INCIDENTS_COLS,
    empty_incidents,
)
from project_07_final_synthesis.src.utils.constants import SEED
from project_07_final_synthesis.src.utils.io import write_parquet

from project_07_final_synthesis.src.fusion.rules import all_candidates
from project_07_final_synthesis.src.fusion.risk_scorer import score_candidate


def _dedup_key(c: dict) -> tuple:
    """Two candidates are 'the same' if they have the same rule and
    the same set of linked events+logs. We sort the linked ids so the
    order doesn't matter.
    """
    return (
        c["rule"],
        tuple(sorted(c.get("linked_event_ids", []))),
        tuple(sorted(c.get("linked_log_ids", []))),
    )


def _incident_id(i: int) -> str:
    return f"INC-{i+1:06d}"


def build_incidents(
    events: pd.DataFrame,
    logs: pd.DataFrame,
    zones: pd.DataFrame,
    devices: pd.DataFrame,
) -> pd.DataFrame:
    """Run rules + scoring + dedup; return a DataFrame in INCIDENTS_COLS.

    Empty summary/recommendation/citation columns are filled later by
    the agent's RAG+LLM step.
    """
    raw = all_candidates(events, logs, zones, devices)
    # Dedup.
    seen: set[tuple] = set()
    deduped: list[dict] = []
    for c in raw:
        k = _dedup_key(c)
        if k in seen:
            continue
        seen.add(k)
        deduped.append(c)

    # Lookup for confidence bonus.
    events_lookup = dict(zip(events["event_id"], events["confidence_score"].astype(float)))

    # Score.
    scored = [score_candidate(c, events_lookup) for c in deduped]

    # Materialize.
    now = datetime.now(timezone.utc)
    rows = []
    for i, c in enumerate(scored):
        rows.append({
            "incident_id": _incident_id(i),
            "site_id": c["site_id"],
            "zone_id": c["zone_id"],
            "incident_start": c["incident_start"],
            "incident_end": c["incident_end"],
            "incident_type": c["incident_type"],
            "linked_event_ids": ",".join(c.get("linked_event_ids", [])),
            "linked_log_ids": ",".join(c.get("linked_log_ids", [])),
            "risk_score": int(c["risk_score"]),
            "risk_band": c["risk_band"],
            "summary_text": "",          # filled by summarize_incident
            "recommended_action": "",    # filled by summarize_incident
            "citation_doc_ids": "",      # filled by summarize_incident
            "human_review_required": c["risk_band"] == "critical",
            "created_at": now,
            # Extra column (not in the strict schema list, but useful for
            # audit) -- store the originating rule.
            "_rule": c["rule"],
        })

    df = empty_incidents()
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df["risk_score"] = df["risk_score"].astype(int)
    df["human_review_required"] = df["human_review_required"].astype(bool)
    df["incident_start"] = pd.to_datetime(df["incident_start"], utc=True)
    df["incident_end"] = pd.to_datetime(df["incident_end"], utc=True)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    return df


def main() -> Path:
    sites = pd.read_csv("data/reference/sites.csv")
    zones = pd.read_csv("data/reference/zones.csv")
    devices = pd.read_csv("data/reference/devices.csv")
    events = pd.read_parquet("project_07_final_synthesis/data/synthetic/surveillance_events.parquet")
    logs = pd.read_parquet("project_07_final_synthesis/data/synthetic/access_logs.parquet")

    df = build_incidents(events, logs, zones, devices)
    path = write_parquet(df, "incidents")
    by_band = df["risk_band"].value_counts().to_dict() if not df.empty else {}
    crit = int(df["human_review_required"].sum()) if not df.empty else 0
    print(f"wrote {path}  rows={len(df)} by_band={by_band} critical={crit}")
    return path


if __name__ == "__main__":
    main()
