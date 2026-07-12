"""Rule-based incident detectors.

Each `detect_*` function returns a list of dicts with the shape of a
candidate incident (incident_type, linked events/logs, time window).
The risk_scorer turns these into final INC- rows with risk_score +
risk_band.

Rules spec:
  1. Intrusion in a restricted zone with confidence > 0.85.
  2. Repeated badge denials (>=3 in 1 hour) in any zone.
  3. Surveillance anomaly + access anomaly in the same zone within 10 min.
  4. Tailgating followed by suspicious door activity.

P3 discipline (locked after the smoke-detection audit): rules are the
first layer. ML is a later upgrade. Honest rule thresholds > clever
heuristics. We report what each rule fired on, not a single black-box
"anomaly score".
"""
from __future__ import annotations
from datetime import timedelta
from typing import Iterable

import pandas as pd

# Rule thresholds. Single source of truth: changing these here is the
# only way the rules change. Each is named so the fusion report can
# cite it.
RULE_INTRUSION_CONFIDENCE_MIN = 0.85
RULE_DENIAL_COUNT_MIN = 3
RULE_DENIAL_WINDOW_MIN = 60            # 1 hour
RULE_CORRELATION_WINDOW_MIN = 10       # same-zone cross-anomaly window
RULE_TAILGATE_TO_DOOR_MIN = 10         # tailgate followed by door activity

# Human-readable rule names. Used in summary text + tests.
RULE_NAMES = {
    "intrusion_restricted": "Restricted-Zone Intrusion",
    "repeated_denials": "Repeated Badge Denials",
    "cross_anomaly": "Surveillance + Access Anomaly Correlation",
    "tailgate_door": "Tailgating + Door Activity",
}


# --- Helpers ------------------------------------------------------------------

def _has(df: pd.DataFrame, **cols) -> bool:
    """True iff every named column is present. Avoids KeyError on schema drift."""
    return all(c in df.columns for c in cols.values())


# --- Rule 1: intrusion in a restricted zone, confidence > 0.85 -----------------

def detect_intrusion_restricted(
    events: pd.DataFrame,
    zones: pd.DataFrame,
) -> list[dict]:
    """Surveillance anomaly in a restricted zone, conf above threshold."""
    if not _has(events, event_id="event_id", zone_id="zone_id",
                event_type="event_type", confidence="confidence_score",
                anomaly="anomaly", timestamp="event_timestamp"):
        return []

    restricted = set(zones.loc[zones["restricted"], "zone_id"])
    ev = events[
        events["anomaly"].astype(bool)
        & (events["zone_id"].isin(restricted))
        & (events["confidence_score"].astype(float) >= RULE_INTRUSION_CONFIDENCE_MIN)
    ]
    return [
        {
            "incident_type": "suspected_unauthorized_entry",
            "rule": "intrusion_restricted",
            "linked_event_ids": [row.event_id],
            "linked_log_ids": [],
            "incident_start": row.event_timestamp,
            "incident_end": row.event_timestamp,
            "site_id": row.site_id,
            "zone_id": row.zone_id,
        }
        for row in ev.itertuples(index=False)
    ]


# --- Rule 2: repeated badge denials in a window -------------------------------

def detect_repeated_denials(
    logs: pd.DataFrame,
    window_min: int = RULE_DENIAL_WINDOW_MIN,
    min_count: int = RULE_DENIAL_COUNT_MIN,
) -> list[dict]:
    """>=N denials in the same zone within `window_min` minutes."""
    if not _has(logs, log_id="log_id", zone_id="zone_id",
                timestamp="log_timestamp", result="access_result"):
        return []
    if logs.empty:
        return []

    denied = logs[logs["access_result"] == "denied"].sort_values("log_timestamp")
    if denied.empty:
        return []

    out: list[dict] = []
    window = timedelta(minutes=window_min)
    # Sliding window by zone. Ponytail: O(n*k) where k is window size;
    # fine at vertical-slice scale. For 10k+ rows, switch to a sorted
    # two-pointer sweep.
    for zone_id, group in denied.groupby("zone_id"):
        rows = list(group.itertuples(index=False))
        n = len(rows)
        for i in range(n):
            window_logs = [rows[i]]
            for j in range(i + 1, n):
                if (rows[j].log_timestamp - rows[i].log_timestamp) <= window:
                    window_logs.append(rows[j])
                else:
                    break
            if len(window_logs) >= min_count:
                out.append({
                    "incident_type": "repeated_badge_denials",
                    "rule": "repeated_denials",
                    "linked_event_ids": [],
                    "linked_log_ids": [r.log_id for r in window_logs],
                    "incident_start": min(r.log_timestamp for r in window_logs),
                    "incident_end": max(r.log_timestamp for r in window_logs),
                    "site_id": rows[0].site_id,
                    "zone_id": zone_id,
                })
                # Don't re-emit overlapping windows for the same starting row.
                break
    return out


# --- Rule 3: cross-anomaly correlation (surveillance + access, same zone) -----

def detect_cross_anomaly(
    events: pd.DataFrame,
    logs: pd.DataFrame,
    window_min: int = RULE_CORRELATION_WINDOW_MIN,
) -> list[dict]:
    """Surveillance anomaly AND an unusual access event in the same zone
    within `window_min` minutes. "Unusual access" = denied / invalid / tailgate.
    """
    if not _has(events, anomaly="anomaly", zone_id="zone_id", timestamp="event_timestamp"):
        return []
    if not _has(logs, result="access_result", zone_id="zone_id", timestamp="log_timestamp"):
        return []
    if events.empty or logs.empty:
        return []

    unusual = logs[logs["access_result"].isin(["denied", "invalid", "tailgate"])]
    if unusual.empty:
        return []

    out: list[dict] = []
    window = timedelta(minutes=window_min)
    # Iterate anomalies; for each, find a co-located unusual access.
    for ev in events[events["anomaly"].astype(bool)].itertuples(index=False):
        candidates = unusual[
            (unusual["zone_id"] == ev.zone_id)
            & (unusual["log_timestamp"].between(
                ev.event_timestamp - window, ev.event_timestamp + window))
        ]
        if candidates.empty:
            continue
        cand = candidates.iloc[0]
        out.append({
            "incident_type": "cross_anomaly_correlation",
            "rule": "cross_anomaly",
            "linked_event_ids": [ev.event_id],
            "linked_log_ids": [cand["log_id"]],
            "incident_start": min(ev.event_timestamp, cand["log_timestamp"]),
            "incident_end": max(ev.event_timestamp, cand["log_timestamp"]),
            "site_id": ev.site_id,
            "zone_id": ev.zone_id,
        })
    return out


# --- Rule 4: tailgating followed by door activity -----------------------------

def detect_tailgate_door(
    events: pd.DataFrame,
    logs: pd.DataFrame,
    devices: pd.DataFrame,
    window_min: int = RULE_TAILGATE_TO_DOOR_MIN,
) -> list[dict]:
    """A tailgate log followed (within window) by door-sensor activity
    at a device in the same zone. We use the access log itself as the
    'door activity' proxy (forced_door reason) at vertical-slice scale;
    when door sensors are added, swap to a door_device flag.
    """
    if logs.empty or events.empty:
        return []
    tails = logs[logs["access_result"] == "tailgate"]
    if tails.empty:
        return []
    door_events = logs[logs["reason"] == "forced_door"]
    if door_events.empty:
        return []

    window = timedelta(minutes=window_min)
    out: list[dict] = []
    for t in tails.itertuples(index=False):
        nearby = door_events[
            (door_events["zone_id"] == t.zone_id)
            & (door_events["log_timestamp"].between(t.log_timestamp, t.log_timestamp + window))
        ]
        if nearby.empty:
            continue
        nb = nearby.iloc[0]
        # The injected tailgate row carries reason="forced_door" too, so it
        # shows up in BOTH `tails` and `door_events` -> nb can be the same row
        # as t. Dedupe so the same log_id isn't listed twice in the audit trail.
        # ponytail: dict.fromkeys preserves order while dropping dups; if a
        # separate door sensor is added later, both ids naturally come through.
        linked_log_ids = list(dict.fromkeys([t.log_id, nb["log_id"]]))
        out.append({
            "incident_type": "tailgate_door_activity",
            "rule": "tailgate_door",
            "linked_event_ids": [],
            "linked_log_ids": linked_log_ids,
            "incident_start": min(t.log_timestamp, nb["log_timestamp"]),
            "incident_end": max(t.log_timestamp, nb["log_timestamp"]),
            "site_id": t.site_id,
            "zone_id": t.zone_id,
        })
    return out


# --- Orchestrator -------------------------------------------------------------

def all_candidates(
    events: pd.DataFrame,
    logs: pd.DataFrame,
    zones: pd.DataFrame,
    devices: pd.DataFrame,
) -> list[dict]:
    """Run every detector, return raw candidate dicts (no dedup, no scoring)."""
    candidates: list[dict] = []
    candidates.extend(detect_intrusion_restricted(events, zones))
    candidates.extend(detect_repeated_denials(logs))
    candidates.extend(detect_cross_anomaly(events, logs))
    candidates.extend(detect_tailgate_door(events, logs, devices))
    return candidates
