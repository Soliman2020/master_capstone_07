"""P7 vertical-slice schema. The spec calls for a SQLite CREATE TABLE set;
for the slice we keep schemas in Python dataclasses/pandas-dtypes and
materialize to Parquet. SQLite is added when the agent needs persistent
state across runs (P7 stretch: incident_case table).

ID prefixes:
  SITE-001, ZONE-003, DEV-014, USR-102,
  EVT-000245, LOG-001992, INC-000041, KB-00012
"""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


# --- Reference tables ---------------------------------------------------------

@dataclass(frozen=True)
class Site:
    site_id: str          # "SITE-001"
    site_name: str        # "HQ-West"
    timezone: str         # "UTC"


@dataclass(frozen=True)
class Zone:
    zone_id: str          # "ZONE-001"
    site_id: str
    zone_name: str
    restricted: bool      # restricted zones trigger the intrusion rule


@dataclass(frozen=True)
class Device:
    device_id: str        # "DEV-001"
    site_id: str
    zone_id: str
    device_type: str      # "camera" | "badge_reader" | "door"


@dataclass(frozen=True)
class User:
    user_id: str          # "USR-001"
    site_id: str
    role: str             # "employee" | "contractor" | "cleaner" | "security"
    authorized_zones: tuple[str, ...]  # which zone_ids this user can badge into


# --- Operational tables -------------------------------------------------------

# Surveillance events: camera detections / anomalies.
# anomaly=True means the camera model flagged the frame (e.g. person in
# restricted zone after hours). anomaly_rate is the seed-controlled knob.
SURVEILLANCE_EVENTS_COLS = [
    "event_id",        # "EVT-000001"
    "site_id",
    "zone_id",
    "device_id",
    "event_timestamp", # tz-aware UTC
    "event_type",      # "person_detected" | "vehicle_detected" | "anomaly"
    "confidence_score",# [0,1]
    "anomaly",         # bool
    "description",     # short NL text (used by RAG summary)
]

# Access logs: badge reader events at doors.
ACCESS_LOGS_COLS = [
    "log_id",          # "LOG-000001"
    "site_id",
    "zone_id",
    "device_id",       # the badge reader's device_id
    "log_timestamp",
    "user_id",         # can be None for unknown badges
    "access_result",   # "granted" | "denied" | "invalid" | "tailgate"
    "reason",          # "expired" | "wrong_zone" | "revoked" | "unknown_badge" | "forced_door"
]

# Incidents: the output of the fusion layer. Links surveillance events
# and access logs that together tripped a rule.
INCIDENTS_COLS = [
    "incident_id",     # "INC-000001"
    "site_id",
    "zone_id",
    "incident_start",
    "incident_end",
    "incident_type",   # "suspected_unauthorized_entry" | "repeated_badge_denials" | ...
    "linked_event_ids",# list[str], serialized as "EVT-001,EVT-002"
    "linked_log_ids",  # list[str]
    "risk_score",      # [0, 100]
    "risk_band",       # "low" | "medium" | "high" | "critical"
    "summary_text",    # populated by RAG+LLM
    "recommended_action",
    "citation_doc_ids",# list[str] (KB-XXXXX)
    "human_review_required",  # bool, True for critical
    "created_at",
]


# --- Helpers ------------------------------------------------------------------

def empty_surveillance() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in SURVEILLANCE_EVENTS_COLS})


def empty_access_logs() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in ACCESS_LOGS_COLS})


def empty_incidents() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in INCIDENTS_COLS})
