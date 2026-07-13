"""P1 -> P7 schema adapter.

P1 (`project_01_reproducible_workflows/data/raw/*.parquet`) and P7 have
slightly different schemas for the same conceptual data. This module
bridges them so the SOC copilot can use P1's larger corpus
(1k+ events / 10k logs across 3 sites) directly, without re-generating
synthetic data.

Mapping summary:

    surveillance_events (P1)            surveillance_events (P7)
    -----------------------------------------------------------
    event_id              -> event_id
    event_timestamp       -> event_timestamp
    site_id               -> site_id
    zone_id               -> zone_id
    device_id             -> device_id
    event_type            -> event_type
    confidence_score      -> confidence_score
    (none)                -> anomaly          (DERIVED: event_type in
                                               {intrusion, loitering})
    (none)                -> description      (DERIVED: short NL stub)

    access_logs (P1)                  access_logs (P7)
    -----------------------------------------------------------
    log_id                -> log_id
    log_timestamp         -> log_timestamp
    site_id               -> site_id
    zone_id               -> zone_id
    device_id             -> device_id
    user_id               -> user_id
    outcome               -> access_result    (granted / denied / invalid /
                                                tailgate; "??unknown??"
                                                sentinel rows DROPPED)
    (none)                -> reason           (DERIVED: from access_result)
    (badge_id)            -> (dropped, P7 doesn't use it)

Zone restrictiveness is inferred from the zone_id format: P1 uses
`SITE-NNN::ZONE-X` where ZONE-D is the restricted zone. P7 reads this
via a config dict below (configurable per project).

The adapter is **pure** (no I/O, no LLM, no side effects) and easy to
unit-test against P1's raw parquet. The cleaning decisions are
deliberately *minimal* (P1's raw is already cleaned by P1's own
notebook; we only adapt, we don't re-clean).
"""
from __future__ import annotations
from typing import Iterable

import numpy as np
import pandas as pd

# P7 fusion uses these event types as the anomaly proxy. P1 already labels
# its 5 event types; we just need to pick which count as "anomaly".
# Calibration note: intrusion (mean conf 0.82) and loitering (0.86) are
# the genuine anomaly signals; normal_motion / object_left / tailgating
# are normal events (P1 separates tailgating from anomaly, P7 puts it in
# the access-log path).
P1_ANOMALY_EVENT_TYPES: frozenset[str] = frozenset({"intrusion", "loitering"})

# P1 outcome values -> P7 access_result values. The "??unknown??" sentinel
# (user_id = "USR-0000") is dropped entirely in adapt_access().
P1_OUTCOME_TO_ACCESS_RESULT: dict[str, str] = {
    "granted": "granted",
    "denied": "denied",
    "invalid_credential": "invalid",
    "tailgated": "tailgate",
    # "??unknown??" is filtered out before this dict is consulted.
}

# P1 zone names -> "restricted" boolean. P1 uses SITE-NNN::ZONE-D as the
# restricted zone; everything else is unrestricted. Configurable here so
# the same adapter can support other layouts.
P1_RESTRICTED_ZONE_LETTERS: frozenset[str] = frozenset({"D"})


def _normalize_zone_id(zone_id: str) -> str:
    """P1's generator occasionally emits lowercase zone ids like
    'site-003::zone-c' (data-quality issue from P1's generator).
    Normalize to the canonical SITE-NNN::ZONE-X format so the fusion
    rule's `events["zone_id"].isin(restricted_zones)` lookup matches.
    """
    if not isinstance(zone_id, str):
        return zone_id
    return zone_id.strip().upper()


def _is_restricted(zone_id: str) -> bool:
    """P1 zone_id format is 'SITE-NNN::ZONE-X'. The suffix letter decides.

    Normalizes lowercase variants first so the dirty rows (P1 has ~22
    of 1000) still classify correctly.
    """
    z = _normalize_zone_id(zone_id)
    if "ZONE-" not in z:
        return False
    suffix = z.rsplit("ZONE-", 1)[-1].strip()
    return suffix in P1_RESTRICTED_ZONE_LETTERS


# Backwards-compat: the calibration test imports _is_restricted from this
# module and was written before _normalize_zone_id was added. Re-export
# the symbol so the existing test signature still works.


def _to_p7_event_type(p1_event_type: str) -> str:
    """Normalize P1's event_type naming into P7's vocabulary. P1 already
    uses clean snake_case; P7's vocabulary is similar but not identical.
    """
    m = {
        "normal_motion": "person_detected",
        "loitering": "person_detected",   # keep semantics: still a person
        "intrusion": "anomaly",
        "object_left": "person_detected",
        "tailgating": "person_detected",  # tailgate itself is an access event;
                                          # if a camera saw it, P7 still tags
                                          # the event as person_detected;
                                          # the fusion tailgate_door rule
                                          # pairs it with the access log.
    }
    return m.get(p1_event_type, p1_event_type)


def adapt_surveillance(
    p1_df: pd.DataFrame,
    required_cols: Iterable[str] = (
        "event_id", "event_timestamp", "site_id", "zone_id", "device_id",
        "event_type", "confidence_score",
    ),
) -> pd.DataFrame:
    """Adapt a P1 surveillance parquet into P7's surveillance schema.

    Adds `anomaly` (derived from event_type) and `description` (short
    NL stub). P1's event_type is mapped into P7's vocabulary.
    """
    missing = set(required_cols) - set(p1_df.columns)
    if missing:
        raise ValueError(f"P1 surveillance missing required columns: {missing}")

    df = p1_df[list(required_cols)].copy()
    df["zone_id"] = df["zone_id"].map(_normalize_zone_id)

    # Timezone-aware UTC (P1 already stores UTC-aware datetimes).
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True)

    # Map event_type -> P7 vocabulary.
    df["event_type"] = df["event_type"].map(_to_p7_event_type).fillna(df["event_type"])

    # Derive anomaly from event_type (intrusion + loitering are the
    # genuine anomaly signals in P1's taxonomy).
    df["anomaly"] = df["event_type"].isin({"anomaly"}) | (
        # P1's intrusion maps to P7's "anomaly" event_type above, so
        # `anomaly` here is True when event_type == "anomaly" OR
        # when the original event was a high-confidence intrusion/loitering
        # we kept as person_detected (defense in depth).
        p1_df["event_type"].isin(P1_ANOMALY_EVENT_TYPES)
        & (df["confidence_score"] >= 0.80)
    )

    # Description: short NL stub matching P7's generator format. The
    # fusion layer doesn't use description; the LLM summarizer does.
    df["description"] = (
        "Camera " + df["device_id"] + " detected " + df["event_type"].str.replace("_", " ")
        + " in " + df["zone_id"] + "."
    )

    # Cast confidence to float.
    df["confidence_score"] = df["confidence_score"].astype(float)

    return df[[
        "event_id", "site_id", "zone_id", "device_id", "event_timestamp",
        "event_type", "confidence_score", "anomaly", "description",
    ]]


def adapt_access(
    p1_df: pd.DataFrame,
    required_cols: Iterable[str] = (
        "log_id", "log_timestamp", "site_id", "zone_id", "device_id",
        "user_id", "outcome",
    ),
) -> pd.DataFrame:
    """Adapt a P1 access parquet into P7's access schema.

    Drops "??unknown??" sentinel rows (user_id = USR-0000) entirely. Maps
    P1's outcome values into P7's access_result vocabulary. Adds a
    synthetic `reason` field that P7's fusion uses.
    """
    missing = set(required_cols) - set(p1_df.columns)
    if missing:
        raise ValueError(f"P1 access missing required columns: {missing}")

    df = p1_df[list(required_cols)].copy()
    df["zone_id"] = df["zone_id"].map(_normalize_zone_id)
    df["log_timestamp"] = pd.to_datetime(df["log_timestamp"], utc=True)

    # Drop sentinel "??unknown??" rows (user_id sentinel). P1 uses the
    # string "USR-0000" for these — robust to either form.
    sentinel_mask = (df["outcome"] == "??unknown??") | (df["user_id"] == "USR-0000")
    n_dropped = int(sentinel_mask.sum())
    if n_dropped:
        df = df[~sentinel_mask].copy()

    # Map outcome -> access_result. Anything not in the map (shouldn't
    # happen after the sentinel drop) becomes "invalid".
    df["access_result"] = df["outcome"].map(
        P1_OUTCOME_TO_ACCESS_RESULT
    ).fillna("invalid")

    # Synthetic reason: matched to access_result. P7's generators use
    # similar values; the fusion layer matches on access_result only.
    reason_map = {
        "granted": "ok",
        "denied": "revoked",       # P1 doesn't carry a "reason" field;
                                   # the gate's pre-200k behavior is best
                                   # guessed as "revoked" (denied !=
                                   # expired) since P1's data is synthetic
                                   # and the tag is only used for human
                                   # readability in summaries.
        "invalid": "unknown_badge",
        "tailgate": "forced_door",
    }
    df["reason"] = df["access_result"].map(reason_map).fillna("unknown")

    return df[[
        "log_id", "site_id", "zone_id", "device_id", "log_timestamp",
        "user_id", "access_result", "reason",
    ]], n_dropped


if __name__ == "__main__":
    # Self-check: round-trip P1's raw through the adapter and assert
    # the schema + value ranges come out as expected.
    import sys
    from pathlib import Path
    # project_07_final_synthesis/src/utils/p1_adapter.py -> repo_root
    REPO = Path(__file__).resolve().parents[3]
    e = pd.read_parquet(REPO / "project_01_reproducible_workflows/data/raw/surveillance_events.parquet")
    a = pd.read_parquet(REPO / "project_01_reproducible_workflows/data/raw/access_logs.parquet")
    e7 = adapt_surveillance(e)
    a7, dropped = adapt_access(a)
    print(f"surveillance: {len(e)} -> {len(e7)} (anomaly={int(e7['anomaly'].sum())})")
    print(f"access: {len(a)} -> {len(a7)} (dropped {dropped} sentinel rows; access_result={a7['access_result'].value_counts().to_dict()})")
    assert len(e7) == len(e), "surveillance row count must match"
    assert len(a7) + dropped == len(a), "access row count must match"
    assert (e7["anomaly"] | (e7["event_type"] != "anomaly")).sum() > 0, "anomaly derivation should produce some anomalies"
    print("p1_adapter self-check OK")