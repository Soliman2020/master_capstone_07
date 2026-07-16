"""Adapter for user-uploaded CSVs.

Mirrors `p1_adapter.py`: pure functions, no I/O, no LLM. Takes a
user-uploaded DataFrame (whatever shape their cameras / badge systems
export), applies:

  1. column_map  — rename `user_col` to `project_col`
                    (e.g. {"camera_event_type": "event_type",
                           "ts": "event_timestamp"})

  2. value_map   — map enum values to the project's vocabulary
                    (e.g. {"event_type": {"motion": "anomaly",
                                          "intrusion": "anomaly"},
                           "access_result": {"fail": "denied"}})

  3. type coercion (timestamps → tz-aware UTC; confidence_score → float)

  4. row drops — rows missing required fields, with unparseable
                  timestamps, or with enum values not in the value_map
                  are dropped; the caller gets a per-reason count.

Returns the cleaned DataFrame in the project's schema, the number of
dropped rows, and a list of human-readable drop reasons.

Precedent: `p1_adapter.py`. Same shape, same defensive defaults.
The same callable can be unit-tested without Streamlit, without a
running session, without any I/O.
"""
from __future__ import annotations
from typing import Iterable

import pandas as pd

# The set of values that the fusion rules' enum-columns can hold. The
# value_map is keyed by user-side strings; missing entries mean "drop
# the row." Listed here so the GUI's value-mapping step can show a
# dropdown of legal targets.
P7_EVENT_TYPES: tuple[str, ...] = (
    "person_detected",
    "vehicle_detected",
    "anomaly",
)
P7_ACCESS_RESULTS: tuple[str, ...] = (
    "granted",
    "denied",
    "invalid",
    "tailgate",
)
P7_ACCESS_REASONS: tuple[str, ...] = (
    "ok",
    "expired",
    "wrong_zone",
    "revoked",
    "unknown_badge",
    "forced_door",
)


def _apply_column_map(df: pd.DataFrame, column_map: dict[str, str]) -> pd.DataFrame:
    """Rename user-side columns to project-side names. Columns not in the
    map are passed through unchanged. Empty target string means "drop
    this user column" (we never see it in the output)."""
    if not column_map:
        return df
    # Rename only the keys that exist in the df. Strip entries whose
    # target is empty (the GUI uses "" to mean "don't keep this column").
    real_renames = {
        u: p for u, p in column_map.items() if p and u in df.columns
    }
    return df.rename(columns=real_renames)


def _drop_missing(df: pd.DataFrame, required: Iterable[str]) -> tuple[pd.DataFrame, int, str]:
    """Drop rows with NaN / empty in any required column. Returns
    (cleaned_df, n_dropped, reason_str)."""
    required = list(required)
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"missing required columns: {missing_cols}")
    # Treat empty string and NaN both as missing.
    mask = pd.Series(True, index=df.index)
    for c in required:
        col = df[c]
        # Empty string -> missing for object cols; NaN -> missing for all.
        not_missing = col.notna() & (col.astype(str).str.strip() != "")
        mask &= not_missing
    n_dropped = int((~mask).sum())
    if n_dropped == 0:
        return df, 0, ""
    return df[mask].copy(), n_dropped, f"missing required field ({', '.join(required)})"


def _apply_value_map(
    df: pd.DataFrame,
    col: str,
    value_map: dict[str, str],
    legal_targets: tuple[str, ...],
) -> tuple[pd.DataFrame, int, str]:
    """Map user-side enum values to project-side. Rows whose value is
    in value_map get the mapped value. Rows whose value is already a
    legal target pass through. Rows whose value is anything else are
    dropped (with a per-value reason)."""
    if col not in df.columns:
        return df, 0, ""
    if not value_map:
        # No value map provided: only allow rows whose value is already legal.
        mask = df[col].isin(legal_targets)
        n_dropped = int((~mask).sum())
        if n_dropped == 0:
            return df, 0, ""
        unique_bad = sorted(df.loc[~mask, col].astype(str).unique())[:5]
        return (df[mask].copy(), n_dropped,
                f"{col} not in value map and not a legal project value "
                f"(examples: {unique_bad})")
    # Apply the map; if a value is not in the map, check whether it's
    # already a legal project-side value (passthrough), else drop.
    mapped = df[col].map(value_map)
    passthrough_mask = df[col].isin(legal_targets) & mapped.isna()
    mapped = mapped.where(~passthrough_mask, df[col])
    bad_mask = mapped.isna() | ~mapped.isin(legal_targets)
    n_dropped = int(bad_mask.sum())
    if n_dropped == 0:
        return df.assign(**{col: mapped}), 0, ""
    unique_bad = sorted(df.loc[bad_mask, col].astype(str).unique())[:5]
    return (df.assign(**{col: mapped}).loc[~bad_mask].copy(), n_dropped,
            f"{col} not in value map (examples: {unique_bad})")


def adapt_uploaded_surveillance(
    user_df: pd.DataFrame,
    column_map: dict[str, str] | None = None,
    value_map: dict[str, dict[str, str]] | None = None,
) -> tuple[pd.DataFrame, dict[str, int], list[str]]:
    """Adapt a user-uploaded surveillance DataFrame to P7's schema.

    Parameters
    ----------
    user_df : the user's raw DataFrame (whatever columns they uploaded).
    column_map : {user_col: project_col}. Renames columns before validation.
                 Pass "" as a value to drop a column.
    value_map : {project_col: {user_value: project_value}}. Maps enum
                values. E.g. {"event_type": {"motion": "anomaly"}}.

    Returns
    -------
    (df, counts, reasons)
        df       : cleaned DataFrame in P7's SURVEILLANCE_EVENTS_COLS schema.
        counts   : {"missing_required": N, "bad_timestamp": M,
                    "bad_value_<col>": K, "kept": total - drops}
        reasons  : human-readable strings explaining the drops, for the GUI.
    """
    if user_df.empty:
        return user_df, {"kept": 0}, ["uploaded surveillance file is empty"]

    counts: dict[str, int] = {"kept": 0}
    reasons: list[str] = []

    df = _apply_column_map(user_df, column_map or {})

    # Required project-side fields for surveillance.
    required = [
        "site_id", "zone_id", "event_timestamp", "event_type",
        "confidence_score",
    ]
    df, n, reason = _drop_missing(df, required)
    if n:
        counts["missing_required"] = counts.get("missing_required", 0) + n
        reasons.append(f"{n} row(s) dropped: {reason}")

    # Coerce timestamps. NaT -> drop.
    if "event_timestamp" in df.columns:
        original = df["event_timestamp"].copy()
        df["event_timestamp"] = pd.to_datetime(df["event_timestamp"],
                                               utc=True, errors="coerce")
        bad = df["event_timestamp"].isna() & original.notna()
        n_bad = int(bad.sum())
        if n_bad:
            df = df[~bad].copy()
            counts["bad_timestamp"] = counts.get("bad_timestamp", 0) + n_bad
            reasons.append(f"{n_bad} row(s) dropped: bad event_timestamp")

    # Coerce confidence to float (errors -> NaN -> drop).
    if "confidence_score" in df.columns:
        original = df["confidence_score"].copy()
        df["confidence_score"] = pd.to_numeric(df["confidence_score"],
                                               errors="coerce")
        bad = df["confidence_score"].isna() & original.notna()
        n_bad = int(bad.sum())
        if n_bad:
            df = df[~bad].copy()
            counts["bad_confidence"] = counts.get("bad_confidence", 0) + n_bad
            reasons.append(f"{n_bad} row(s) dropped: bad confidence_score")

    # event_type enum mapping. value_map has shape
    # {"event_type": {user_val: project_val}, ...}.
    vm = (value_map or {}).get("event_type", {})
    df, n, reason = _apply_value_map(df, "event_type", vm, P7_EVENT_TYPES)
    if n:
        counts["bad_value_event_type"] = counts.get("bad_value_event_type", 0) + n
        reasons.append(f"{n} row(s) dropped: {reason}")

    # Derive anomaly from event_type (defense in depth: a row with
    # event_type == "anomaly" gets anomaly=True; we keep the project
    # convention from p1_adapter).
    if "event_type" in df.columns:
        df["anomaly"] = df["event_type"] == "anomaly"
    else:
        df["anomaly"] = False

    # Description: short NL stub. The LLM summarizer uses this; the
    # fusion layer doesn't.
    if "description" not in df.columns:
        device_col = "device_id" if "device_id" in df.columns else None
        if device_col:
            df["description"] = (
                "Camera " + df[device_col].astype(str)
                + " detected " + df["event_type"].astype(str).str.replace("_", " ")
                + " in " + df["zone_id"].astype(str) + "."
            )
        else:
            df["description"] = (
                df["event_type"].astype(str)
                + " in " + df["zone_id"].astype(str) + "."
            )

    # Fill missing optional columns so the downstream schema is exact.
    for col in ("event_id", "site_id", "zone_id", "device_id",
                "event_type", "description"):
        if col not in df.columns:
            df[col] = ""

    counts["kept"] = int(len(df))
    return df, counts, reasons


def adapt_uploaded_access(
    user_df: pd.DataFrame,
    column_map: dict[str, str] | None = None,
    value_map: dict[str, dict[str, str]] | None = None,
) -> tuple[pd.DataFrame, dict[str, int], list[str]]:
    """Adapt a user-uploaded access-log DataFrame to P7's schema."""
    if user_df.empty:
        return user_df, {"kept": 0}, ["uploaded access file is empty"]

    counts: dict[str, int] = {"kept": 0}
    reasons: list[str] = []

    df = _apply_column_map(user_df, column_map or {})

    required = [
        "site_id", "zone_id", "log_timestamp", "access_result",
    ]
    df, n, reason = _drop_missing(df, required)
    if n:
        counts["missing_required"] = counts.get("missing_required", 0) + n
        reasons.append(f"{n} row(s) dropped: {reason}")

    if "log_timestamp" in df.columns:
        original = df["log_timestamp"].copy()
        df["log_timestamp"] = pd.to_datetime(df["log_timestamp"],
                                             utc=True, errors="coerce")
        bad = df["log_timestamp"].isna() & original.notna()
        n_bad = int(bad.sum())
        if n_bad:
            df = df[~bad].copy()
            counts["bad_timestamp"] = counts.get("bad_timestamp", 0) + n_bad
            reasons.append(f"{n_bad} row(s) dropped: bad log_timestamp")

    # access_result enum mapping.
    vm_ar = (value_map or {}).get("access_result", {})
    df, n, reason = _apply_value_map(df, "access_result", vm_ar,
                                     P7_ACCESS_RESULTS)
    if n:
        counts["bad_value_access_result"] = (
            counts.get("bad_value_access_result", 0) + n)
        reasons.append(f"{n} row(s) dropped: {reason}")

    # reason: if not provided, derive from access_result.
    if "reason" not in df.columns and "access_result" in df.columns:
        reason_map = {
            "granted": "ok",
            "denied": "revoked",
            "invalid": "unknown_badge",
            "tailgate": "forced_door",
        }
        df["reason"] = df["access_result"].map(reason_map).fillna("unknown")
    elif "reason" in df.columns:
        # If the user mapped `reason` themselves, validate the values.
        vm_r = (value_map or {}).get("reason", {})
        df, n, reason = _apply_value_map(df, "reason", vm_r, P7_ACCESS_REASONS)
        if n:
            counts["bad_value_reason"] = counts.get("bad_value_reason", 0) + n
            reasons.append(f"{n} row(s) dropped: {reason}")

    # Fill missing optional columns.
    for col in ("log_id", "site_id", "zone_id", "device_id", "user_id",
                "access_result", "reason"):
        if col not in df.columns:
            df[col] = ""

    counts["kept"] = int(len(df))
    return df, counts, reasons


# --- self-check ---------------------------------------------------------------

if __name__ == "__main__":
    # Round-trip a tiny example: minimal columns, one rename, one value map.
    import sys
    from pathlib import Path
    REPO = Path(__file__).resolve().parents[3]

    sv = pd.DataFrame([
        {"ts": "2025-01-03T02:28:29+00:00", "site": "SITE-001",
         "zone": "SITE-001::ZONE-D", "kind": "motion",
         "conf": 0.91, "device": "CAM-1"},
        {"ts": "2025-01-03T02:30:00+00:00", "site": "SITE-001",
         "zone": "SITE-001::ZONE-D", "kind": "intrusion",
         "conf": 0.95, "device": "CAM-2"},
        {"ts": "not a timestamp", "site": "SITE-001", "zone": "X", "kind": "motion",
         "conf": "0.40", "device": "CAM-3"},
    ])
    ac = pd.DataFrame([
        {"ts": "2025-01-03T02:30:30+00:00", "site": "SITE-001",
         "zone": "SITE-001::ZONE-D", "result": "fail", "user": "USR-1"},
        {"ts": "2025-01-03T02:31:00+00:00", "site": "SITE-001",
         "zone": "SITE-001::ZONE-D", "result": "denied", "user": "USR-2"},
    ])
    sv_col = {"ts": "event_timestamp", "site": "site_id",
              "zone": "zone_id", "kind": "event_type",
              "conf": "confidence_score", "device": "device_id"}
    ac_col = {"ts": "log_timestamp", "site": "site_id",
              "zone": "zone_id", "result": "access_result",
              "user": "user_id"}
    sv_val = {"event_type": {"motion": "anomaly", "intrusion": "anomaly"}}
    ac_val = {"access_result": {"fail": "denied"}}

    sv_out, sv_c, sv_r = adapt_uploaded_surveillance(sv, sv_col, sv_val)
    ac_out, ac_c, ac_r = adapt_uploaded_access(ac, ac_col, ac_val)
    print(f"surveillance: in={len(sv)} kept={sv_c['kept']} counts={sv_c}")
    print(f"  reasons: {sv_r}")
    print(f"  columns: {list(sv_out.columns)}")
    print(f"access:      in={len(ac)} kept={ac_c['kept']} counts={ac_c}")
    print(f"  reasons: {ac_r}")
    print(f"  columns: {list(ac_out.columns)}")
    assert sv_c["kept"] == 2, f"expected 2 kept, got {sv_c['kept']}"
    assert ac_c["kept"] == 2, f"expected 2 kept, got {ac_c['kept']}"
    print("upload_adapter self-check OK")