"""Tests for `src.utils.upload_adapter`.

Pure unit tests on the adapter functions that map user-uploaded CSVs
to the P7 schema. No I/O, no LLM, no Streamlit. These run from the
project venv under pytest and assert the contracts the GUI relies on:

  - Minimal required columns are accepted.
  - Extra columns are kept (not dropped silently).
  - Missing required columns raise ValueError.
  - Bad timestamps / bad confidence scores are dropped with a reason.
  - Enum values not in the value map are dropped with a reason.
  - Empty value map falls back to "only legal project values pass."
  - Column map renames user columns to project columns.
  - The adapt_ functions return a DataFrame in the project schema.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pytest

from src.utils.upload_adapter import (
    P7_ACCESS_REASONS,
    P7_ACCESS_RESULTS,
    P7_EVENT_TYPES,
    adapt_uploaded_access,
    adapt_uploaded_surveillance,
)


# --- Surveillance adapter -------------------------------------------------

def test_surveillance_minimal_passes_through():
    """A row with all 5 required fields and matching values passes
    through unchanged, plus the derived `anomaly` and `description`."""
    df = pd.DataFrame([{
        "site_id": "SITE-001",
        "zone_id": "SITE-001::ZONE-D",
        "event_timestamp": "2025-01-03T02:28:29+00:00",
        "event_type": "anomaly",
        "confidence_score": 0.95,
    }])
    out, counts, reasons = adapt_uploaded_surveillance(df)
    assert counts["kept"] == 1
    assert reasons == []
    assert out.iloc[0]["anomaly"] is True or out.iloc[0]["anomaly"] == True
    assert "anomaly" in out.iloc[0]["description"].lower()


def test_surveillance_extra_columns_kept():
    """User-side columns the adapter doesn't recognize are passed
    through; we never drop data silently."""
    df = pd.DataFrame([{
        "site_id": "SITE-001",
        "zone_id": "SITE-001::ZONE-A",
        "event_timestamp": "2025-01-03T02:28:29+00:00",
        "event_type": "person_detected",
        "confidence_score": 0.60,
        "vendor_specific_id": "ABC-123",
    }])
    out, counts, _ = adapt_uploaded_surveillance(df)
    assert counts["kept"] == 1
    assert "vendor_specific_id" in out.columns
    assert out.iloc[0]["vendor_specific_id"] == "ABC-123"


def test_surveillance_missing_required_columns_raises():
    """If a required column is absent after mapping, raise early — the
    GUI surfaces the error and disables Run."""
    df = pd.DataFrame([{"site_id": "SITE-001"}])  # missing most fields
    with pytest.raises(ValueError, match="missing required columns"):
        adapt_uploaded_surveillance(df)


def test_surveillance_missing_required_rows_dropped():
    """If a column is present but a row has NaN, the row is dropped with
    a 'missing required field' reason — not a hard error."""
    df = pd.DataFrame([
        {"site_id": "SITE-001", "zone_id": "ZONE-A",
         "event_timestamp": "2025-01-03T02:28:29+00:00",
         "event_type": "anomaly", "confidence_score": 0.95},
        {"site_id": None, "zone_id": "ZONE-A",
         "event_timestamp": "2025-01-03T02:29:00+00:00",
         "event_type": "anomaly", "confidence_score": 0.90},
    ])
    out, counts, reasons = adapt_uploaded_surveillance(df)
    assert counts["kept"] == 1
    assert counts["missing_required"] == 1
    assert any("missing required" in r for r in reasons)


def test_surveillance_bad_timestamp_dropped():
    """Non-parseable timestamps are dropped with a 'bad event_timestamp'
    reason. The fusion layer can't reason about NaT."""
    df = pd.DataFrame([
        {"site_id": "SITE-001", "zone_id": "ZONE-A",
         "event_timestamp": "2025-01-03T02:28:29+00:00",
         "event_type": "anomaly", "confidence_score": 0.95},
        {"site_id": "SITE-001", "zone_id": "ZONE-A",
         "event_timestamp": "not a timestamp",
         "event_type": "anomaly", "confidence_score": 0.90},
    ])
    out, counts, reasons = adapt_uploaded_surveillance(df)
    assert counts["kept"] == 1
    assert counts["bad_timestamp"] == 1
    assert any("bad event_timestamp" in r for r in reasons)


def test_surveillance_value_map_drops_unknown_event_type():
    """If the user maps a column to event_type but doesn't tell us what
    one of its values means, those rows are dropped — they can't reach
    the fusion rules without a project-side label."""
    df = pd.DataFrame([
        {"site_id": "SITE-001", "zone_id": "ZONE-A",
         "event_timestamp": "2025-01-03T02:28:29+00:00",
         "event_type": "anomaly", "confidence_score": 0.95},
        {"site_id": "SITE-001", "zone_id": "ZONE-A",
         "event_timestamp": "2025-01-03T02:29:00+00:00",
         "event_type": "unmapped_kind",  # not in value_map
         "confidence_score": 0.90},
    ])
    out, counts, reasons = adapt_uploaded_surveillance(
        df, value_map={"event_type": {"anomaly": "anomaly"}})
    assert counts["kept"] == 1
    assert "bad_value_event_type" in counts
    assert any("event_type" in r for r in reasons)


def test_surveillance_value_map_translates_user_to_project():
    """If the user maps 'motion' -> 'anomaly', the fusion layer sees
    'anomaly' (the rule's only anomaly signal)."""
    df = pd.DataFrame([{
        "site_id": "SITE-001", "zone_id": "SITE-001::ZONE-D",
        "event_timestamp": "2025-01-03T02:28:29+00:00",
        "event_type": "motion",
        "confidence_score": 0.95,
    }])
    out, counts, _ = adapt_uploaded_surveillance(
        df, value_map={"event_type": {"motion": "anomaly"}})
    assert counts["kept"] == 1
    assert out.iloc[0]["event_type"] == "anomaly"
    assert out.iloc[0]["anomaly"] is True or out.iloc[0]["anomaly"] == True


def test_surveillance_column_map_renames():
    """A column_map renames user columns to project columns. The output
    uses the project-side names regardless of what the user uploaded."""
    df = pd.DataFrame([{
        "ts": "2025-01-03T02:28:29+00:00",
        "site": "SITE-001", "zone": "SITE-001::ZONE-D",
        "kind": "anomaly", "conf": 0.95, "device": "CAM-1",
    }])
    column_map = {"ts": "event_timestamp", "site": "site_id",
                  "zone": "zone_id", "kind": "event_type",
                  "conf": "confidence_score", "device": "device_id"}
    out, counts, _ = adapt_uploaded_surveillance(df, column_map=column_map)
    assert counts["kept"] == 1
    for proj_col in ("event_timestamp", "site_id", "zone_id",
                     "event_type", "confidence_score", "device_id"):
        assert proj_col in out.columns


def test_surveillance_empty_input():
    """An empty DataFrame is reported back with kept=0 and a reason."""
    df = pd.DataFrame()
    out, counts, reasons = adapt_uploaded_surveillance(df)
    assert counts["kept"] == 0
    assert any("empty" in r for r in reasons)


# --- Access adapter -------------------------------------------------------

def test_access_minimal_passes_through():
    """A row with all 4 required access fields and a legal access_result
    value passes through; reason is derived from access_result."""
    df = pd.DataFrame([{
        "site_id": "SITE-001", "zone_id": "ZONE-A",
        "log_timestamp": "2025-01-03T02:30:00+00:00",
        "access_result": "denied",
    }])
    out, counts, reasons = adapt_uploaded_access(df)
    assert counts["kept"] == 1
    assert reasons == []
    assert out.iloc[0]["reason"] == "revoked"  # derived from "denied"


def test_access_bad_access_result_dropped():
    """Unmapped access_result values are dropped, with a per-value reason
    so the GUI can tell the user 'X rows dropped: access_result=foo'."""
    df = pd.DataFrame([
        {"site_id": "SITE-001", "zone_id": "ZONE-A",
         "log_timestamp": "2025-01-03T02:30:00+00:00",
         "access_result": "denied"},
        {"site_id": "SITE-001", "zone_id": "ZONE-A",
         "log_timestamp": "2025-01-03T02:30:30+00:00",
         "access_result": "explosion"},  # not a legal project value
    ])
    out, counts, reasons = adapt_uploaded_access(
        df, value_map={"access_result": {}})
    assert counts["kept"] == 1
    assert "bad_value_access_result" in counts
    assert any("access_result" in r for r in reasons)


def test_access_value_map_translates_user_to_project():
    """'fail' -> 'denied' is the canonical case; the rule layer only
    knows the project-side vocabulary."""
    df = pd.DataFrame([{
        "site_id": "SITE-001", "zone_id": "ZONE-A",
        "log_timestamp": "2025-01-03T02:30:00+00:00",
        "access_result": "fail",
    }])
    out, counts, _ = adapt_uploaded_access(
        df, value_map={"access_result": {"fail": "denied"}})
    assert counts["kept"] == 1
    assert out.iloc[0]["access_result"] == "denied"
    assert out.iloc[0]["reason"] == "revoked"


def test_access_missing_required_columns_raises():
    df = pd.DataFrame([{"site_id": "SITE-001"}])
    with pytest.raises(ValueError, match="missing required columns"):
        adapt_uploaded_access(df)


def test_access_passthrough_when_value_already_legal():
    """If the user maps a column to access_result and the values are
    already in the legal set (e.g. 'denied'), they pass through even
    with no value map."""
    df = pd.DataFrame([{
        "site_id": "SITE-001", "zone_id": "ZONE-A",
        "log_timestamp": "2025-01-03T02:30:00+00:00",
        "access_result": "granted",
    }])
    out, counts, reasons = adapt_uploaded_access(df)
    assert counts["kept"] == 1
    assert reasons == []


def test_access_column_map_renames():
    df = pd.DataFrame([{
        "ts": "2025-01-03T02:30:00+00:00",
        "site": "SITE-001", "zone": "ZONE-A",
        "result": "denied", "user": "USR-1",
    }])
    column_map = {"ts": "log_timestamp", "site": "site_id",
                  "zone": "zone_id", "result": "access_result",
                  "user": "user_id"}
    out, counts, _ = adapt_uploaded_access(df, column_map=column_map)
    assert counts["kept"] == 1
    for proj_col in ("log_timestamp", "site_id", "zone_id",
                     "access_result", "user_id"):
        assert proj_col in out.columns


# --- Enum tuple sanity ----------------------------------------------------

def test_legal_value_tuples_match_schema_docstring():
    """The legal-enum tuples match the project's documented vocabulary
    in `src/schema.py`. If they drift, this test catches it."""
    assert P7_EVENT_TYPES == ("person_detected", "vehicle_detected", "anomaly")
    assert P7_ACCESS_RESULTS == ("granted", "denied", "invalid", "tailgate")
    assert "forced_door" in P7_ACCESS_REASONS
    assert "unknown_badge" in P7_ACCESS_REASONS