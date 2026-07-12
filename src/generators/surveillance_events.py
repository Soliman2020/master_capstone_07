"""Surveillance events generator.

Vertical-slice scale: ~50 events. Anomaly rate in [0.03, 0.08]
(we pick 0.06 to get 3 anomalies on 50 rows so the
fusion layer has critical-band material to fire on).

Output: /data/synthetic/surveillance_events.parquet
"""
from __future__ import annotations
from datetime import datetime, timedelta, timezone

from pathlib import Path
import os, sys

_PROJECT = Path(__file__).resolve().parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT))
sys.path.insert(0, str(_REPO_ROOT))
os.chdir(_REPO_ROOT)

import numpy as np
import pandas as pd

from src.schema import SURVEILLANCE_EVENTS_COLS, empty_surveillance
from src.utils.constants import SEED
from src.utils.io import write_parquet

# Vertical-slice knobs.
N_EVENTS = 50
ANOMALY_RATE = 0.06  # 3 anomalies on 50 rows
SIM_DURATION_HOURS = 24  # spread events across one day
CONFIDENCE_BETA_A, CONFIDENCE_BETA_B = 8.0, 2.0  # skews toward high confidence

# event_type distribution: mostly person_detected, some anomaly.
EVENT_TYPES = ["person_detected", "person_detected", "person_detected",
               "vehicle_detected", "anomaly"]


def _event_id(i: int) -> str:
    return f"EVT-{i+1:06d}"


def generate_surveillance_events(
    rng: np.random.Generator,
    sites: pd.DataFrame,
    zones: pd.DataFrame,
    devices: pd.DataFrame,
    n: int = N_EVENTS,
) -> pd.DataFrame:
    """Build n surveillance events referencing existing sites/zones/devices.

    A small fraction (ANOMALY_RATE) is marked anomaly=True. The rest are
    routine detections. Anomalies are concentrated in restricted zones and
    get higher confidence scores (so the fusion rule trips naturally).
    """
    cameras = devices[devices["device_type"] == "camera"].reset_index(drop=True)
    if cameras.empty:
        raise ValueError("No camera devices in reference data; cannot generate surveillance events.")

    restricted_zone_ids = set(zones.loc[zones["restricted"], "zone_id"])
    base_time = datetime(2026, 7, 1, 0, 0, 0, tzinfo=timezone.utc)

    rows = []
    for i in range(n):
        cam = cameras.iloc[i % len(cameras)]
        is_anomaly = bool(rng.random() < ANOMALY_RATE)
        # Anomalies concentrate in restricted zones (intentional: gives
        # the fusion rule something to fire on). Routine detections are
        # spread across the camera's actual zone.
        if is_anomaly and restricted_zone_ids:
            zone_id = next(iter(restricted_zone_ids))
        else:
            zone_id = cam["zone_id"]
        site_id = cam["site_id"]
        # Spread events across the 24h window.
        seconds_offset = int(rng.uniform(0, SIM_DURATION_HOURS * 3600))
        ts = base_time + timedelta(seconds=seconds_offset)
        # Anomalies skew toward high confidence; routine detections are
        # broader. Beta(8, 2) gives mean ~0.8, mostly 0.6-0.95.
        conf = float(np.clip(rng.beta(CONFIDENCE_BETA_A, CONFIDENCE_BETA_B), 0.0, 1.0))
        etype = "anomaly" if is_anomaly else str(rng.choice(EVENT_TYPES))
        if is_anomaly:
            desc = f"Camera {cam['device_id']} flagged an anomaly in {zone_id}."
        else:
            desc = f"Camera {cam['device_id']} detected a {etype.replace('_', ' ')} in {zone_id}."
        rows.append({
            "event_id": _event_id(i),
            "site_id": site_id,
            "zone_id": zone_id,
            "device_id": cam["device_id"],
            "event_timestamp": ts,
            "event_type": etype,
            "confidence_score": round(conf, 3),
            "anomaly": is_anomaly,
            "description": desc,
        })

    df = empty_surveillance()
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    # Force dtypes we want to read later.
    df["confidence_score"] = df["confidence_score"].astype(float)
    df["anomaly"] = df["anomaly"].astype(bool)
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True)
    return df[SURVEILLANCE_EVENTS_COLS]


def main() -> Path:
    """CLI entrypoint: load reference CSVs, generate events, write Parquet."""
    from src.utils.io import read_parquet
    # Reference CSVs live at data/reference/. Re-read so this script
    # is the single source of truth for the slice.
    sites = pd.read_csv("data/reference/sites.csv")
    zones = pd.read_csv("data/reference/zones.csv")
    devices = pd.read_csv("data/reference/devices.csv")

    rng = np.random.default_rng(SEED)
    df = generate_surveillance_events(rng, sites, zones, devices)
    path = write_parquet(df, "surveillance_events")
    anomaly_count = int(df["anomaly"].sum())
    print(f"wrote {path}  rows={len(df)} anomalies={anomaly_count} "
          f"({anomaly_count/len(df):.1%})")
    return path


if __name__ == "__main__":
    main()
