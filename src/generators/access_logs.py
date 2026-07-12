"""Access logs generator.

Vertical-slice scale: ~200 logs. Access results skewed toward 'granted'
(routine), with a small tail of denials and invalid badges. The fusion
layer's "repeated badge denials" rule needs >=3 denials within 1 hour in
the same zone, so we seed extra denials for a couple of user_ids to make
the rule fire deterministically.

Output: /data/synthetic/access_logs.parquet
"""
from __future__ import annotations
from datetime import datetime, timedelta, timezone

from pathlib import Path
import os, sys

import numpy as np
import pandas as pd

_PROJECT = Path(__file__).resolve().parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT))
sys.path.insert(0, str(_REPO_ROOT))
os.chdir(_REPO_ROOT)

from src.schema import ACCESS_LOGS_COLS, empty_access_logs
from src.utils.constants import SEED
from src.utils.io import write_parquet

# Vertical-slice knobs.
N_LOGS = 200
# Rates rescaled to sum to 1.0 over granted/denied/invalid only. Tailgate is
# NOT in the random draw — the single deterministic injection below is the
# only source of tailgate rows, so we get exactly 1 (was 4: 1 injected + ~3
# random draws at TAILGATE_RATE=0.02).
GRANT_RATE = 0.87          # ~87% granted
DENY_RATE = 0.08           # 8% denied
INVALID_RATE = 0.05        # 5% invalid (unknown badge)
SIM_DURATION_HOURS = 24

# 1 invalid-badge burst + 1 denial burst, both small. Keeps
# the test deterministic and gives the fusion rule a clear seed event.
# Scaling up: replace with a small Dirichlet draw on (user_id, zone).
DENIAL_BURST_USER = "USR-005"   # repeated denied attempts in zone 0
DENIAL_BURST_COUNT = 4          # >=3 trips the rule
TAILGATE_DEVICE = "DEV-008"     # one specific door sees a tailgate


def _log_id(i: int) -> str:
    return f"LOG-{i+1:06d}"


def _access_result(rng: np.random.Generator) -> tuple[str, str]:
    """Pick an access_result and a matching reason.

    Returns only granted/denied/invalid. Tailgate is injected once
    deterministically downstream, never sampled here.
    """
    r = rng.random()
    if r < GRANT_RATE:
        return "granted", "ok"
    if r < GRANT_RATE + DENY_RATE:
        return "denied", str(rng.choice(["expired", "wrong_zone", "revoked"]))
    return "invalid", "unknown_badge"


def generate_access_logs(
    rng: np.random.Generator,
    sites: pd.DataFrame,
    zones: pd.DataFrame,
    devices: pd.DataFrame,
    users: pd.DataFrame,
    n: int = N_LOGS,
) -> pd.DataFrame:
    readers = devices[devices["device_type"] == "badge_reader"].reset_index(drop=True)
    if readers.empty:
        raise ValueError("No badge_reader devices in reference data; cannot generate access logs.")

    base_time = datetime(2026, 7, 1, 0, 0, 0, tzinfo=timezone.utc)
    user_ids = users["user_id"].tolist()

    rows = []
    for i in range(n):
        reader = readers.iloc[i % len(readers)]
        result, reason = _access_result(rng)
        # The denial-burst user is the seeded anomaly: a non-existent
        # user_id so the result is always 'denied' (or 'invalid'). But
        # we keep it in the user_ids list so the FK stays valid, and
        # override the result on those rows below.
        uid = DENIAL_BURST_USER if i < DENIAL_BURST_COUNT else str(rng.choice(user_ids))
        # Default: spread across the 24h window.
        seconds_offset = int(rng.uniform(0, SIM_DURATION_HOURS * 3600))
        ts = base_time + timedelta(seconds=seconds_offset)
        rows.append({
            "log_id": _log_id(i),
            "site_id": reader["site_id"],
            "zone_id": reader["zone_id"],
            "device_id": reader["device_id"],
            "log_timestamp": ts,
            "user_id": uid,
            "access_result": result,
            "reason": reason,
        })

    # Override: the first DENIAL_BURST_COUNT rows are all denied denials
    # clustered in the same hour. Sort by time first, then re-stamp them
    # to a tight window so the fusion rule's "3 denials / 1 hour" trips.
    df = pd.DataFrame(rows)
    burst_start = base_time + timedelta(hours=10)  # mid-morning, deterministic
    for k in range(DENIAL_BURST_COUNT):
        idx = k
        df.at[idx, "log_timestamp"] = burst_start + timedelta(minutes=k * 5)
        df.at[idx, "access_result"] = "denied"
        df.at[idx, "reason"] = "revoked"
        # Force a specific reader so the rule's "same zone" clause holds.
        df.at[idx, "device_id"] = readers.iloc[0]["device_id"]
        df.at[idx, "zone_id"] = readers.iloc[0]["zone_id"]

    # Inject one tailgate event deterministically at the marked device.
    tailgate_idx = min(N_LOGS - 1, DENIAL_BURST_COUNT + 2)
    df.at[tailgate_idx, "device_id"] = TAILGATE_DEVICE
    df.at[tailgate_idx, "access_result"] = "tailgate"
    df.at[tailgate_idx, "reason"] = "forced_door"
    df.at[tailgate_idx, "log_timestamp"] = burst_start + timedelta(minutes=DENIAL_BURST_COUNT * 5 + 1)

    df = df.sort_values("log_timestamp").reset_index(drop=True)
    # Re-stamp log_ids so they're in chronological order.
    df["log_id"] = [f"LOG-{i+1:06d}" for i in range(len(df))]

    # Cast types.
    df["log_timestamp"] = pd.to_datetime(df["log_timestamp"], utc=True)
    return df[ACCESS_LOGS_COLS]


def main() -> Path:
    sites = pd.read_csv("data/reference/sites.csv")
    zones = pd.read_csv("data/reference/zones.csv")
    devices = pd.read_csv("data/reference/devices.csv")
    users = pd.read_csv("data/reference/users.csv")

    rng = np.random.default_rng(SEED)
    df = generate_access_logs(rng, sites, zones, devices, users)
    path = write_parquet(df, "access_logs")
    denied = int((df["access_result"] == "denied").sum())
    print(f"wrote {path}  rows={len(df)} denied={denied} "
          f"tailgate={int((df['access_result'] == 'tailgate').sum())}")
    return path


if __name__ == "__main__":
    main()
