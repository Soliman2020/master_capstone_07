"""Reference data generator: sites / zones / devices / users.

Two layouts are supported, controlled by N_SITES / N_ZONES_PER_SITE:

  Vertical slice (N_SITES=1, N_ZONES_PER_SITE=1): 1 site, 1 zone, ~10
  devices, ~20 users. Used by `run_all.py --source small`.

  Scaled slice (N_SITES=3, N_ZONES_PER_SITE=4): 3 sites, 4 zones each
  (12 zones total), ~30 devices, ~60 users. The 4 zones per site use
  P1's naming convention: SITE-NNN::ZONE-{A,B,C,D}, where ZONE-D is
  restricted. Used by `run_all.py --source p1` to scale to P1's
  ~1k-event / ~10k-log corpus.

All deterministic from SEED.

Writes four CSVs to data/reference/:
  sites.csv, zones.csv, devices.csv, users.csv
"""
from __future__ import annotations
import csv
from pathlib import Path
import os, sys

_PROJECT = Path(__file__).resolve().parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT))
sys.path.insert(0, str(_REPO_ROOT))
os.chdir(_REPO_ROOT)

from src.utils.constants import SEED

# Same RNG instance reused per call. numpy's default_rng
# from a seed is what gives us reproducible CSVs across runs.
import numpy as np

REF_DIR = Path("data/reference")

# Vertical-slice counts (1 site, 1 zone). Override via env vars to scale
# to the P1 corpus (3 sites, 4 zones each). The `scaled` helper below
# sets the env vars and calls write_csvs again to emit the scaled layout.
N_SITES = int(os.environ.get("P7_N_SITES", "1"))
N_ZONES_PER_SITE = int(os.environ.get("P7_N_ZONES_PER_SITE", "1"))
N_DEVICES = 10 * N_SITES * N_ZONES_PER_SITE  # 10 per zone
N_USERS = 20 * N_SITES * N_ZONES_PER_SITE    # 20 per zone

# When N_ZONES_PER_SITE > 1, use P1's naming scheme (SITE-NNN::ZONE-{A,B,C,D})
# and mark ZONE-D as restricted. Otherwise the original vertical-slice
# format (SITE-NNN + ZONE-NNN) is used.
USE_SCALED_LAYOUT = N_ZONES_PER_SITE > 1
SCALED_ZONE_LETTERS = ["A", "B", "C", "D"]
SCALED_RESTRICTED_LETTER = "D"


def _site_id(i: int) -> str:
    return f"SITE-{i+1:03d}"


def _zone_id(site_idx: int, zone_idx: int) -> str:
    """P7's small-slice layout uses bare ZONE-NNN; the scaled layout uses
    SITE-NNN::ZONE-X (matching P1's generator convention) so the fusion
    rule's `events["zone_id"].isin(restricted_zones)` lookup matches
    the data.
    """
    if USE_SCALED_LAYOUT:
        return f"{_site_id(site_idx)}::ZONE-{SCALED_ZONE_LETTERS[zone_idx]}"
    return f"ZONE-{zone_idx+1:03d}"


def _device_id(i: int) -> str:
    return f"DEV-{i+1:03d}"


def _user_id(i: int) -> str:
    return f"USR-{i+1:03d}"


def generate_sites(rng: np.random.Generator) -> list[dict]:
    return [
        {"site_id": _site_id(i), "site_name": f"Building-{i+1}", "timezone": "UTC"}
        for i in range(N_SITES)
    ]


def generate_zones(rng: np.random.Generator) -> list[dict]:
    rows = []
    for s in range(N_SITES):
        sid = _site_id(s)
        for z in range(N_ZONES_PER_SITE):
            is_restricted = True
            if USE_SCALED_LAYOUT:
                # ZONE-D is restricted at every site (P1's convention).
                is_restricted = SCALED_ZONE_LETTERS[z] == SCALED_RESTRICTED_LETTER
            rows.append({
                "zone_id": _zone_id(s, z),
                "site_id": sid,
                "zone_name": f"Zone-{z+1}",
                "restricted": is_restricted,
            })
    return rows


def generate_devices(rng: np.random.Generator) -> list[dict]:
    rows = []
    device_types = ["camera", "badge_reader", "door"]
    # Even split: ~7 cameras, ~2 badge readers, ~1 door per zone.
    type_counts = [7, 2, 1]
    counter = 0
    for s in range(N_SITES):
        sid = _site_id(s)
        for z in range(N_ZONES_PER_SITE):
            zid = _zone_id(s, z)
            for dtype, n in zip(device_types, type_counts):
                for _ in range(n):
                    rows.append({
                        "device_id": _device_id(counter),
                        "site_id": sid,
                        "zone_id": zid,
                        "device_type": dtype,
                    })
                    counter += 1
    return rows


def generate_users(rng: np.random.Generator) -> list[dict]:
    """Mix of roles. The fusion layer cares about role-mismatch (a
    cleaner in a server room at 3am) and repeated badge denials, so
    we want enough variety to exercise those paths.
    """
    if USE_SCALED_LAYOUT:
        # 60 users/site, mix of roles.
        n_total = N_USERS
        roles = (["employee"] * int(n_total * 0.6)
                 + ["contractor"] * int(n_total * 0.2)
                 + ["cleaner"] * int(n_total * 0.15)
                 + ["security"] * int(n_total * 0.05))
        assert len(roles) == n_total
    else:
        roles = ["employee"] * 12 + ["contractor"] * 4 + ["cleaner"] * 3 + ["security"] * 1
        assert len(roles) == N_USERS

    # Each user is assigned to one (site, zone) in the scaled layout,
    # or the single zone in the small slice.
    rows = []
    for i, role in enumerate(roles):
        uid = _user_id(i)
        if USE_SCALED_LAYOUT:
            site_idx = i % N_SITES
            zone_idx = i % N_ZONES_PER_SITE
            auth_zones = [_zone_id(site_idx, zone_idx)]
        else:
            auth_zones = [_zone_id(0, z) for z in range(N_ZONES_PER_SITE)]
        rows.append({
            "user_id": uid,
            "site_id": _site_id(i % N_SITES) if USE_SCALED_LAYOUT else _site_id(0),
            "role": role,
            "authorized_zones": ";".join(auth_zones),  # CSV-friendly; tuple split downstream
        })
    return rows


def write_csvs(out_dir: Path = REF_DIR) -> dict[str, Path]:
    """Generate all four reference tables and write to CSV. Returns
    the paths so callers (notebooks, tests) can log them.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    tables = {
        "sites.csv": generate_sites(rng),
        "zones.csv": generate_zones(rng),
        "devices.csv": generate_devices(rng),
        "users.csv": generate_users(rng),
    }
    paths: dict[str, Path] = {}
    for name, rows in tables.items():
        path = out_dir / name
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        paths[name] = path
    return paths


def write_scaled_csvs(out_dir: Path = REF_DIR) -> dict[str, Path]:
    """One-shot: set the env vars to the scaled layout, generate, restore.

    Used by `run_all.py --source p1` so the scaled pipeline can call
    write_csvs() in the standard order (reference first, then synthetic).
    """
    saved = {k: os.environ.get(k) for k in ("P7_N_SITES", "P7_N_ZONES_PER_SITE")}
    os.environ["P7_N_SITES"] = "3"
    os.environ["P7_N_ZONES_PER_SITE"] = "4"
    try:
        # Force a re-read of the module-level constants.
        global N_SITES, N_ZONES_PER_SITE, N_DEVICES, N_USERS, USE_SCALED_LAYOUT
        N_SITES = 3
        N_ZONES_PER_SITE = 4
        N_DEVICES = 10 * N_SITES * N_ZONES_PER_SITE
        N_USERS = 20 * N_SITES * N_ZONES_PER_SITE
        USE_SCALED_LAYOUT = True
        return write_csvs(out_dir)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        N_SITES = int(os.environ.get("P7_N_SITES", "1"))
        N_ZONES_PER_SITE = int(os.environ.get("P7_N_ZONES_PER_SITE", "1"))
        N_DEVICES = 10 * N_SITES * N_ZONES_PER_SITE
        N_USERS = 20 * N_SITES * N_ZONES_PER_SITE
        USE_SCALED_LAYOUT = N_ZONES_PER_SITE > 1


if __name__ == "__main__":
    paths = write_csvs()
    for name, p in paths.items():
        print(f"wrote {p}  ({p.stat().st_size} bytes)")
