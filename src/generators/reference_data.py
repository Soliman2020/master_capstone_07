"""Reference data generator: sites / zones / devices / users.

Vertical-slice scale: 1 site, 1 zone, ~10 devices, ~20 users. All
deterministic from SEED.

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

# Vertical-slice counts. Bumping these is a one-line change.
N_SITES = 1
N_ZONES_PER_SITE = 1
N_DEVICES = 10
N_USERS = 20


def _site_id(i: int) -> str:
    return f"SITE-{i+1:03d}"


def _zone_id(i: int) -> str:
    return f"ZONE-{i+1:03d}"


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
            rows.append({
                "zone_id": _zone_id(z),
                "site_id": sid,
                "zone_name": f"Zone-{z+1}",
                # Vertical slice: single zone is restricted (so the intrusion
                # rule has something to fire on). Bump zone count + randomize
                # this flag when scaling past the slice.
                "restricted": True,
            })
    return rows


def generate_devices(rng: np.random.Generator) -> list[dict]:
    rows = []
    device_types = ["camera", "badge_reader", "door"]
    # Even split: ~7 cameras, ~2 badge readers, ~1 door (10 total).
    type_counts = [7, 2, 1]
    counter = 0
    for s in range(N_SITES):
        sid = _site_id(s)
        for z in range(N_ZONES_PER_SITE):
            zid = _zone_id(z)
            for dtype, n in zip(device_types, type_counts):
                for _ in range(n):
                    rows.append({
                        "device_id": _device_id(counter),
                        "site_id": sid,
                        "zone_id": zid,
                        "device_type": dtype,
                    })
                    counter += 1
    assert counter == N_DEVICES, f"device count off: {counter}"
    return rows


def generate_users(rng: np.random.Generator) -> list[dict]:
    """Mix of roles. The fusion layer cares about role-mismatch (a
    cleaner in a server room at 3am) and repeated badge denials, so
    we want enough variety to exercise those paths.
    """
    roles = ["employee"] * 12 + ["contractor"] * 4 + ["cleaner"] * 3 + ["security"] * 1
    assert len(roles) == N_USERS
    zone_ids = [_zone_id(z) for z in range(N_ZONES_PER_SITE)]
    rows = []
    for i in range(N_USERS):
        uid = _user_id(i)
        role = roles[i]
        # Security gets all zones; everyone else gets the one zone in the slice.
        # When scaling, expand this to role-specific zone sets.
        if role == "security":
            auth_zones = zone_ids
        else:
            auth_zones = zone_ids
        rows.append({
            "user_id": uid,
            "site_id": _site_id(0),
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


if __name__ == "__main__":
    paths = write_csvs()
    for name, p in paths.items():
        print(f"wrote {p}  ({p.stat().st_size} bytes)")
