"""P1 -> P7 data pipeline: scale up the slice to P1's full corpus.

Reads `project_01_reproducible_workflows/data/raw/*.parquet` (P1's
foundation deliverable: ~1,000 surveillance events / ~10,000 access logs
across 3 sites, produced by P1's generator with SEED=42), adapts each
to P7's schema (P1 doesn't have an `anomaly` flag, a `reason` field, or
P7's exact `access_result` values), and writes P7-shaped Parquet + CSV
to `data/synthetic/`.

This is the **scaled-up slice**. The smaller `surveillance_events.py` /
`access_logs.py` generators (1 site, 50 events, 200 logs) are kept as
the default for reviewers who don't have P1's data built. Use this
pipeline when you want P2's hypothesis tests to have statistical power
on the data the copilot actually runs against.

CLI:
    python -m project_07_final_synthesis.src.generators.p1_pipeline
    python src/generators/p1_pipeline.py            # from project_07_final_synthesis/

The orchestrator is intentionally tiny: read, adapt, write. All
schema-mapping decisions live in `src/utils/p1_adapter.py`.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

# Same bootstrap as the rest of P7: project dir + src/ + repo root on path
# so `from src...` and `from utils.p1_adapter import ...` both resolve.
_PROJECT = Path(__file__).resolve().parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _PROJECT / "src"
sys.path.insert(0, str(_PROJECT))
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_REPO_ROOT))
os.chdir(_REPO_ROOT)

import numpy as np  # noqa: E402  (after sys.path bootstrap)
import pandas as pd  # noqa: E402  (after sys.path bootstrap)

from src.utils.constants import SEED  # noqa: E402  (re-exported for run_all's consistency)
from src.utils.io import write_parquet  # noqa: E402
from src.utils.p1_adapter import adapt_surveillance, adapt_access  # noqa: E402

# P1's data location. Hardcoded because the path is part of the
# program-level convention (root CLAUDE.md "shared seed data").
P1_RAW_DIR = _REPO_ROOT / "project_01_reproducible_workflows" / "data" / "raw"
P1_SURVEILLANCE_PARQUET = "surveillance_events.parquet"
P1_ACCESS_PARQUET = "access_logs.parquet"


def _require_p1_data() -> None:
    """Fail loudly with an actionable message if P1's raw parquet is missing."""
    if not P1_RAW_DIR.is_dir():
        raise FileNotFoundError(
            f"P1 raw data not found at {P1_RAW_DIR}. "
            "Build P1 first: `cd project_01_reproducible_workflows && "
            "p3_venv/Scripts/python.exe src/generators.py` (or its equivalent). "
            "Alternatively, run `python src/generators/run_all.py` to use P7's "
            "small slice instead."
        )


def _read_p1() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read P1's raw parquet. The schema is fixed by P1's generator."""
    surv_path = P1_RAW_DIR / P1_SURVEILLANCE_PARQUET
    acc_path = P1_RAW_DIR / P1_ACCESS_PARQUET
    if not surv_path.exists() or not acc_path.exists():
        raise FileNotFoundError(
            f"P1 raw parquet missing: {surv_path if not surv_path.exists() else acc_path}"
        )
    return pd.read_parquet(surv_path), pd.read_parquet(acc_path)


def build_scaled_slice(seed_critical: bool = True) -> tuple[Path, Path, int, int, int]:
    """Build the scaled slice: P1's raw -> adapter -> P7's synthetic/.

    Returns (events_path, access_path, n_events, n_access, n_critical_seeded).
    If `seed_critical` (default True), one synthetic anomaly is appended
    to the slice so the fusion escalation gate fires end-to-end on the
    scaled data. The seeded event is flagged in `description` as
    "INJECTED-CRITICAL" and is independent of the P1 adapter (the
    rule's confidence threshold (0.85) is high enough that P1's
    natural distribution rarely produces critical-band events; see
    test_threshold_calibration.py for the calibration evidence).

    The seed is **deterministic** (uses SEED from constants.py) so the
    scaled slice is reproducible.
    """
    _require_p1_data()
    p1_surv, p1_acc = _read_p1()

    e7 = adapt_surveillance(p1_surv)
    a7, n_dropped = adapt_access(p1_acc)

    n_critical = 0
    if seed_critical:
        from src.utils.constants import SEED
        rng = np.random.default_rng(SEED)
        # Pick a high-confidence person_detected event in a restricted
        # zone, lift the confidence to 0.95, mark anomaly=True, and
        # change event_type to "anomaly" so fusion's intrusion_restricted
        # rule fires and the risk scorer reaches the critical band.
        restricted_zones = e7.loc[
            e7["zone_id"].str.contains("ZONE-D", na=False), "zone_id"
        ].unique().tolist()
        if restricted_zones:
            target_idx = int(rng.integers(0, len(e7)))
            e7.at[target_idx, "confidence_score"] = 0.95
            e7.at[target_idx, "anomaly"] = True
            e7.at[target_idx, "event_type"] = "anomaly"
            e7.at[target_idx, "description"] = (
                f"INJECTED-CRITICAL: {e7.at[target_idx, 'description']} "
                f"(seeded by p1_pipeline to exercise the escalation gate)"
            )
            n_critical = 1

    e_path = write_parquet(e7, "surveillance_events")
    a_path = write_parquet(a7, "access_logs")

    if n_dropped:
        print(f"[p1_pipeline] dropped {n_dropped} sentinel rows from access_logs "
              f"(P1's ??unknown?? + USR-0000 sentinels)")
    print(f"[p1_pipeline] surveillance: {len(p1_surv)} -> {len(e7)} events, "
          f"{int(e7['anomaly'].sum())} anomalies"
          + (f" (1 INJECTED-CRITICAL)" if n_critical else ""))
    print(f"[p1_pipeline] access:      {len(p1_acc)} -> {len(a7)} logs")
    print(f"[p1_pipeline] sites:       {sorted(e7['site_id'].unique().tolist())}")
    return e_path, a_path, len(e7), len(a7), n_critical


def main(seed_critical: bool = True) -> None:
    paths = build_scaled_slice(seed_critical=seed_critical)
    print(f"wrote {paths[0].name} + {paths[1].name} (sites=3, events={paths[2]}, access={paths[3]}, seeded_critical={paths[4]})")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="P1 -> P7 scaled data pipeline.")
    ap.add_argument("--seed-critical", action=argparse.BooleanOptionalAction, default=True,
                    help="Inject one deterministic critical-band anomaly so the "
                         "fusion escalation gate fires end-to-end on the scaled slice. "
                         "Default: enabled. Use --no-seed-critical to disable.")
    args = ap.parse_args()
    main(seed_critical=args.seed_critical)