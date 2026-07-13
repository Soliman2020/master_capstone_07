"""
One-shot generator: reference CSVs -> surveillance_events.parquet -> access_logs.parquet.

Run as:
    python -m project_07_final_synthesis.src.generators.run_all                    # small slice
    python -m project_07_final_synthesis.src.generators.run_all --source p1         # scale to P1's corpus (3 sites, ~1k events, ~10k logs)
    python src/generators/run_all.py                                               # from project_07_final_synthesis/
    python src/generators/run_all.py --source p1
"""
import os, sys
from pathlib import Path

#   Two roots on sys.path, cwd at repo root.
#   parents[2] = project_07_final_synthesis/  -> makes `from src...` work here.
#   parents[3] = repo root                    -> keeps the generators' internal
#                `from project_07_final_synthesis.src...` working, unchanged.
# chdir(repo_root) lands `data/reference/` and the synthetic outputs at the

_PROJECT = Path(__file__).resolve().parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT))
sys.path.insert(0, str(_REPO_ROOT))
os.chdir(_REPO_ROOT)

from src.generators.reference_data import write_csvs
from src.generators.surveillance_events import main as gen_events_small
from src.generators.access_logs import main as gen_logs_small


def main(source: str = "small") -> None:
    """source='small' (default) = the 1-site/50-event/200-log slice.
    source='p1' = P1's full corpus (3 sites, ~1k events, ~10k logs).
    """
    paths = write_csvs()
    for name, p in paths.items():
        print(f"ref: {p}")
    print("---")
    if source == "p1":
        # Lazy import so reviewers without P1 built don't pay the cost
        # (and don't get a confusing error at module-load time).
        from src.generators.p1_pipeline import build_scaled_slice
        from src.generators.reference_data import write_scaled_csvs
        # Re-emit reference CSVs in the scaled layout (3 sites, 4 zones each).
        # This OVERWRITES the small-slice CSVs; run --source small afterwards
        # to restore them.
        scaled_paths = write_scaled_csvs()
        for name, p in scaled_paths.items():
            print(f"ref (scaled): {p}")
        e, a, n_e, n_a, n_crit = build_scaled_slice(seed_critical=True)
        print(f"events: {e}  (n={n_e})")
        print(f"logs:   {a}  (n={n_a})")
        print(f"seeded_critical: {n_crit}")
    elif source == "small":
        print(f"events: {gen_events_small()}")
        print(f"logs:   {gen_logs_small()}")
    else:
        raise ValueError(f"unknown --source: {source!r} (use 'small' or 'p1')")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="P7 generator (reference + synthetic data).")
    ap.add_argument("--source", default="small", choices=["small", "p1"],
                    help="small = 1-site/50-event/200-log slice (default). "
                         "p1 = P1's full corpus (3 sites, ~1k events, ~10k logs).")
    args = ap.parse_args()
    main(source=args.source)
