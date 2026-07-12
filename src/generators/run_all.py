"""
One-shot generator: reference CSVs -> surveillance_events.parquet -> access_logs.parquet.

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
from src.generators.surveillance_events import main as gen_events
from src.generators.access_logs import main as gen_logs


def main() -> None:
    paths = write_csvs()
    for name, p in paths.items():
        print(f"ref: {p}")
    print("---")
    print(f"events: {gen_events()}")
    print(f"logs:   {gen_logs()}")


if __name__ == "__main__":
    main()
