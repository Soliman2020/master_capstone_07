"""Parquet + CSV I/O helpers.

ponytail: pandas to_parquet / read_parquet is enough. We don't need
pyarrow's full Table API; the round-trip is for caches, not for big-data
shuffle. Add pyarrow-specific tuning only if profiling shows a bottleneck.

Vertical slice writes BOTH Parquet (fast I/O) and CSV (reviewer-friendly,
opens in Excel). Set WRITE_CSV = False to skip CSV when scaling past
the slice.
"""
from pathlib import Path
import pandas as pd

# All P7 generated artifacts land here. Per spec, this is a synthetic/
# subfolder under data/.
DEFAULT_SYNTHETIC_DIR = Path("project_07_final_synthesis/data/synthetic")

# Ponytail: kill the switch by deleting the line once the slice is
# the spec. The CSV is for reviewers; the Parquet is for the agent.
WRITE_CSV = True


def write_parquet(df: pd.DataFrame, name: str, out_dir: Path = DEFAULT_SYNTHETIC_DIR) -> Path:
    """Write a DataFrame to <out_dir>/<name>.parquet, creating the dir if needed.

    Also writes <out_dir>/<name>.csv when WRITE_CSV is True. Returns
    the Parquet path (CSV path is sibling).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.parquet"
    df.to_parquet(path, index=False)
    if WRITE_CSV:
        df.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8")
    return path


def read_parquet(name: str, in_dir: Path = DEFAULT_SYNTHETIC_DIR) -> pd.DataFrame:
    """Read <in_dir>/<name>.parquet. Raises FileNotFoundError if missing."""
    path = in_dir / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Parquet not found: {path}. Run the generator first.")
    return pd.read_parquet(path)
