"""One-shot generator: reference CSVs -> surveillance_events.parquet -> access_logs.parquet.

Run as:
    python -m project_07_final_synthesis.src.generators.run_all
"""
from project_07_final_synthesis.src.generators.reference_data import write_csvs
from project_07_final_synthesis.src.generators.surveillance_events import main as gen_events
from project_07_final_synthesis.src.generators.access_logs import main as gen_logs


def main() -> None:
    paths = write_csvs()
    for name, p in paths.items():
        print(f"ref: {p}")
    print("---")
    print(f"events: {gen_events()}")
    print(f"logs:   {gen_logs()}")


if __name__ == "__main__":
    main()
