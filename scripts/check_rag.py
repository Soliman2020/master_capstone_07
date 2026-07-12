"""RAG self-check: route each fused incident_type to its policy doc.

Runnable directly (no -c quoting hell):
    python scripts/check_rag.py            # from project_07_final_synthesis/
    python scripts/check_rag.py --k 3      # show k=3 diverse neighbors per type

Verifies: every incident_type -> matching KB doc at rank 1, and that
k=3 still returns diverse neighbors (context for the summarizer).
This is the ponytail self-check for the RAG layer.
"""
import argparse
import os
import sys
from pathlib import Path

_PROJECT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT))
sys.path.insert(0, str(_REPO_ROOT))
os.chdir(_REPO_ROOT)

from src.rag.retriever import retrieve_for_incident

# Each incident_type paired with a representative query. Mirrors the
# fusion layer's actual incident_type values (see data/synthetic/incidents).
CASES = [
    ("suspected_unauthorized_entry", "unauthorized entry into restricted zone high confidence after hours"),
    ("repeated_badge_denials",      "three denied badge attempts same zone within an hour"),
    ("cross_anomaly_correlation",    "camera anomaly plus denied access same zone ten minutes"),
    ("tailgate_door_activity",       "tailgate followed by forced door activity same zone"),
]

# incident_type -> the KB doc_id whose category must rank first.
EXPECTED = {
    "suspected_unauthorized_entry": "KB-00001",
    "repeated_badge_denials":       "KB-00002",
    "cross_anomaly_correlation":    "KB-00003",
    "tailgate_door_activity":       "KB-00004",
}


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG routing self-check.")
    ap.add_argument("--k", type=int, default=1, help="docs per query (1=correctness, 3=diversity)")
    args = ap.parse_args()

    all_ok = True
    for it, q in CASES:
        r = retrieve_for_incident(it, q, k=args.k)
        if not r:
            print(f"XX  {it:30s} -> MISS (vector store built? run retriever.py --build)")
            all_ok = False
            continue
        top = r[0]
        ok = top["doc_id"] == EXPECTED[it]
        all_ok = all_ok and ok
        flag = "OK " if ok else "XX "
        print(f"{flag}{it:30s} -> {top['doc_id']} ({top['category']}) score={top['score']:.3f}")
        if args.k > 1:
            for d in r[1:]:
                print(f"    neighbor  {d['doc_id']} ({d['category']}) score={d['score']:.3f}")
    print("ALL CORRECT" if all_ok else "FAILURES PRESENT")


if __name__ == "__main__":
    main()