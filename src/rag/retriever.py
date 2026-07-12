"""RAG retriever: query the policy KB with MMR (maximal marginal relevance).

Returns up to k=3 docs per query. MMR balances
relevance against redundancy so the summarizer gets distinct policy
angles, not three near-duplicate paragraphs. On a 5-doc KB the diversity
term has little to choose from, so MMR degrades gracefully to top-k;
the formula is still correct and will matter when the KB grows.

Native chromadb + sentence-transformers. No langchain
retriever wrapper (not installed; nothing gained on 5 docs). The MMR
re-rank is ~15 lines over the already-returned candidates.

CLI:
    python -m project_07_final_synthesis.src.rag.retriever --query "intrusion restricted zone"
    python src/rag/retriever.py --query "intrusion restricted zone"
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

_PROJECT = Path(__file__).resolve().parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT))
sys.path.insert(0, str(_REPO_ROOT))
os.chdir(_REPO_ROOT)

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from src.rag.knowledge_base_loader import (
    COLLECTION_NAME,
    EMBED_MODEL_NAME,
    VECTOR_STORE_DIR,
)

TOP_K = 3
# Candidate pool for MMR. With only 5 KB docs this is capped at 5 in
# _retrieve(); bumping it matters once the KB grows.
MMR_POOL_N = 10
# Lambda: relevance vs diversity trade-off. 0.5 = balanced. Higher =
# more relevance, less diversity. 0.5 is the textbook default.
MMR_LAMBDA = 0.5

# Category routing: maps each fusion incident_type to its KB category.
# The KB docs share heavy policy vocabulary (badge, human review, retention),
# so pure semantic similarity mis-ranks tailgate/privacy. The fusion layer
# already KNOWS the incident type, so we route on it: the routed category's
# docs get a relevance bonus before MMR, guaranteeing the right policy ranks
# first. MMR still adds diverse neighbors for context. The category field is
# the 1:1 alignment between the rule layer and the KB — keeping this map in
# the retriever (not the agent) makes the citation guarantee testable here.
INCIDENT_TYPE_TO_CATEGORY = {
    "suspected_unauthorized_entry": "intrusion",
    "repeated_badge_denials": "denials",
    "cross_anomaly_correlation": "correlation",
    "tailgate_door_activity": "tailgate",
}
# How much the routed category's cosine score is boosted before MMR. 0.3 is
# enough to lift a 0.45 (tailgate) above a 0.68 (denials) without making the
# bonus dominate the diversity term. Tunable; not a magic number.
CATEGORY_BONUS = 0.3


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine between a (1,d) and b (m,d). Vectors are pre-normalized
    in the loader, but we normalize again defensively in case the caller
    embedded a fresh query (also normalized)."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return b @ a  # (m,)


def _mmr_rerank(
    rel_sim: np.ndarray, cand_embs: np.ndarray, k: int, lam: float = MMR_LAMBDA,
) -> list[int]:
    """Greedy MMR selection over candidate embeddings.

    rel_sim: cosine sim of each candidate to the query, shape (m,).
    cand_embs: candidate embeddings, shape (m, d) (assumed normalized).
    Returns the selected indices (length = min(k, m)).

    Note: O(m*k) greedy. Fine at vertical-slice m<=10. For a large KB
    you'd cap m harder or use an approximate nearest-neighbor pool first.
    """
    m = len(rel_sim)
    if m == 0:
        return []
    k = min(k, m)
    selected: list[int] = []
    remaining = list(range(m))
    # Track max sim of each remaining candidate to any already-selected doc.
    max_sim_to_sel = np.full(m, -np.inf)
    while len(selected) < k and remaining:
        best_idx = None
        best_score = -np.inf
        for i in remaining:
            diversity = 0.0 if not selected else max_sim_to_sel[i]
            score = lam * rel_sim[i] - (1 - lam) * diversity
            if score > best_score:
                best_score = score
                best_idx = i
        selected.append(best_idx)
        remaining.remove(best_idx)
        # Update max-sim-to-selected for the rest using the newly added doc.
        new_sim = _cos(cand_embs[best_idx], cand_embs).ravel()
        max_sim_to_sel = np.maximum(max_sim_to_sel, new_sim)
    return selected


def retrieve(
    query: str,
    k: int = TOP_K,
    pool_n: int = MMR_POOL_N,
    lam: float = MMR_LAMBDA,
    category_hint: str | None = None,
    store_dir: Path = VECTOR_STORE_DIR,
    collection_name: str = COLLECTION_NAME,
    model_name: str = EMBED_MODEL_NAME,
) -> list[dict]:
    """Retrieve up to k policy docs for `query` using MMR.

    Returns a list of dicts in MMR order:
        {"doc_id", "title", "category", "body", "score"}
    where score is the cosine similarity to the query (for transparency).
    Empty list if the store is empty / missing (caller decides what to do).

    `category_hint`: if set, candidates whose metadata `category` matches get
    a +CATEGORY_BONUS boost to their relevance score before MMR. This is the
    fix for the shared-vocabulary mis-ranking (tailgate/privacy): the fusion
    layer already knows the incident type, so routing on it guarantees the
    matching policy ranks first. MMR still adds diverse neighbors. The
    returned `score` is the RAW cosine (pre-bonus) so transparency is preserved.
    """
    client = chromadb.PersistentClient(path=str(store_dir))
    try:
        collection = client.get_collection(collection_name)
    except Exception:
        # Store not built yet.
        return []

    # Stage 1: pull a relevance pool via Chroma's cosine query.
    pool_n = min(pool_n, collection.count())
    if pool_n == 0:
        return []
    pool = collection.query(query_texts=[query], n_results=pool_n)

    ids = pool["ids"][0]
    docs = pool["documents"][0]
    metas = pool["metadatas"][0]
    if not ids:
        return []

    # Stage 2: MMR re-rank using actual embeddings (Chroma's query gives
    # distances, not the embeddings themselves, so we fetch them by id).
    emb_rows = collection.get(ids=ids, include=["embeddings"])
    cand_embs = np.asarray(emb_rows["embeddings"], dtype=np.float32)

    # Relevance sim = cosine(query, candidate). We embed the query fresh
    # (normalized) and dot against the normalized candidate embeddings.
    model = SentenceTransformer(model_name)
    q_emb = model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
    rel_sim = _cos(q_emb, cand_embs)

    # Apply category bonus to the SCORES used for MMR ordering only.
    # rel_sim is left untouched so the returned `score` stays the raw cosine.
    if category_hint:
        mmr_sim = rel_sim.copy()
        for i, m in enumerate(metas):
            if m.get("category") == category_hint:
                mmr_sim[i] += CATEGORY_BONUS
    else:
        mmr_sim = rel_sim

    order = _mmr_rerank(mmr_sim, cand_embs, k=k, lam=lam)

    out: list[dict] = []
    for idx in order:
        out.append({
            "doc_id": metas[idx]["doc_id"],
            "title": metas[idx]["title"],
            "category": metas[idx]["category"],
            "body": docs[idx],
            "score": float(rel_sim[idx]),
        })
    return out


def retrieve_for_incident(
    incident_type: str,
    query: str,
    k: int = TOP_K,
    **kwargs,
) -> list[dict]:
    """Retrieve policies for a fused incident, routing on its incident_type.

    The fusion layer knows the rule that fired, so we map it to a KB category
    and pass that as `category_hint` to retrieve(). This guarantees the policy
    matching the incident's own type ranks first; MMR still fills the rest
    with diverse neighbors for context. Unknown incident_types fall back to
    plain semantic retrieval (no hint) — safe but un-boosted.
    """
    hint = INCIDENT_TYPE_TO_CATEGORY.get(incident_type)
    return retrieve(query, k=k, category_hint=hint, **kwargs)


def _main() -> None:
    ap = argparse.ArgumentParser(description="P7 policy KB retriever (MMR, k=3).")
    ap.add_argument("--query", required=True, help="Natural-language query.")
    ap.add_argument("--build", action="store_true",
                    help="Build the vector store first (delegates to the loader).")
    ap.add_argument("--incident-type", default=None,
                    help="Route on a fusion incident_type (boosts its category).")
    ap.add_argument("-k", type=int, default=TOP_K)
    args = ap.parse_args()

    if args.build:
        from src.rag.knowledge_base_loader import build_vector_store
        n, _ = build_vector_store()
        print(f"built vector store: {n} docs")

    if args.incident_type:
        results = retrieve_for_incident(args.incident_type, args.query, k=args.k)
        print(f"(routed on incident_type={args.incident_type!r})")
    else:
        results = retrieve(args.query, k=args.k)
    if not results:
        print("no results (vector store not built? run with --build)")
        return
    for r in results:
        print(f"[{r['doc_id']}] ({r['category']}) score={r['score']:.3f}  {r['title']}")
        print(f"    {r['body'][:120]}...")


if __name__ == "__main__":
    _main()