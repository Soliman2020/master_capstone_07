"""Knowledge-base loader: JSONL policy docs -> Chroma vector store.

Each line of data/knowledge_base/knowledge_base.jsonl is one policy doc:
    {"doc_id": "KB-00001", "title": "...", "category": "...", "body": "..."}

We embed `title + "\n" + body` (the title carries signal for short policy
docs) with sentence-transformers all-MiniLM-L6-v2 (384-dim) and store the
resulting vectors in a persistent Chroma collection at
data/knowledge_base/vector_store/. Metadata = doc_id, title, category so
the retriever can return doc_id for citation without re-parsing.

Native chromadb + sentence-transformers. The LangChain wrappers
(langchain_chroma / langchain_huggingface) are not installed in final_venv
and would add a dep + a layer for nothing on a 5-doc KB. Add them only if
we later want LangChain's retriever/chain plumbing for the agent.

Determinism: Chroma's order is the JSONL order; embeddings are pure
function of the model (no RNG). SEED not needed here but imported for
consistency with the rest of src/.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

# Bootstraps, same as the generators: short `from src...` imports work and
# cwd lands at repo root so the cwd-relative `data/...` paths resolve.
_PROJECT = Path(__file__).resolve().parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT))
sys.path.insert(0, str(_REPO_ROOT))
os.chdir(_REPO_ROOT)

import chromadb
from sentence_transformers import SentenceTransformer

from src.utils.constants import SEED  # noqa: F401  (documented above)

# --- Locations (cwd-relative after the bootstrap chdir) ---------------------
KB_JSONL = Path("project_07_final_synthesis/data/knowledge_base/knowledge_base.jsonl")
VECTOR_STORE_DIR = Path("project_07_final_synthesis/data/knowledge_base/vector_store")
COLLECTION_NAME = "p7_policy_kb"

# Embedding model: lightweight, sufficient for short policy docs.
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def load_kb_docs(path: Path = KB_JSONL) -> list[dict]:
    """Read the JSONL KB into a list of dicts. Validates the required keys."""
    if not path.exists():
        raise FileNotFoundError(f"KB JSONL not found: {path}")
    docs: list[dict] = []
    required = {"doc_id", "title", "category", "body"}
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = required - row.keys()
            if missing:
                raise ValueError(f"{path}:{i} missing keys {missing}")
            docs.append(row)
    return docs


def _embed_text(doc: dict) -> str:
    """Embedding text = title + body. Title carries signal for short docs."""
    return f"{doc['title']}\n{doc['body']}"


def build_vector_store(
    kb_path: Path = KB_JSONL,
    store_dir: Path = VECTOR_STORE_DIR,
    collection_name: str = COLLECTION_NAME,
    model_name: str = EMBED_MODEL_NAME,
) -> tuple[int, Path]:
    """Build (rebuild) the Chroma collection from the KB JSONL.

    Returns (n_docs_added, store_dir). Idempotent: drops + recreates the
    collection so re-running after a KB edit reflects the new docs.
    """
    docs = load_kb_docs(kb_path)
    if not docs:
        raise ValueError(f"KB JSONL is empty: {kb_path}")

    store_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(store_dir))

    # Drop any prior collection so re-builds don't accumulate stale vectors.
    try:
        client.delete_collection(collection_name)
    except Exception:
        # Collection not present on first build -> ignore.
        pass

    collection = client.get_or_create_collection(
        collection_name,
        metadata={"hnsw:space": "cosine"},  # cosine suits normalized semantic sim
    )

    # One model instance; batch-encode for speed (small N here, but the
    # pattern scales).
    model = SentenceTransformer(model_name)
    texts = [_embed_text(d) for d in docs]
    embeddings = model.encode(texts, normalize_embeddings=True).tolist()

    collection.add(
        ids=[d["doc_id"] for d in docs],
        documents=texts,
        metadatas=[
            {"doc_id": d["doc_id"], "title": d["title"], "category": d["category"]}
            for d in docs
        ],
        embeddings=embeddings,
    )
    return len(docs), store_dir


def _main_build() -> None:
    n, store_dir = build_vector_store()
    print(f"built {COLLECTION_NAME}: {n} docs -> {store_dir}")


if __name__ == "__main__":
    _main_build()