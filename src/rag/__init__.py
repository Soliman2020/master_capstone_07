"""RAG package: policy KB loader + MMR retriever.

Native chromadb + sentence-transformers (all-MiniLM-L6-v2, 384-dim).
No langchain wrappers. The retriever returns doc_id-bearing dicts so the
agent/summarizer can cite KB-XXXXX in every recommendation.
"""