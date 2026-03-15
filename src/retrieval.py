"""
Retrieval helpers: embed a query and retrieve top-k chunks from the global corpus.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

# Module-level cache so the model is only loaded once per process
_embedder_cache: dict[str, SentenceTransformer] = {}


def get_embedder(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load (and cache) a SentenceTransformer model."""
    if model_name not in _embedder_cache:
        _embedder_cache[model_name] = SentenceTransformer(model_name)
    return _embedder_cache[model_name]


def embed_text(text: str, embedder: SentenceTransformer) -> np.ndarray:
    """
    Embed a single string.

    Returns:
        Unit-normalised 1-D numpy array.
    """
    vec = embedder.encode(text, normalize_embeddings=True, convert_to_numpy=True)
    return vec


def retrieve(query: str, embedder: SentenceTransformer, corpus, top_k: int = 5) -> list[dict]:
    """
    Embed the query and retrieve top-k chunks from the global corpus.

    Args:
        query: natural-language question.
        embedder: loaded SentenceTransformer.
        corpus: Corpus object (from src.corpus).
        top_k: number of chunks to return.

    Returns:
        List of dicts: {text, score, page_name, page_url, query_id, chunk_idx}
    """
    q_emb = embed_text(query, embedder)
    return corpus.retrieve(q_emb, top_k=top_k)
