"""
Retrieval helpers: embed a query and retrieve top-k chunks from the global corpus.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np
from sentence_transformers import SentenceTransformer

# Module-level cache so the model is only loaded once per process
_embedder_cache: dict[str, object] = {}


class HashingEmbedder:
    """
    Lightweight local embedder fallback.

    It is not semantically comparable to sentence-transformers, but it keeps the
    project runnable when the environment cannot fetch models from Hugging Face.
    """

    is_fallback = True

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _encode_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = re.findall(r"\w+", (text or "").lower())
        if not tokens:
            return vec

        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            idx = int(digest[:8], 16) % self.dim
            sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
            vec[idx] += sign

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def encode(self, texts, normalize_embeddings: bool = True, convert_to_numpy: bool = True, **kwargs):
        if isinstance(texts, str):
            vec = self._encode_one(texts)
            return vec

        arr = np.stack([self._encode_one(text) for text in texts], axis=0)
        return arr


def get_embedder(model_name: str = "all-MiniLM-L6-v2"):
    """Load (and cache) a SentenceTransformer model."""
    if model_name not in _embedder_cache:
        try:
            _embedder_cache[model_name] = SentenceTransformer(model_name)
        except Exception as exc:
            print(f"[retrieval] Warning: could not load embedding model '{model_name}': {exc}")
            print("[retrieval] Falling back to local hashing embedder.")
            _embedder_cache[model_name] = HashingEmbedder()
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
