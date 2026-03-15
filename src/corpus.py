"""
Build, save, and load the global corpus embedding index.
All four pipelines retrieve from this shared index.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.data_loader import load_examples


@dataclass
class Corpus:
    """Holds the global chunk corpus with precomputed embeddings."""
    chunks: list[str] = field(default_factory=list)
    embeddings: np.ndarray = field(default_factory=lambda: np.empty((0, 384)))
    metadata: list[dict] = field(default_factory=list)  # page_name, page_url, query_id per chunk

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        """
        Cosine similarity search over the corpus.

        Args:
            query_embedding: 1-D numpy array (unit-normed).
            top_k: number of results to return.

        Returns:
            List of dicts with keys: text, score, page_name, page_url, query_id, chunk_idx.
        """
        if len(self.chunks) == 0:
            return []

        # Normalise query
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)

        # Dot product = cosine similarity (corpus embeddings are already unit-normed)
        scores = self.embeddings @ q_norm  # shape: (N,)

        # Top-k indices (descending)
        top_k = min(top_k, len(scores))
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = []
        for idx in top_indices:
            meta = self.metadata[idx] if idx < len(self.metadata) else {}
            results.append({
                "text": self.chunks[idx],
                "score": float(scores[idx]),
                "chunk_idx": int(idx),
                "page_name": meta.get("page_name", ""),
                "page_url": meta.get("page_url", ""),
                "query_id": meta.get("query_id", ""),
            })
        return results


def build_index(
    dataset_path: str,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    index_path: Optional[str] = None,
    limit: Optional[int] = 500,
) -> Corpus:
    """
    Build the global corpus from the dataset and embed all chunks.

    Args:
        dataset_path: path to the JSONL file.
        embedding_model_name: sentence-transformers model name.
        index_path: if given, save the built index here.
        limit: max number of dataset rows to read (None = all). Keep small for speed.

    Returns:
        A populated Corpus object.
    """
    print(f"[corpus] Loading dataset (limit={limit}) from {dataset_path} ...")
    model = SentenceTransformer(embedding_model_name)

    chunks: list[str] = []
    metadata: list[dict] = []

    for example in tqdm(load_examples(path=dataset_path, limit=limit), desc="Collecting chunks"):
        qid = example.get("interaction_id", "")
        for sr in example.get("search_results", []):
            snippet = (sr.get("page_snippet") or "").strip()
            if snippet:
                chunks.append(snippet)
                metadata.append({
                    "page_name": sr.get("page_name", ""),
                    "page_url": sr.get("page_url", ""),
                    "query_id": qid,
                })

    if not chunks:
        raise ValueError("No chunks collected — check dataset path and format.")

    print(f"[corpus] Embedding {len(chunks)} chunks with '{embedding_model_name}' ...")
    t0 = time.time()
    embeddings = model.encode(
        chunks,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,  # unit-norm for cosine via dot product
        convert_to_numpy=True,
    )
    print(f"[corpus] Embedding done in {time.time() - t0:.1f}s. Shape: {embeddings.shape}")

    corpus = Corpus(chunks=chunks, embeddings=embeddings, metadata=metadata)

    if index_path:
        save_index(corpus, index_path)

    return corpus


def save_index(corpus: Corpus, index_path: str) -> None:
    """Save corpus embeddings and metadata to disk."""
    path = Path(index_path)
    path.mkdir(parents=True, exist_ok=True)

    np.save(str(path / "embeddings.npy"), corpus.embeddings)
    with open(path / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(corpus.chunks, f, ensure_ascii=False)
    with open(path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(corpus.metadata, f, ensure_ascii=False)

    print(f"[corpus] Index saved to {path} ({len(corpus.chunks)} chunks)")


def load_index(index_path: str, embedding_model_name: str = "all-MiniLM-L6-v2") -> Corpus:
    """
    Load a previously saved corpus index from disk.

    Args:
        index_path: directory created by save_index().
        embedding_model_name: only stored for reference; not used for loading.

    Returns:
        Populated Corpus.
    """
    path = Path(index_path)
    if not path.exists():
        raise FileNotFoundError(f"Index not found: {path}. Run build_index first.")

    print(f"[corpus] Loading index from {path} ...")
    embeddings = np.load(str(path / "embeddings.npy"))
    with open(path / "chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(path / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    corpus = Corpus(chunks=chunks, embeddings=embeddings, metadata=metadata)
    print(f"[corpus] Loaded {len(chunks)} chunks. Embedding shape: {embeddings.shape}")
    return corpus
