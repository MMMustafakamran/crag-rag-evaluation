"""
Basic RAG: query -> retrieve top-k from global index -> generate answer.
Do not remove or rename this file.
"""

from __future__ import annotations

from src.generation import generate_answer
from src.retrieval import retrieve


def run(query: str, corpus, embedder, generator, top_k: int = 5) -> dict:
    retrieved = retrieve(query, embedder, corpus, top_k=top_k)
    answer = generate_answer(query, retrieved, generator)
    return {
        "retrieved": retrieved,
        "answer": answer,
        "meta": {"pipeline": "basic_rag"},
    }
