"""
Rewrite-Retrieve-Read: rewrite query for clarity, then retrieve from global index, then generate answer.
Do not remove or rename this file.
"""

from __future__ import annotations

from src.generation import generate_answer, generate_text
from src.retrieval import retrieve


def _rewrite_query(query: str, generator) -> str:
    prompt = (
        "Rewrite the question as a concise search query without changing its meaning.\n\n"
        f"Question: {query}\n\nRewritten query:"
    )
    rewritten = generate_text(prompt, generator).strip()
    return rewritten or query


def run(query, corpus, embedder, generator, top_k: int = 5) -> dict:
    rewritten_query = _rewrite_query(query, generator)
    retrieved = retrieve(rewritten_query, embedder, corpus, top_k=top_k)
    answer = generate_answer(query, retrieved, generator)
    return {
        "retrieved": retrieved,
        "answer": answer,
        "meta": {"pipeline": "rrr", "rewritten_query": rewritten_query},
    }
