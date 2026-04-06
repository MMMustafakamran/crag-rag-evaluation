"""
Multi-Query RAG: generate multiple queries, retrieve from global index for each, merge results, generate answer.
Do not remove or rename this file.
"""

from __future__ import annotations

from src.generation import generate_answer
from src.pipelines.rag_fusion import _generate_query_variants, _reciprocal_rank_fusion
from src.retrieval import embed_text


def run(query, corpus, embedder, generator, top_k: int = 5) -> dict:
    queries = _generate_query_variants(query, generator, n=3)
    ranked_lists = [corpus.retrieve(embed_text(q, embedder), top_k=top_k) for q in queries]
    retrieved = _reciprocal_rank_fusion(ranked_lists)[:top_k]
    answer = generate_answer(query, retrieved, generator)
    return {
        "retrieved": retrieved,
        "answer": answer,
        "meta": {"pipeline": "multi_query_rag", "queries": queries},
    }
