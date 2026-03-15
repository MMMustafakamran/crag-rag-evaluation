"""
RAG Fusion: generate multiple query variants, retrieve from global index for each,
merge ranked lists with Reciprocal Rank Fusion (RRF), then generate answer.
"""

from __future__ import annotations

from collections import defaultdict

from src.generation import generate_text, generate_answer


def _generate_query_variants(query: str, generator, n: int = 5) -> list[str]:
    """Use LLM to generate n alternative phrasings of the query."""
    prompt = (
        f"Generate {n} different search queries that would help answer the following question. "
        f"Each query should approach the topic from a different angle. "
        f"Output ONLY the queries, one per line, no numbering or extra text.\n\n"
        f"Original question: {query}"
    )
    raw = generate_text(prompt, generator)
    variants = [line.strip() for line in raw.splitlines() if line.strip()]
    # Always include the original query
    all_queries = [query] + variants[:n]
    return all_queries


def _reciprocal_rank_fusion(ranked_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.

    RRF score = Σ 1 / (k + rank)   for each list in which the document appears.

    Args:
        ranked_lists: each sub-list is [{text, score, chunk_idx, ...}] in rank order.
        k: RRF constant (typically 60).

    Returns:
        Merged and re-ranked list of chunk dicts with an added 'rrf_score' field.
    """
    rrf_scores: dict[int, float] = defaultdict(float)
    chunk_store: dict[int, dict] = {}

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            idx = chunk["chunk_idx"]
            rrf_scores[idx] += 1.0 / (k + rank)
            if idx not in chunk_store:
                chunk_store[idx] = chunk

    # Sort by RRF score descending
    sorted_idxs = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)
    fused = []
    for idx in sorted_idxs:
        c = dict(chunk_store[idx])
        c["rrf_score"] = rrf_scores[idx]
        c["score"] = rrf_scores[idx]  # unify score field for frontend
        fused.append(c)
    return fused


def run(
    query: str,
    corpus,
    embedder,
    generator,
    top_k: int = 5,
) -> dict:
    """
    RAG Fusion pipeline.

    Returns:
        {retrieved: list[dict], answer: str, meta: {variants: list[str], pipeline: str}}
    """
    from src.retrieval import embed_text

    # Step 1: Generate query variants
    variants = _generate_query_variants(query, generator, n=4)

    # Step 2: Retrieve top-k from corpus for each variant
    ranked_lists = []
    for variant in variants:
        q_emb = embed_text(variant, embedder)
        results = corpus.retrieve(q_emb, top_k=top_k * 2)  # retrieve more before fusion
        ranked_lists.append(results)

    # Step 3: Fuse with RRF
    fused = _reciprocal_rank_fusion(ranked_lists)
    top_chunks = fused[:top_k]

    # Step 4: Generate answer
    answer = generate_answer(query, top_chunks, generator)

    return {
        "retrieved": top_chunks,
        "answer": answer,
        "meta": {
            "pipeline": "rag_fusion",
            "variants": variants,
            "num_lists_fused": len(ranked_lists),
        },
    }
