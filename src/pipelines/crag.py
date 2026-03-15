"""
CRAG (Corrective RAG): retrieve from global index, assess confidence,
use retrieved chunks if confidence is high, fall back otherwise.
Answer must include APA-style source citations.
"""

from __future__ import annotations

import numpy as np

from src.generation import generate_answer

# Confidence threshold: if avg cosine similarity of top-k chunks to query < this, retrieval is unreliable
CONFIDENCE_THRESHOLD = 0.40


def _assess_confidence(query_embedding: np.ndarray, retrieved: list[dict]) -> tuple[float, bool]:
    """
    Compute average cosine similarity of retrieved chunks to the query embedding.
    Since corpus embeddings are already unit-normed and the query is embedded the
    same way, the score field IS the cosine similarity.

    Returns:
        (avg_score, is_high_confidence)
    """
    if not retrieved:
        return 0.0, False
    avg_score = float(np.mean([c["score"] for c in retrieved]))
    return avg_score, avg_score >= CONFIDENCE_THRESHOLD


def run(
    query: str,
    corpus,
    embedder,
    generator,
    top_k: int = 5,
) -> dict:
    """
    CRAG pipeline.

    Strategy:
    - Retrieve top-k chunks from the global index.
    - Assess average retrieval confidence (cosine similarity).
    - HIGH confidence (>= threshold): generate answer from retrieved chunks with citations.
    - LOW confidence (< threshold): generate from query alone (no retrieved context).

    Returns:
        {retrieved: list[dict], answer: str, meta: {confidence: float, used_retrieval: bool, pipeline: str}}
    """
    from src.retrieval import embed_text

    # Step 1: Retrieve
    q_emb = embed_text(query, embedder)
    retrieved = corpus.retrieve(q_emb, top_k=top_k)

    # Step 2: Assess confidence
    confidence, high_confidence = _assess_confidence(q_emb, retrieved)

    # Step 3: Generate with or without retrieval context
    if high_confidence:
        # Use retrieved chunks; request APA citations
        answer = generate_answer(query, retrieved, generator, cite=True)
        used_retrieval = True
    else:
        # Retrieval is unreliable — generate from parametric knowledge
        answer = generate_answer(query, [], generator, cite=False)
        # Append note explaining fallback
        answer += (
            f"\n\n*Note: Retrieval confidence was low ({confidence:.2f} < {CONFIDENCE_THRESHOLD}). "
            f"This answer was generated without retrieved context.*"
        )
        used_retrieval = False

    return {
        "retrieved": retrieved,  # always return retrieved for inspection
        "answer": answer,
        "meta": {
            "pipeline": "crag",
            "confidence": confidence,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "used_retrieval": used_retrieval,
        },
    }
