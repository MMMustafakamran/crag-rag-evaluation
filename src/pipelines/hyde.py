"""
HyDE (Hypothetical Document Embedding): generate a hypothetical document,
embed it, retrieve from global index by similarity, then generate the real answer.
"""

from __future__ import annotations

from src.generation import generate_text, generate_answer


def _generate_hypothetical_doc(query: str, generator) -> str:
    """
    Ask the LLM to write a short passage that *would* contain the answer.
    This hypothetical passage is then used as the retrieval query.
    """
    prompt = (
        f"Write a short factual passage (2-3 sentences) that directly answers the following question. "
        f"Write it as if it were from an encyclopedia or reference article. "
        f"Do not say you are writing a hypothetical — just write the passage.\n\n"
        f"Question: {query}"
    )
    doc = generate_text(prompt, generator)
    return doc if doc else query  # fallback to original query if generation fails


def run(
    query: str,
    corpus,
    embedder,
    generator,
    top_k: int = 5,
) -> dict:
    """
    HyDE pipeline.

    Returns:
        {retrieved: list[dict], answer: str, meta: {hypothetical_doc: str, pipeline: str}}
    """
    from src.retrieval import embed_text

    # Step 1: Generate hypothetical document
    hyp_doc = _generate_hypothetical_doc(query, generator)

    # Step 2: Embed the hypothetical doc (not the query)
    hyp_emb = embed_text(hyp_doc, embedder)

    # Step 3: Retrieve top-k from corpus by similarity to hypothetical embedding
    retrieved = corpus.retrieve(hyp_emb, top_k=top_k)

    # Step 4: Generate final answer from retrieved chunks
    answer = generate_answer(query, retrieved, generator)

    return {
        "retrieved": retrieved,
        "answer": answer,
        "meta": {
            "pipeline": "hyde",
            "hypothetical_doc": hyp_doc,
        },
    }
