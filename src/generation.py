"""
LLM answer generation using Google Gemini (free tier).
Do not use OpenAI. Uses google-generativeai SDK.
"""

from __future__ import annotations

import google.generativeai as genai


_generator_cache: dict[str, genai.GenerativeModel] = {}


class FallbackGenerator:
    """
    Local fallback used when no API key is configured.

    This keeps evaluation and the frontend functional in offline/demo setups.
    """

    is_fallback = True

    def __init__(self, model_name: str = "fallback-extractive") -> None:
        self.model_name = model_name


def get_generator(api_key: str, model_name: str = "gemini-1.5-flash"):
    """
    Configure Gemini and return a cached GenerativeModel.

    Args:
        api_key: Google Gemini API key.
        model_name: Gemini model name (e.g. gemini-1.5-flash).

    Returns:
        Configured GenerativeModel instance.
    """
    if not api_key:
        return FallbackGenerator()

    cache_key = f"{api_key[:8]}:{model_name}"
    if cache_key not in _generator_cache:
        genai.configure(api_key=api_key)
        _generator_cache[cache_key] = genai.GenerativeModel(model_name)
    return _generator_cache[cache_key]


def _extractive_fallback_answer(query: str, chunks: list[dict], cite: bool = False) -> str:
    """
    Cheap fallback answer when the external LLM is unavailable.
    """
    if not chunks:
        return "I don't know."

    best = max(chunks, key=lambda chunk: chunk.get("score", 0))
    answer = best.get("text", "").strip() or "I don't know."
    if cite:
        source = best.get("page_name") or best.get("page_url") or "Retrieved source"
        answer = f"{answer}\n\nSource: {source}"
    return answer


def build_prompt(query: str, chunks: list[dict], cite: bool = False) -> str:
    """
    Build a prompt for the LLM from the query and retrieved chunks.

    Args:
        query: user question.
        chunks: list of dicts with at least 'text' key (may also have 'page_name', 'page_url').
        cite: if True, ask the model to include APA citations (used by CRAG).

    Returns:
        Formatted prompt string.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        snippet = chunk.get("text", "").strip()
        url = chunk.get("page_url", "")
        name = chunk.get("page_name", f"Source {i}")
        label = f"[{i}] {name} ({url})" if url else f"[{i}] {name}"
        context_parts.append(f"{label}\n{snippet}")

    context_str = "\n\n".join(context_parts) if context_parts else "(no retrieved context)"

    citation_instruction = (
        "\n\nCite your sources at the end of the answer using APA format. "
        "Example: (Author/Site, Year). Use the source labels [1], [2], etc. above as references."
        if cite else ""
    )

    prompt = (
        f"You are a helpful factual assistant. Answer the question using ONLY the provided context. "
        f"Be concise and accurate. If the context does not contain the answer, say 'I don't know.'"
        f"{citation_instruction}\n\n"
        f"### Context\n{context_str}\n\n"
        f"### Question\n{query}\n\n"
        f"### Answer"
    )
    return prompt


def generate_answer(
    query: str,
    chunks: list[dict],
    generator: genai.GenerativeModel,
    cite: bool = False,
) -> str:
    """
    Generate an answer from the query and retrieved chunks.

    Args:
        query: user question.
        chunks: retrieved context chunks.
        generator: Gemini GenerativeModel instance.
        cite: request APA citations in the answer (for CRAG).

    Returns:
        Generated answer string.
    """
    if getattr(generator, "is_fallback", False):
        return _extractive_fallback_answer(query, chunks, cite=cite)

    if not chunks:
        # No-retrieval fallback: answer from parametric knowledge only
        prompt = (
            f"Answer the following question concisely and factually. "
            f"If you are not sure, say 'I don't know.'\n\n"
            f"Question: {query}\n\nAnswer:"
        )
    else:
        prompt = build_prompt(query, chunks, cite=cite)

    for attempt in range(3):
        try:
            response = generator.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                continue
            return _extractive_fallback_answer(query, chunks, cite=cite)
    return _extractive_fallback_answer(query, chunks, cite=cite)


def generate_text(prompt: str, generator: genai.GenerativeModel) -> str:
    """
    Raw text generation from an arbitrary prompt.
    Used by pipelines that need intermediate LLM calls (e.g. HyDE, RAG Fusion).

    Returns:
        Generated text string, or empty string on error.
    """
    if getattr(generator, "is_fallback", False):
        return ""

    for attempt in range(3):
        try:
            response = generator.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                continue
            print(f"[generation] Warning: LLM call failed: {e}")
            return ""
    return ""
