from __future__ import annotations
from typing import List, Dict, Any

from app.services.cache import get_cache, hash_context


def _build_chat_prompt(query: str, snippets: List[Dict[str, Any]]) -> list:
    system = (
        "You are a careful assistant. Answer ONLY using the provided context snippets. "
        "Cite sources in-line as [#] and include a final 'Citations' section listing source and section. "
        "If the answer is not in the context, reply: 'I don't know based on the available context.'"
    )
    # Build a compact context block
    lines = ["Query:", query, "\nContext snippets:"]
    for i, s in enumerate(snippets, start=1):
        txt = (s.get("text") or "").strip().replace("\n", " ")
        src = s.get("source_path") or ""
        sec = s.get("section_path") or ""
        lines.append(f"[{i}] {txt}\n(Source: {src}#{sec})")
    user_msg = "\n".join(lines)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]


def generate_answer(query: str, snippets: List[Dict[str, Any]], *, model: str = None, api_key: str = None,  # noqa: ARG001
                    use_cache: bool = True) -> str:
    """
    Generate an answer using the configured LLM backend (Vertex AI, OpenAI, etc.)

    Uses the provider abstraction which respects LLM_BACKEND env var.
    The model and api_key parameters are kept for backward compatibility but ignored
    when using non-OpenAI backends.
    """
    _ = model, api_key  # Silence unused warnings; kept for backward compat
    from app.services.providers import get_llm

    if not snippets:
        return "I don't know based on the available context."

    # Check cache first
    cache = get_cache() if use_cache else None
    llm = get_llm()
    model_name = getattr(llm, 'model', 'unknown')
    context_hash = hash_context(snippets)

    if cache:
        cached_response = cache.get_llm_response(query, context_hash, model_name)
        if cached_response is not None:
            return cached_response

    # Build messages and generate
    messages = _build_chat_prompt(query, snippets)

    try:
        response = llm.generate(messages, temperature=0.2)
        text = response.text.strip()

        # Cache the successful response
        if cache:
            cache.set_llm_response(query, context_hash, model_name, text)

        return text
    except Exception as e:
        # Fallback to extractive answer
        top = snippets[0]
        src = f"{top.get('source_path','')}#{top.get('section_path','')}".strip('#')
        return f"Based on [1], {top.get('text','')[:200]}...\n\nCitations: [1] {src}"

