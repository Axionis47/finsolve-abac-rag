from __future__ import annotations
from typing import List, Dict, Any, Optional

from app.services.cache import get_cache, hash_context
from app.cache import get_rag_cache, PolicyFingerprint


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


def generate_answer(
    query: str,
    snippets: List[Dict[str, Any]],
    *,
    model: str = None,  # noqa: ARG001
    api_key: str = None,  # noqa: ARG001
    use_cache: bool = True,
    pf: Optional[PolicyFingerprint] = None,
) -> str:
    """
    Generate an answer using the configured LLM backend (Vertex AI, OpenAI, etc.)

    Uses the provider abstraction which respects LLM_BACKEND env var.
    The model and api_key parameters are kept for backward compatibility but ignored
    when using non-OpenAI backends.

    Args:
        query: The user's question
        snippets: List of context snippets (already policy-filtered)
        model: Deprecated, ignored
        api_key: Deprecated, ignored
        use_cache: Whether to use caching
        pf: Optional PolicyFingerprint for policy-safe caching. If provided,
            the LLM cache will be scoped by this PF to prevent cross-user leakage.
    """
    _ = model, api_key  # Silence unused warnings; kept for backward compat
    from app.services.providers import get_llm

    if not snippets:
        return "I don't know based on the available context."

    llm = get_llm()
    model_name = getattr(llm, 'model', 'unknown')
    context_hash = hash_context(snippets)

    # Use policy-safe RAG cache if PF is provided, otherwise fall back to old cache
    if use_cache and pf is not None:
        rag_cache = get_rag_cache()

        def do_generation():
            messages = _build_chat_prompt(query, snippets)
            response = llm.generate(messages, temperature=0.2)
            return response.text.strip()

        try:
            answer, _ = rag_cache.get_llm_answer(
                pf=pf,
                query=query,
                context_hash=context_hash,
                model=model_name,
                compute_fn=do_generation,
            )
            return answer
        except Exception as e:
            # Fall through to error handling below
            import logging
            import traceback
            error_msg = f"LLM generation failed: {type(e).__name__}: {e}"
            logging.error(error_msg)
            logging.error(f"Full traceback:\n{traceback.format_exc()}")

            top = snippets[0]
            src = f"{top.get('source_path','')}#{top.get('section_path','')}".strip('#')
            return f"[LLM Error: {type(e).__name__}: {str(e)[:100]}] Based on [1], {top.get('text','')[:200]}...\n\nCitations: [1] {src}"

    # Legacy caching path (without PF - less safe but backward compatible)
    cache = get_cache() if use_cache else None

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
        # Log the error for debugging with full traceback
        import logging
        import traceback
        error_msg = f"LLM generation failed: {type(e).__name__}: {e}"
        logging.error(error_msg)
        logging.error(f"Full traceback:\n{traceback.format_exc()}")
        print(f"ERROR: {error_msg}", flush=True)
        print(f"Traceback:\n{traceback.format_exc()}", flush=True)

        # Fallback to extractive answer
        top = snippets[0]
        src = f"{top.get('source_path','')}#{top.get('section_path','')}".strip('#')
        return f"[LLM Error: {type(e).__name__}: {str(e)[:100]}] Based on [1], {top.get('text','')[:200]}...\n\nCitations: [1] {src}"

