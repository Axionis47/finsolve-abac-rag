from __future__ import annotations
from typing import List, Dict, Any, Optional

from app.services.cache import hash_context
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

    # Determine which PF to use for caching
    effective_pf = pf
    if use_cache and effective_pf is None:
        import logging
        logging.warning("generate_answer called without PolicyFingerprint; using fallback PF")
        effective_pf = PolicyFingerprint(role="unknown")

    def _do_generation():
        messages = _build_chat_prompt(query, snippets)
        response = llm.generate(messages, temperature=0.2)
        return response.text.strip()

    # If caching disabled, generate directly
    if not use_cache:
        try:
            return _do_generation()
        except Exception as e:
            import logging
            import traceback
            logging.error("LLM generation failed: %s: %s", type(e).__name__, e)
            logging.error("Full traceback:\n%s", traceback.format_exc())
            top = snippets[0]
            src = f"{top.get('source_path','')}#{top.get('section_path','')}".strip('#')
            return f"[LLM Error: {type(e).__name__}: {str(e)[:100]}] Based on [1], {top.get('text','')[:200]}...\n\nCitations: [1] {src}"

    # Use policy-safe RAG cache scoped by PF
    rag_cache = get_rag_cache()
    try:
        answer, _ = rag_cache.get_llm_answer(
            pf=effective_pf,
            query=query,
            context_hash=context_hash,
            model=model_name,
            compute_fn=_do_generation,
        )
        return answer
    except Exception as e:
        import logging
        import traceback
        logging.error("LLM generation failed: %s: %s", type(e).__name__, e)
        logging.error("Full traceback:\n%s", traceback.format_exc())
        top = snippets[0]
        src = f"{top.get('source_path','')}#{top.get('section_path','')}".strip('#')
        return f"[LLM Error: {type(e).__name__}: {str(e)[:100]}] Based on [1], {top.get('text','')[:200]}...\n\nCitations: [1] {src}"

