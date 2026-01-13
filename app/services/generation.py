from __future__ import annotations
from typing import List, Dict, Any, Optional
import json
import os
import ssl
import urllib.request
from datetime import timezone, datetime

from app.services.cache import get_cache, hash_context

# Dynamically select API base based on backend
def _get_api_base() -> str:
    backend = os.getenv("LLM_BACKEND", "openai")
    if backend == "vllm":
        return os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    elif backend == "ollama":
        return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/v1"
    return "https://api.openai.com/v1"


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


def generate_answer(query: str, snippets: List[Dict[str, Any]], *, model: str, api_key: str,
                    use_cache: bool = True) -> str:
    """
    Call OpenAI chat completions to synthesize an answer from snippets.
    If api_key is empty, return a local fallback answer.
    Uses persistent cache to avoid re-generating identical responses.
    """
    if not api_key:
        # Local fallback: lightweight extractive response
        if not snippets:
            return "I don't know based on the available context."
        top = snippets[0]
        src = f"{top.get('source_path','')}#{top.get('section_path','')}".strip('#')
        return f"Based on [1], {top.get('text','')[:200]}...\n\nCitations: [1] {src}"

    # Check cache first
    cache = get_cache() if use_cache else None
    context_hash = hash_context(snippets)

    if cache:
        cached_response = cache.get_llm_response(query, context_hash, model)
        if cached_response is not None:
            return cached_response

    url = f"{_get_api_base()}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": _build_chat_prompt(query, snippets),
        "temperature": 0.2,
    }

    req = urllib.request.Request(url, method="POST", data=json.dumps(payload).encode("utf-8"))
    for k, v in headers.items():
        req.add_header(k, v)
    context = ssl.create_default_context()

    try:
        with urllib.request.urlopen(req, context=context) as resp:
            body = resp.read().decode("utf-8")
            data = json.loads(body)
            text = data["choices"][0]["message"]["content"].strip()

            # Cache the successful response
            if cache:
                cache.set_llm_response(query, context_hash, model, text)

            return text
    except Exception:
        return "I don't know based on the available context."

