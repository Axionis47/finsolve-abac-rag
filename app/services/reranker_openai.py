from __future__ import annotations
from typing import List, Dict, Any, Optional
from math import sqrt
import json
import ssl
import urllib.request

from .embeddings import embed_texts


OPENAI_API_BASE = "https://api.openai.com/v1"


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x*y for x, y in zip(a, b))
    na = sqrt(sum(x*x for x in a))
    nb = sqrt(sum(y*y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def openai_embedding_rerank(query: str, items: List[Dict[str, Any]], top_k: Optional[int] = None,
                             model: Optional[str] = None, api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Rerank items using OpenAI embeddings cosine similarity to the query.
    Uses the same embed_texts service and avoids extra deps.
    """
    if not items:
        return items
    q_emb = embed_texts([query], model=model, api_key=api_key)[0]
    texts = [it.get("text", "") for it in items]
    doc_embs = embed_texts(texts, model=model, api_key=api_key)
    scored = list(zip(range(len(items)), (_cosine(q_emb, e) for e in doc_embs)))
    scored.sort(key=lambda t: t[1], reverse=True)
    ordered = [items[i] for i, _ in scored]
    return ordered[:top_k] if top_k else ordered


def _build_llm_prompt(query: str, items: List[Dict[str, Any]]) -> str:
    lines = [
        "You are a helpful reranking assistant. Given a query and a list of snippets,",
        "return the indices of the top snippets in descending relevance order as a JSON list of integers.",
        "Only output the JSON array (e.g., [2,0,1]).",
        f"Query: {query}",
        "Snippets:",
    ]
    for idx, it in enumerate(items):
        txt = (it.get("text") or "").replace("\n", " ")
        lines.append(f"{idx}: {txt}")
    return "\n".join(lines)


def openai_llm_rerank(query: str, items: List[Dict[str, Any]], top_k: Optional[int] = None,
                       model: str = "gpt-4o-mini", api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Rerank items with an LLM instruction prompt. Avoids extra SDK deps by using urllib.
    Tests should monkeypatch this function; in production provide OPENAI_API_KEY.
    """
    if not items:
        return items
    if not api_key:
        # No key -> fallback to identity order
        return items[:top_k] if top_k else items

    url = f"{OPENAI_API_BASE}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You rerank snippets for relevance."},
            {"role": "user", "content": _build_llm_prompt(query, items)},
        ],
        "temperature": 0.0,
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
            # Expect a JSON list like [2,0,1]
            try:
                order = json.loads(text)
                if not isinstance(order, list):
                    raise ValueError("order not a list")
            except Exception:
                # If parsing fails, leave order as identity
                order = list(range(len(items)))
            # Build output in provided order
            ordered = [items[i] for i in order if 0 <= i < len(items)]
            return ordered[:top_k] if top_k else ordered
    except Exception:
        # Network or parse errors -> fallback identity
        return items[:top_k] if top_k else items

