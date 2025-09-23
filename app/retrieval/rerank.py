from __future__ import annotations
from typing import List, Dict, Any, Optional
from .bm25 import BM25


def rerank_simple(query: str, items: List[Dict[str, Any]], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Simple local reranker using BM25 over item["text"].
    Preserves only rank order among the provided items; does not fetch new items.
    """
    texts = [it.get("text", "") for it in items]
    bm = BM25(texts)
    # score only over provided items
    q_tokens = bm.tokenize(query) if hasattr(bm, "tokenize") else None  # fallback guard
    scored = []
    for i in range(len(items)):
        s = bm.score(q_tokens, i) if q_tokens is not None else 0.0
        scored.append((i, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    order = [i for i, _ in scored]
    ordered = [items[i] for i in order]
    return ordered[:top_k] if top_k else ordered


def rerank(query: str, items: List[Dict[str, Any]], backend: str = "simple", top_k: Optional[int] = None, **kwargs) -> List[Dict[str, Any]]:
    """
    Pluggable reranker.
    - backend="simple": BM25-based local reranker (no network)
    - backend="openai" or "openai_llm": use provided callable via kwargs["openai_rerank_fn"]
    You can pass a custom callable under kwargs["openai_rerank_fn"] for testing or runtime.
    """
    if backend in ("openai", "openai_llm"):
        fn = kwargs.get("openai_rerank_fn")
        if callable(fn):
            return fn(query, items, top_k=top_k)
        # Fallback to simple if no callable provided
    return rerank_simple(query, items, top_k=top_k)

