from __future__ import annotations
from typing import List
import os

from app.utils.config import get_settings
from app.services.cache import get_cache

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


def embed_texts(texts: List[str], model: str | None = None, api_key: str | None = None,
                use_cache: bool = True) -> List[List[float]]:
    """Batch-embed texts using OpenAI embeddings API with persistent caching.

    Reads defaults from env settings if model/api_key are not provided.
    Uses disk-based cache to avoid re-embedding identical texts.
    """
    settings = get_settings()
    model = model or settings.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    api_key = api_key or settings.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not configured")
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Please install openai.")

    cache = get_cache() if use_cache else None

    # Check cache for existing embeddings
    cached_hits = {}
    texts_to_embed = []
    text_indices = []

    if cache:
        cached_hits = cache.get_embeddings_batch(texts, model)
        for i, text in enumerate(texts):
            if i not in cached_hits:
                texts_to_embed.append(text)
                text_indices.append(i)
    else:
        texts_to_embed = texts
        text_indices = list(range(len(texts)))

    # If all cached, return immediately
    if not texts_to_embed:
        return [cached_hits[i] for i in range(len(texts))]

    # Embed only the uncached texts
    client = OpenAI(api_key=api_key)
    resp = client.embeddings.create(model=model, input=texts_to_embed)
    new_embeddings = [d.embedding for d in resp.data]  # type: ignore[attr-defined]

    # Cache the new embeddings
    if cache:
        cache.set_embeddings_batch(texts_to_embed, model, new_embeddings)

    # Merge cached and new embeddings in original order
    result = [None] * len(texts)  # type: ignore
    for i, emb in cached_hits.items():
        result[i] = emb
    for idx, emb in zip(text_indices, new_embeddings):
        result[idx] = emb

    return result  # type: ignore

