from __future__ import annotations
from typing import List
import os

from app.utils.config import get_settings

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


def embed_texts(texts: List[str], model: str | None = None, api_key: str | None = None) -> List[List[float]]:
    """Batch-embed texts using OpenAI embeddings API.

    Reads defaults from env settings if model/api_key are not provided.
    """
    settings = get_settings()
    model = model or settings.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    api_key = api_key or settings.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not configured")
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Please install openai.")

    client = OpenAI(api_key=api_key)
    resp = client.embeddings.create(model=model, input=texts)
    # resp.data is a list in the same order
    return [d.embedding for d in resp.data]  # type: ignore[attr-defined]

