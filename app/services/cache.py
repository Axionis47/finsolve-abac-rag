"""
Persistent LLM and Embedding Cache

Uses diskcache for SQLite-backed persistent caching that survives restarts.
Provides significant speedup for repeated queries by caching:
- Embedding vectors (deterministic, safe to cache indefinitely)
- LLM responses (cached by query + context hash)
"""
from __future__ import annotations
import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Callable, TypeVar
from functools import wraps

# Try to import diskcache, fall back to in-memory cache if not available
try:
    import diskcache
    HAS_DISKCACHE = True
except ImportError:
    HAS_DISKCACHE = False
    diskcache = None  # type: ignore

T = TypeVar("T")


class LLMCache:
    """Persistent cache for LLM and embedding results."""

    def __init__(self, cache_dir: str = ".llm_cache", enabled: bool = True):
        self.enabled = enabled and HAS_DISKCACHE
        self.cache_dir = cache_dir
        self._cache: Optional[Any] = None

        if self.enabled:
            os.makedirs(cache_dir, exist_ok=True)
            self._cache = diskcache.Cache(cache_dir, size_limit=500 * 1024 * 1024)  # 500MB limit

    def _hash_key(self, *args: Any) -> str:
        """Create a deterministic hash key from arguments."""
        data = json.dumps(args, sort_keys=True, default=str)
        return hashlib.sha256(data.encode()).hexdigest()

    def get_embedding(self, text: str, model: str) -> Optional[List[float]]:
        """Get cached embedding for a single text."""
        if not self.enabled or self._cache is None:
            return None
        key = f"emb:{model}:{self._hash_key(text)}"
        return self._cache.get(key)

    def set_embedding(self, text: str, model: str, embedding: List[float], expire: Optional[int] = None) -> None:
        """Cache an embedding. No expiry by default since embeddings are deterministic."""
        if not self.enabled or self._cache is None:
            return
        key = f"emb:{model}:{self._hash_key(text)}"
        self._cache.set(key, embedding, expire=expire)

    def get_embeddings_batch(self, texts: List[str], model: str) -> Dict[int, List[float]]:
        """Get cached embeddings for a batch. Returns {index: embedding} for hits."""
        if not self.enabled or self._cache is None:
            return {}
        result = {}
        for i, text in enumerate(texts):
            cached = self.get_embedding(text, model)
            if cached is not None:
                result[i] = cached
        return result

    def set_embeddings_batch(self, texts: List[str], model: str, embeddings: List[List[float]]) -> None:
        """Cache a batch of embeddings."""
        if not self.enabled or self._cache is None:
            return
        for text, emb in zip(texts, embeddings):
            self.set_embedding(text, model, emb)

    def get_llm_response(self, query: str, context_hash: str, model: str) -> Optional[str]:
        """Get cached LLM response."""
        if not self.enabled or self._cache is None:
            return None
        key = f"llm:{model}:{self._hash_key(query, context_hash)}"
        return self._cache.get(key)

    def set_llm_response(self, query: str, context_hash: str, model: str, response: str,
                         expire: int = 3600) -> None:
        """Cache an LLM response. Default 1 hour expiry since context may change."""
        if not self.enabled or self._cache is None:
            return
        key = f"llm:{model}:{self._hash_key(query, context_hash)}"
        self._cache.set(key, response, expire=expire)

    def get_rerank_result(self, query: str, items_hash: str, model: str) -> Optional[List[int]]:
        """Get cached rerank ordering."""
        if not self.enabled or self._cache is None:
            return None
        key = f"rerank:{model}:{self._hash_key(query, items_hash)}"
        return self._cache.get(key)

    def set_rerank_result(self, query: str, items_hash: str, model: str, order: List[int],
                          expire: int = 3600) -> None:
        """Cache a rerank result."""
        if not self.enabled or self._cache is None:
            return
        key = f"rerank:{model}:{self._hash_key(query, items_hash)}"
        self._cache.set(key, order, expire=expire)

    def clear(self) -> None:
        """Clear all cached data."""
        if self._cache is not None:
            self._cache.clear()

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.enabled or self._cache is None:
            return {"enabled": False, "reason": "diskcache not installed" if not HAS_DISKCACHE else "disabled"}
        return {
            "enabled": True,
            "size_bytes": self._cache.volume(),
            "size_mb": round(self._cache.volume() / (1024 * 1024), 2),
            "item_count": len(self._cache),
            "cache_dir": self.cache_dir,
        }

    def close(self) -> None:
        """Close the cache."""
        if self._cache is not None:
            self._cache.close()


# Global cache instance - lazy initialized
_global_cache: Optional[LLMCache] = None


def get_cache(cache_dir: Optional[str] = None) -> LLMCache:
    """Get or create the global cache instance."""
    global _global_cache
    if _global_cache is None:
        dir_path = cache_dir or os.environ.get("LLM_CACHE_DIR", ".llm_cache")
        enabled = os.environ.get("LLM_CACHE_ENABLED", "true").lower() in ("true", "1", "yes")
        _global_cache = LLMCache(cache_dir=dir_path, enabled=enabled)
    return _global_cache


def hash_context(snippets: List[Dict[str, Any]]) -> str:
    """Create a hash of context snippets for cache key."""
    # Use text and source as key components
    data = [(s.get("text", ""), s.get("source_path", "")) for s in snippets]
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]

