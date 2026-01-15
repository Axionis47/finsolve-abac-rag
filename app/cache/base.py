"""
Cache adapter interface and implementations.

Provides:
- Abstract CacheAdapter interface
- InMemoryAdapter with TTL support and thread safety
- RedisAdapter (optional, if redis is available)
- Factory function to get configured adapter
"""
from __future__ import annotations

import json
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, TypeVar

T = TypeVar("T")


@dataclass
class CacheEntry:
    """Cache entry with value and expiration time."""
    value: Any
    expires_at: Optional[float] = None  # Unix timestamp, None = never expires

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


class CacheAdapter(ABC):
    """Abstract cache adapter interface."""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value by key. Returns None if not found or expired."""
        ...

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value with optional TTL in seconds."""
        ...

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete a key."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear all entries."""
        ...

    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        ...

    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], T],
        ttl: Optional[int] = None,
    ) -> T:
        """Get value or compute and cache it. Default implementation (no singleflight)."""
        cached = self.get(key)
        if cached is not None:
            return cached
        value = compute_fn()
        self.set(key, value, ttl)
        return value


class InMemoryAdapter(CacheAdapter):
    """Thread-safe in-memory cache with TTL support."""

    def __init__(self, max_size: int = 10000):
        self._store: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            if entry.is_expired():
                del self._store[key]
                self._misses += 1
                return None
            self._hits += 1
            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        with self._lock:
            # Simple eviction: remove oldest entries if over limit
            if len(self._store) >= self._max_size and key not in self._store:
                self._evict_oldest()
            
            expires_at = None
            if ttl is not None:
                expires_at = time.time() + ttl
            self._store[key] = CacheEntry(value=value, expires_at=expires_at)

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0.0
            return {
                "type": "memory",
                "entries": len(self._store),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_pct": round(hit_rate, 2),
                "evictions": self._evictions,
            }

    def _evict_oldest(self) -> None:
        """Evict oldest 10% of entries."""
        to_evict = max(1, self._max_size // 10)
        keys = list(self._store.keys())[:to_evict]
        for k in keys:
            del self._store[k]
            self._evictions += 1


class NoOpAdapter(CacheAdapter):
    """No-op cache adapter for when caching is disabled."""

    def get(self, key: str) -> Optional[Any]:
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        pass

    def delete(self, key: str) -> None:
        pass

    def clear(self) -> None:
        pass

    def stats(self) -> Dict[str, Any]:
        return {"type": "noop", "enabled": False}


# Global cache instance
_cache_adapter: Optional[CacheAdapter] = None


def get_cache_adapter() -> CacheAdapter:
    """Get or create the global cache adapter based on configuration."""
    global _cache_adapter
    if _cache_adapter is None:
        backend = os.environ.get("CACHE_BACKEND", "memory")
        enabled = os.environ.get("CACHE_ENABLED", "true").lower() in ("true", "1", "yes")

        if not enabled:
            # Return a no-op adapter
            _cache_adapter = NoOpAdapter()
        elif backend == "redis":
            # Try Redis, fall back to memory
            try:
                from app.cache.redis_adapter import RedisAdapter
                redis_url = os.environ.get("CACHE_REDIS_URL", "redis://localhost:6379/0")
                _cache_adapter = RedisAdapter(redis_url)
            except Exception:
                _cache_adapter = InMemoryAdapter()
        else:
            _cache_adapter = InMemoryAdapter()
    return _cache_adapter


def reset_cache_adapter() -> None:
    """Reset the global cache adapter. Useful for testing."""
    global _cache_adapter
    if _cache_adapter is not None:
        _cache_adapter.clear()
    _cache_adapter = None

