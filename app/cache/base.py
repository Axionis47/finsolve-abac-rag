"""
Cache adapter interface and implementations.

Provides:
- Abstract CacheAdapter interface
- InMemoryAdapter with LRU eviction, TTL support, and thread safety
- RedisAdapter (optional, if redis is available)
- Factory function to get configured adapter
"""
from __future__ import annotations

import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, TypeVar

T = TypeVar("T")


@dataclass
class CacheEntry:
    """Cache entry with value, expiration time, and metadata."""
    value: Any
    expires_at: Optional[float] = None  # Unix timestamp, None = never expires
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    size_bytes: int = 0  # Approximate size in bytes

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def touch(self) -> None:
        """Update last accessed time for LRU tracking."""
        self.last_accessed = time.time()


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
    """Thread-safe in-memory cache with LRU eviction, TTL, and memory bounds."""

    def __init__(
        self,
        max_size: int = 10000,
        max_memory_mb: float = 100.0,  # Maximum memory in MB
    ):
        self._store: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._max_size = max_size
        self._max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self._current_memory_bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        # LRU tracking: list of keys in access order (most recent at end)
        self._lru_order: list[str] = []

    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of a value in bytes."""
        try:
            import json
            serialized = json.dumps(value)
            return len(serialized.encode('utf-8'))
        except (TypeError, ValueError):
            return 1024  # Default estimate for non-serializable objects

    def _update_lru(self, key: str) -> None:
        """Move key to end of LRU list (most recently used)."""
        if key in self._lru_order:
            self._lru_order.remove(key)
        self._lru_order.append(key)

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            if entry.is_expired():
                self._remove_key(key)
                self._misses += 1
                return None
            # Update LRU tracking
            entry.touch()
            self._update_lru(key)
            self._hits += 1
            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        with self._lock:
            size = self._estimate_size(value)

            # If key exists, remove old entry first
            if key in self._store:
                self._remove_key(key)

            # Evict LRU entries if over limits
            while self._should_evict(size):
                if not self._evict_lru():
                    break  # No more entries to evict

            expires_at = None
            if ttl is not None:
                expires_at = time.time() + ttl

            entry = CacheEntry(value=value, expires_at=expires_at, size_bytes=size)
            self._store[key] = entry
            self._current_memory_bytes += size
            self._update_lru(key)

    def _should_evict(self, new_size: int) -> bool:
        """Check if eviction is needed for a new entry."""
        over_count = len(self._store) >= self._max_size
        over_memory = (self._current_memory_bytes + new_size) > self._max_memory_bytes
        return over_count or over_memory

    def _evict_lru(self) -> bool:
        """Evict least recently used entry. Returns True if an entry was evicted."""
        if not self._lru_order:
            return False

        # Get least recently used key
        lru_key = self._lru_order[0]
        self._remove_key(lru_key)
        self._evictions += 1
        return True

    def _remove_key(self, key: str) -> None:
        """Remove a key from the cache and update memory tracking."""
        if key in self._store:
            entry = self._store.pop(key)
            self._current_memory_bytes -= entry.size_bytes
        if key in self._lru_order:
            self._lru_order.remove(key)

    def delete(self, key: str) -> None:
        with self._lock:
            self._remove_key(key)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._lru_order.clear()
            self._current_memory_bytes = 0
            self._hits = 0
            self._misses = 0

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0.0
            memory_mb = self._current_memory_bytes / (1024 * 1024)
            max_mb = self._max_memory_bytes / (1024 * 1024)
            return {
                "type": "memory",
                "entries": len(self._store),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_pct": round(hit_rate, 2),
                "evictions": self._evictions,
                "memory_mb": round(memory_mb, 2),
                "max_memory_mb": round(max_mb, 2),
                "memory_usage_pct": round((memory_mb / max_mb) * 100, 1) if max_mb > 0 else 0,
            }


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
            except Exception as exc:
                import logging
                logging.warning("Redis cache unavailable (%s), falling back to in-memory cache", exc)
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

