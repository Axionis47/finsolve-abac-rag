"""
Redis cache adapter implementation.

Optional adapter that uses Redis for distributed caching.
Falls back to InMemoryAdapter if Redis is not available.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from app.cache.base import CacheAdapter

# Try to import redis
try:
    import redis
    HAS_REDIS = True
except ImportError:
    redis = None  # type: ignore
    HAS_REDIS = False


class RedisAdapter(CacheAdapter):
    """Redis-backed cache adapter with JSON serialization."""

    def __init__(self, redis_url: Optional[str] = None):
        if not HAS_REDIS:
            raise RuntimeError("redis package not installed. Install with: pip install redis")
        
        url = redis_url or os.environ.get("CACHE_REDIS_URL", "redis://localhost:6379/0")
        self._client: redis.Redis = redis.from_url(url, decode_responses=True)
        self._hits = 0
        self._misses = 0
        self._key_prefix = os.environ.get("CACHE_KEY_PREFIX", "abac_rag:")

    def _make_key(self, key: str) -> str:
        """Add prefix to key for namespace isolation."""
        return f"{self._key_prefix}{key}"

    def get(self, key: str) -> Optional[Any]:
        try:
            value = self._client.get(self._make_key(key))
            if value is None:
                self._misses += 1
                return None
            self._hits += 1
            return json.loads(value)
        except (redis.RedisError, json.JSONDecodeError):
            self._misses += 1
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        try:
            serialized = json.dumps(value)
            if ttl is not None:
                self._client.setex(self._make_key(key), ttl, serialized)
            else:
                self._client.set(self._make_key(key), serialized)
        except (redis.RedisError, TypeError):
            pass  # Silently fail on cache errors

    def delete(self, key: str) -> None:
        try:
            self._client.delete(self._make_key(key))
        except redis.RedisError:
            pass

    def clear(self) -> None:
        """Clear all keys with our prefix."""
        try:
            # Use SCAN to find keys with our prefix (safe for production)
            cursor = 0
            pattern = f"{self._key_prefix}*"
            while True:
                cursor, keys = self._client.scan(cursor, match=pattern, count=100)
                if keys:
                    self._client.delete(*keys)
                if cursor == 0:
                    break
            self._hits = 0
            self._misses = 0
        except redis.RedisError:
            pass

    def stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0.0
        
        # Try to get Redis info
        info = {}
        try:
            redis_info = self._client.info("memory")
            info["redis_memory_used"] = redis_info.get("used_memory_human", "unknown")
        except redis.RedisError:
            info["redis_memory_used"] = "unavailable"
        
        # Count keys with our prefix
        key_count = 0
        try:
            cursor = 0
            pattern = f"{self._key_prefix}*"
            while True:
                cursor, keys = self._client.scan(cursor, match=pattern, count=1000)
                key_count += len(keys)
                if cursor == 0:
                    break
        except redis.RedisError:
            key_count = -1
        
        return {
            "type": "redis",
            "entries": key_count,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_pct": round(hit_rate, 2),
            **info,
        }

    def ping(self) -> bool:
        """Check if Redis is reachable."""
        try:
            return self._client.ping()
        except redis.RedisError:
            return False

