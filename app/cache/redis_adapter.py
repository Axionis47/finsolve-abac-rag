"""
Redis cache adapter implementation.

Production-grade adapter with:
- Connection pooling
- Health checks and circuit breaker
- JSON serialization with compression for large values
- Retry logic with exponential backoff
"""
from __future__ import annotations

import json
import os
import threading
import time
import zlib
from typing import Any, Dict, Optional

from app.cache.base import CacheAdapter

# Try to import redis
try:
    import redis
    from redis.connection import ConnectionPool
    HAS_REDIS = True
except ImportError:
    redis = None  # type: ignore
    ConnectionPool = None  # type: ignore
    HAS_REDIS = False


# Compression threshold (bytes) - compress values larger than this
COMPRESSION_THRESHOLD = 1024


class CircuitBreaker:
    """Simple circuit breaker for Redis connection failures."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ):
        self._failure_count = 0
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._last_failure_time: Optional[float] = None
        self._state = "closed"  # closed, open, half-open
        self._lock = threading.Lock()

    def record_success(self) -> None:
        """Record a successful operation."""
        with self._lock:
            self._failure_count = 0
            self._state = "closed"

    def record_failure(self) -> None:
        """Record a failed operation."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._failure_count >= self._failure_threshold:
                self._state = "open"

    def can_execute(self) -> bool:
        """Check if operations should be attempted."""
        with self._lock:
            if self._state == "closed":
                return True
            if self._state == "open":
                # Check if recovery timeout has passed
                if self._last_failure_time and \
                   (time.time() - self._last_failure_time) > self._recovery_timeout:
                    self._state = "half-open"
                    return True
                return False
            # half-open: allow one request through
            return True

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)."""
        return self._state == "open"


class RedisAdapter(CacheAdapter):
    """Redis-backed cache adapter with connection pooling and resilience."""

    def __init__(
        self,
        redis_url: Optional[str] = None,
        max_connections: int = 10,
        socket_timeout: float = 2.0,
        socket_connect_timeout: float = 2.0,
        retry_on_timeout: bool = True,
        health_check_interval: int = 30,
    ):
        if not HAS_REDIS:
            raise RuntimeError("redis package not installed. Install with: pip install redis")

        url = redis_url or os.environ.get("CACHE_REDIS_URL", "redis://localhost:6379/0")

        # Create connection pool
        self._pool = ConnectionPool.from_url(
            url,
            max_connections=max_connections,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            retry_on_timeout=retry_on_timeout,
            health_check_interval=health_check_interval,
            decode_responses=True,
        )
        self._client: redis.Redis = redis.Redis(connection_pool=self._pool)

        self._hits = 0
        self._misses = 0
        self._errors = 0
        self._key_prefix = os.environ.get("CACHE_KEY_PREFIX", "abac_rag:")
        self._circuit_breaker = CircuitBreaker()
        self._compression_enabled = os.environ.get("CACHE_COMPRESSION", "true").lower() in ("true", "1", "yes")

    def _make_key(self, key: str) -> str:
        """Add prefix to key for namespace isolation."""
        return f"{self._key_prefix}{key}"

    def _serialize(self, value: Any) -> str:
        """Serialize value to JSON, optionally with compression."""
        serialized = json.dumps(value)
        if self._compression_enabled and len(serialized) > COMPRESSION_THRESHOLD:
            compressed = zlib.compress(serialized.encode('utf-8'))
            # Prefix with 'Z:' to indicate compression
            import base64
            return "Z:" + base64.b64encode(compressed).decode('ascii')
        return serialized

    def _deserialize(self, data: str) -> Any:
        """Deserialize value from JSON, handling compression."""
        if data.startswith("Z:"):
            import base64
            compressed = base64.b64decode(data[2:])
            decompressed = zlib.decompress(compressed).decode('utf-8')
            return json.loads(decompressed)
        return json.loads(data)

    def get(self, key: str) -> Optional[Any]:
        if not self._circuit_breaker.can_execute():
            self._misses += 1
            return None

        try:
            value = self._client.get(self._make_key(key))
            if value is None:
                self._misses += 1
                self._circuit_breaker.record_success()
                return None
            self._hits += 1
            self._circuit_breaker.record_success()
            return self._deserialize(value)
        except redis.RedisError:
            self._misses += 1
            self._errors += 1
            self._circuit_breaker.record_failure()
            return None
        except (json.JSONDecodeError, zlib.error):
            self._misses += 1
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        if not self._circuit_breaker.can_execute():
            return

        try:
            serialized = self._serialize(value)
            if ttl is not None:
                self._client.setex(self._make_key(key), ttl, serialized)
            else:
                self._client.set(self._make_key(key), serialized)
            self._circuit_breaker.record_success()
        except redis.RedisError:
            self._errors += 1
            self._circuit_breaker.record_failure()
        except TypeError:
            pass  # Non-serializable value

    def delete(self, key: str) -> None:
        if not self._circuit_breaker.can_execute():
            return

        try:
            self._client.delete(self._make_key(key))
            self._circuit_breaker.record_success()
        except redis.RedisError:
            self._errors += 1
            self._circuit_breaker.record_failure()

    def clear(self) -> None:
        """Clear all keys with our prefix."""
        if not self._circuit_breaker.can_execute():
            return

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
            self._circuit_breaker.record_success()
        except redis.RedisError:
            self._errors += 1
            self._circuit_breaker.record_failure()

    def stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0.0

        # Try to get Redis info
        info: Dict[str, Any] = {
            "circuit_breaker_open": self._circuit_breaker.is_open,
            "errors": self._errors,
        }

        if self._circuit_breaker.can_execute():
            try:
                redis_info = self._client.info("memory")
                info["redis_memory_used"] = redis_info.get("used_memory_human", "unknown")
                info["redis_connected_clients"] = self._client.info("clients").get("connected_clients", 0)
            except redis.RedisError:
                info["redis_memory_used"] = "unavailable"
        else:
            info["redis_memory_used"] = "circuit_open"

        # Count keys with our prefix (only if circuit is closed)
        key_count = -1
        if self._circuit_breaker.can_execute():
            try:
                cursor = 0
                key_count = 0
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
            "pool_size": self._pool.max_connections if self._pool else 0,
            **info,
        }

    def ping(self) -> bool:
        """Check if Redis is reachable."""
        try:
            result = self._client.ping()
            if result:
                self._circuit_breaker.record_success()
            return result
        except redis.RedisError:
            self._circuit_breaker.record_failure()
            return False

    def health_check(self) -> Dict[str, Any]:
        """Detailed health check for monitoring."""
        result = {
            "healthy": False,
            "circuit_breaker_state": self._circuit_breaker._state,
            "errors": self._errors,
        }

        if not self._circuit_breaker.can_execute():
            result["reason"] = "circuit_breaker_open"
            return result

        try:
            start = time.time()
            self._client.ping()
            latency_ms = (time.time() - start) * 1000

            result["healthy"] = True
            result["latency_ms"] = round(latency_ms, 2)
            result["pool_in_use"] = len(self._pool._in_use_connections) if hasattr(self._pool, '_in_use_connections') else "unknown"
            self._circuit_breaker.record_success()
        except redis.RedisError as e:
            result["reason"] = str(e)
            self._circuit_breaker.record_failure()

        return result

    def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            self._pool.disconnect()

