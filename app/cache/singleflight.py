"""
Singleflight request coalescing to prevent cache stampedes.

When multiple concurrent requests need the same uncached value,
only one request computes it while others wait and share the result.
This prevents thundering herd problems.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class _InFlight:
    """Represents an in-flight computation."""
    lock: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: Optional[Exception] = None
    waiters: int = 0


class Singleflight:
    """
    Singleflight ensures that only one computation runs for a given key
    even when multiple callers request it concurrently.
    
    Usage:
        sf = Singleflight()
        
        # Multiple threads calling this will share the same computation
        result = sf.do("my-key", lambda: expensive_operation())
    """

    DEFAULT_TIMEOUT = 30.0  # seconds

    def __init__(self, timeout: float = DEFAULT_TIMEOUT):
        self._in_flight: Dict[str, _InFlight] = {}
        self._lock = threading.Lock()
        self._dedup_count = 0
        self._timeout_count = 0
        self._timeout = timeout

    def do(self, key: str, fn: Callable[[], T], timeout: Optional[float] = None) -> T:
        """
        Execute fn() for key, deduplicating concurrent calls.

        If another caller is already computing for this key,
        wait for their result instead of computing again.
        Waits at most `timeout` seconds before falling back to direct execution.

        Args:
            key: Unique identifier for this computation
            fn: Function to call if no computation is in progress
            timeout: Max seconds to wait for an in-flight result (default: instance timeout)

        Returns:
            Result of fn() (possibly from another caller)

        Raises:
            Exception: If fn() raises an exception
        """
        wait_timeout = timeout if timeout is not None else self._timeout

        with self._lock:
            if key in self._in_flight:
                # Someone else is computing, wait for them
                flight = self._in_flight[key]
                flight.waiters += 1
                self._dedup_count += 1
            else:
                # We're the first, create in-flight entry
                flight = _InFlight()
                self._in_flight[key] = flight
                flight.waiters = 0

        if flight.waiters > 0:
            # Wait for the computing thread with timeout
            completed = flight.lock.wait(timeout=wait_timeout)
            if not completed:
                self._timeout_count += 1
                logger.warning("Singleflight timeout for key=%s after %.1fs, executing directly", key[:30], wait_timeout)
                return fn()
            if flight.error is not None:
                raise flight.error
            return flight.result

        # We're the computing thread
        try:
            result = fn()
            flight.result = result
        except Exception as e:
            flight.error = e
            raise
        finally:
            # Signal all waiters
            flight.lock.set()
            # Clean up
            with self._lock:
                del self._in_flight[key]

        return result

    @property
    def dedup_count(self) -> int:
        """Number of deduplicated calls (waiters that shared results)."""
        return self._dedup_count

    @property
    def in_flight_count(self) -> int:
        """Number of currently in-flight computations."""
        with self._lock:
            return len(self._in_flight)

    def stats(self) -> Dict[str, Any]:
        """Return singleflight statistics."""
        return {
            "dedup_count": self._dedup_count,
            "timeout_count": self._timeout_count,
            "in_flight": self.in_flight_count,
        }


# Global singleflight instance for cache operations
_singleflight: Optional[Singleflight] = None


def get_singleflight() -> Singleflight:
    """Get the global singleflight instance."""
    global _singleflight
    if _singleflight is None:
        _singleflight = Singleflight()
    return _singleflight

