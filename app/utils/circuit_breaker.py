"""
Circuit Breaker Pattern Implementation.

Prevents cascading failures by failing fast when a service is unhealthy.

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Service is failing, requests fail immediately
- HALF_OPEN: Testing if service has recovered

Usage:
    breaker = CircuitBreaker("llm-api")

    try:
        result = breaker.call(lambda: call_llm_api())
    except CircuitOpenError:
        # Handle circuit open (fail fast)
        pass
"""
from __future__ import annotations
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, TypeVar, Optional, Any

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when circuit is open and requests are being rejected."""

    def __init__(self, name: str, time_until_retry: float):
        self.name = name
        self.time_until_retry = time_until_retry
        super().__init__(
            f"Circuit '{name}' is open. Retry in {time_until_retry:.1f}s"
        )


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3
    success_threshold: int = 2
    excluded_exceptions: tuple = field(default_factory=tuple)


class CircuitBreaker:
    """
    Circuit breaker for external service calls.

    Tracks failures and opens the circuit when threshold is exceeded,
    preventing further calls until recovery timeout expires.
    """

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            return self._get_state()

    def _get_state(self) -> CircuitState:
        """Internal state getter (must hold lock)."""
        if self._state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                self._success_count = 0
        return self._state

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self._last_failure_time is None:
            return True
        elapsed = time.time() - self._last_failure_time
        return elapsed >= self.config.recovery_timeout

    def _record_success(self) -> None:
        """Record a successful call (must hold lock)."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.config.success_threshold:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._success_count = 0
        else:
            self._failure_count = 0

    def _record_failure(self) -> None:
        """Record a failed call (must hold lock)."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
        elif self._failure_count >= self.config.failure_threshold:
            self._state = CircuitState.OPEN

    def call(
        self,
        func: Callable[[], T],
        fallback: Optional[Callable[[], T]] = None,
    ) -> T:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            fallback: Optional fallback function if circuit is open

        Returns:
            Result of func() or fallback()

        Raises:
            CircuitOpenError: If circuit is open and no fallback provided
        """
        with self._lock:
            state = self._get_state()

            if state == CircuitState.OPEN:
                if fallback is not None:
                    return fallback()
                time_until_retry = (
                    self.config.recovery_timeout
                    - (time.time() - (self._last_failure_time or 0))
                )
                raise CircuitOpenError(self.name, max(0, time_until_retry))

            if state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    if fallback is not None:
                        return fallback()
                    raise CircuitOpenError(self.name, 0)
                self._half_open_calls += 1

        # Execute outside lock
        try:
            result = func()
            with self._lock:
                self._record_success()
            return result
        except Exception as e:
            if isinstance(e, self.config.excluded_exceptions):
                raise
            with self._lock:
                self._record_failure()
            raise

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self._half_open_calls = 0

    def stats(self) -> dict:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "last_failure_time": self._last_failure_time,
            }


# Global circuit breakers for common services
_breakers: dict[str, CircuitBreaker] = {}
_breakers_lock = threading.Lock()


def get_circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None,
) -> CircuitBreaker:
    """Get or create a named circuit breaker."""
    with _breakers_lock:
        if name not in _breakers:
            _breakers[name] = CircuitBreaker(name, config)
        return _breakers[name]


def reset_all_breakers() -> None:
    """Reset all circuit breakers (useful for testing)."""
    with _breakers_lock:
        for breaker in _breakers.values():
            breaker.reset()
