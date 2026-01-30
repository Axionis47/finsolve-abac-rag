"""
Tests for circuit breaker utility.
"""
import pytest
import time

from app.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    CircuitOpenError,
    CircuitBreakerConfig,
    get_circuit_breaker,
    reset_all_breakers,
)


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    def test_initial_state_is_closed(self):
        """Circuit starts in CLOSED state."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test", config)
        assert cb.state == CircuitState.CLOSED

    def test_successful_calls_keep_circuit_closed(self):
        """Successful calls don't trip the circuit."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test", config)

        def success():
            return "ok"

        for _ in range(5):
            result = cb.call(success)
            assert result == "ok"

        assert cb.state == CircuitState.CLOSED

    def test_failures_increment_counter(self):
        """Failures increment the failure counter."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test", config)

        def fail():
            raise ValueError("error")

        with pytest.raises(ValueError):
            cb.call(fail)

        stats = cb.stats()
        assert stats["failure_count"] == 1
        assert cb.state == CircuitState.CLOSED

    def test_threshold_trips_circuit(self):
        """Reaching failure threshold opens the circuit."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test", config)

        def fail():
            raise ValueError("error")

        for _ in range(3):
            with pytest.raises(ValueError):
                cb.call(fail)

        assert cb.state == CircuitState.OPEN

    def test_open_circuit_raises_immediately(self):
        """Open circuit raises without calling function."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=60)
        cb = CircuitBreaker("test", config)

        def fail():
            raise ValueError("error")

        with pytest.raises(ValueError):
            cb.call(fail)

        assert cb.state == CircuitState.OPEN

        call_count = 0
        def track_calls():
            nonlocal call_count
            call_count += 1
            return "ok"

        with pytest.raises(CircuitOpenError):
            cb.call(track_calls)

        assert call_count == 0  # Function was never called

    def test_fallback_on_open_circuit(self):
        """Fallback is called when circuit is open."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=60)
        cb = CircuitBreaker("test", config)

        def fail():
            raise ValueError("error")

        with pytest.raises(ValueError):
            cb.call(fail)

        def fallback():
            return "fallback_value"

        result = cb.call(fail, fallback=fallback)
        assert result == "fallback_value"

    def test_half_open_after_timeout(self):
        """Circuit transitions to HALF_OPEN after recovery timeout."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=0.1,
            success_threshold=1
        )
        cb = CircuitBreaker("test", config)

        def fail():
            raise ValueError("error")

        with pytest.raises(ValueError):
            cb.call(fail)

        assert cb.state == CircuitState.OPEN

        time.sleep(0.15)

        # Next call should be allowed (half-open)
        def success():
            return "ok"

        result = cb.call(success)
        assert result == "ok"
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self):
        """Failure in HALF_OPEN state reopens the circuit."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.1)
        cb = CircuitBreaker("test", config)

        def fail():
            raise ValueError("error")

        with pytest.raises(ValueError):
            cb.call(fail)

        time.sleep(0.15)

        # Circuit should be half-open, another failure reopens it
        with pytest.raises(ValueError):
            cb.call(fail)

        assert cb.state == CircuitState.OPEN

    def test_success_resets_failure_count(self):
        """Successful call resets failure count."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test", config)

        def fail():
            raise ValueError("error")

        def success():
            return "ok"

        # 2 failures
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(fail)

        assert cb.stats()["failure_count"] == 2

        # 1 success resets
        cb.call(success)
        assert cb.stats()["failure_count"] == 0

    def test_stats_tracking(self):
        """Circuit breaker tracks statistics."""
        config = CircuitBreakerConfig(failure_threshold=5)
        cb = CircuitBreaker("test-stats", config)

        def success():
            return "ok"

        def fail():
            raise ValueError("error")

        cb.call(success)
        cb.call(success)
        with pytest.raises(ValueError):
            cb.call(fail)

        stats = cb.stats()
        assert stats["name"] == "test-stats"
        assert stats["failure_count"] == 1
        assert stats["state"] == "closed"

    def test_reset(self):
        """Reset clears all state."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker("test", config)

        def fail():
            raise ValueError("error")

        with pytest.raises(ValueError):
            cb.call(fail)

        assert cb.state == CircuitState.OPEN

        cb.reset()

        assert cb.state == CircuitState.CLOSED
        assert cb.stats()["failure_count"] == 0


class TestCircuitState:
    """Tests for CircuitState enum."""

    def test_states_exist(self):
        """All circuit states exist."""
        assert CircuitState.CLOSED is not None
        assert CircuitState.OPEN is not None
        assert CircuitState.HALF_OPEN is not None

    def test_state_values(self):
        """State values are correct."""
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig."""

    def test_default_values(self):
        """Default config values."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.recovery_timeout == 30.0
        assert config.half_open_max_calls == 3
        assert config.success_threshold == 2

    def test_custom_values(self):
        """Custom config values."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            recovery_timeout=60.0,
            half_open_max_calls=5,
        )
        assert config.failure_threshold == 10
        assert config.recovery_timeout == 60.0
        assert config.half_open_max_calls == 5


class TestCircuitOpenError:
    """Tests for CircuitOpenError."""

    def test_error_message(self):
        """Error includes circuit name and retry time."""
        error = CircuitOpenError("my-service", 15.5)
        assert "my-service" in str(error)
        assert "15.5" in str(error)

    def test_error_attributes(self):
        """Error has name and time_until_retry attributes."""
        error = CircuitOpenError("test", 10.0)
        assert error.name == "test"
        assert error.time_until_retry == 10.0


class TestGlobalBreakers:
    """Tests for global circuit breaker management."""

    def test_get_circuit_breaker_creates_new(self):
        """get_circuit_breaker creates a new breaker."""
        reset_all_breakers()
        cb = get_circuit_breaker("new-service")
        assert cb.name == "new-service"
        assert cb.state == CircuitState.CLOSED

    def test_get_circuit_breaker_returns_same(self):
        """get_circuit_breaker returns same instance for same name."""
        reset_all_breakers()
        cb1 = get_circuit_breaker("shared-service")
        cb2 = get_circuit_breaker("shared-service")
        assert cb1 is cb2

    def test_reset_all_breakers(self):
        """reset_all_breakers resets all breakers."""
        reset_all_breakers()
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = get_circuit_breaker("to-reset", config)

        def fail():
            raise ValueError("error")

        with pytest.raises(ValueError):
            cb.call(fail)

        assert cb.state == CircuitState.OPEN

        reset_all_breakers()
        assert cb.state == CircuitState.CLOSED
