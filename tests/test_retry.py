"""
Tests for retry utility.
"""
import pytest
import time

from app.utils.retry import retry_with_backoff, retry_call, RetryConfig


class TestRetryWithBackoff:
    """Tests for retry_with_backoff decorator."""

    def test_successful_call_no_retry(self):
        """Successful call doesn't retry."""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01, jitter=False)
        def success():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = success()
        assert result == "ok"
        assert call_count == 1

    def test_retries_on_failure(self):
        """Function is retried on failure."""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01, jitter=False)
        def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("error")
            return "ok"

        result = fail_then_succeed()
        assert result == "ok"
        assert call_count == 3

    def test_max_retries_exceeded(self):
        """Raises after max retries exceeded."""
        call_count = 0

        @retry_with_backoff(max_retries=2, base_delay=0.01, jitter=False)
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise ValueError("error")

        with pytest.raises(ValueError):
            always_fail()

        assert call_count == 3  # Initial + 2 retries

    def test_specific_exceptions_only(self):
        """Only retries specified exceptions."""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01, exceptions=(ValueError,))
        def raise_type_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("not retried")

        with pytest.raises(TypeError):
            raise_type_error()

        assert call_count == 1  # No retries for TypeError

    def test_max_delay_cap(self):
        """Delay is capped at max_delay."""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=0.05, jitter=False)
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise ValueError("error")

        start = time.time()
        with pytest.raises(ValueError):
            always_fail()
        elapsed = time.time() - start

        # 3 retries at max 0.05s each = max ~0.15s
        assert elapsed < 0.5  # Would be much longer without cap

    def test_returns_value_on_success(self):
        """Returns the function's return value on success."""
        @retry_with_backoff(max_retries=1, base_delay=0.01)
        def return_dict():
            return {"key": "value", "count": 42}

        result = return_dict()
        assert result == {"key": "value", "count": 42}

    def test_preserves_function_metadata(self):
        """Decorator preserves function name and docstring."""
        @retry_with_backoff(max_retries=1)
        def documented_function():
            """This is my docstring."""
            pass

        assert documented_function.__name__ == "documented_function"
        assert "docstring" in documented_function.__doc__

    def test_on_retry_callback(self):
        """on_retry callback is called on each retry."""
        retry_calls = []

        def track_retry(exc, attempt):
            retry_calls.append((type(exc).__name__, attempt))

        call_count = 0

        @retry_with_backoff(max_retries=2, base_delay=0.01, jitter=False, on_retry=track_retry)
        def fail_twice():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ValueError("error")
            return "ok"

        result = fail_twice()
        assert result == "ok"
        assert len(retry_calls) == 2
        assert retry_calls[0] == ("ValueError", 1)
        assert retry_calls[1] == ("ValueError", 2)


class TestRetryCall:
    """Tests for retry_call function."""

    def test_successful_call(self):
        """Successful call returns result."""
        result = retry_call(lambda: "success", max_retries=3)
        assert result == "success"

    def test_retries_on_failure(self):
        """Retries on failure."""
        call_count = 0

        def fail_once():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("first call fails")
            return "ok"

        result = retry_call(fail_once, max_retries=3, base_delay=0.01, jitter=False)
        assert result == "ok"
        assert call_count == 2

    def test_raises_after_max_retries(self):
        """Raises exception after max retries."""
        def always_fail():
            raise ValueError("always fails")

        with pytest.raises(ValueError, match="always fails"):
            retry_call(always_fail, max_retries=2, base_delay=0.01, jitter=False)


class TestRetryConfig:
    """Tests for RetryConfig class."""

    def test_default_config(self):
        """Default configuration values."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True

    def test_custom_config(self):
        """Custom configuration values."""
        config = RetryConfig(
            max_retries=5,
            base_delay=0.5,
            max_delay=30.0,
            exponential_base=3.0,
            jitter=False,
            exceptions=(ValueError, TypeError),
        )
        assert config.max_retries == 5
        assert config.base_delay == 0.5
        assert config.exceptions == (ValueError, TypeError)

    def test_decorator_from_config(self):
        """Config produces working decorator."""
        config = RetryConfig(max_retries=2, base_delay=0.01, jitter=False)

        call_count = 0

        @config.decorator()
        def fail_once():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("first")
            return "success"

        result = fail_once()
        assert result == "success"
        assert call_count == 2
