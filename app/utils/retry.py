"""
Retry with Exponential Backoff.

Provides decorators and utilities for retrying failed operations
with exponential backoff and optional jitter.

Usage:
    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def call_external_api():
        ...

    # Or use directly
    result = retry_call(lambda: api.fetch(), max_retries=3)
"""
from __future__ import annotations
import functools
import random
import time
import logging
from typing import Callable, TypeVar, Type, Tuple, Optional, Any

T = TypeVar("T")
logger = logging.getLogger(__name__)


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
):
    """
    Decorator for retry with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Add random jitter to delay
        exceptions: Tuple of exception types to retry on
        on_retry: Optional callback called on each retry with (exception, attempt)

    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Optional[Exception] = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.warning(
                            f"Retry exhausted for {func.__name__} after {max_retries + 1} attempts"
                        )
                        raise

                    delay = min(
                        base_delay * (exponential_base ** attempt),
                        max_delay
                    )

                    if jitter:
                        delay *= (0.5 + random.random())

                    logger.info(
                        f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                        f"after {type(e).__name__}: {e}. "
                        f"Waiting {delay:.2f}s"
                    )

                    if on_retry:
                        on_retry(e, attempt + 1)

                    time.sleep(delay)

            # Should never reach here, but satisfy type checker
            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected retry loop exit")

        return wrapper
    return decorator


def retry_call(
    func: Callable[[], T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> T:
    """
    Execute a function with retry logic.

    Args:
        func: Zero-argument callable to execute
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Add random jitter to delay
        exceptions: Tuple of exception types to retry on

    Returns:
        Result of successful function call

    Raises:
        Last exception if all retries exhausted
    """
    last_exception: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except exceptions as e:
            last_exception = e

            if attempt == max_retries:
                raise

            delay = min(
                base_delay * (exponential_base ** attempt),
                max_delay
            )

            if jitter:
                delay *= (0.5 + random.random())

            time.sleep(delay)

    # Should never reach here
    if last_exception:
        raise last_exception
    raise RuntimeError("Unexpected retry loop exit")


class RetryConfig:
    """Configuration class for retry behavior."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        exceptions: Tuple[Type[Exception], ...] = (Exception,),
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.exceptions = exceptions

    def decorator(self) -> Callable:
        """Return a decorator with this configuration."""
        return retry_with_backoff(
            max_retries=self.max_retries,
            base_delay=self.base_delay,
            max_delay=self.max_delay,
            exponential_base=self.exponential_base,
            jitter=self.jitter,
            exceptions=self.exceptions,
        )


# Pre-configured retry configs for common use cases
LLM_RETRY_CONFIG = RetryConfig(
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
    exceptions=(ConnectionError, TimeoutError, Exception),
)

EMBEDDING_RETRY_CONFIG = RetryConfig(
    max_retries=2,
    base_delay=0.5,
    max_delay=10.0,
    exceptions=(ConnectionError, TimeoutError, Exception),
)
