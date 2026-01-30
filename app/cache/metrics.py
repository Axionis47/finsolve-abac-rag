"""
Prometheus metrics for cache monitoring.

Provides:
- Cache hit/miss counters by cache type
- Latency histograms for cache operations
- Memory usage gauges
- Eviction counters

If prometheus_client is not installed, provides no-op implementations.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, Generator, Optional

# Try to import prometheus_client
try:
    from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry, REGISTRY
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False


# Metrics singleton
_metrics: Optional["CacheMetrics"] = None


class CacheMetrics:
    """Prometheus metrics for cache operations."""
    
    def __init__(self, registry: Optional[Any] = None):
        self._registry = registry or (REGISTRY if HAS_PROMETHEUS else None)
        self._enabled = HAS_PROMETHEUS
        
        if self._enabled:
            self._init_metrics()
        else:
            self._init_noop()
    
    def _init_metrics(self) -> None:
        """Initialize Prometheus metrics."""
        # Counters
        self.cache_hits = Counter(
            'rag_cache_hits_total',
            'Total cache hits',
            ['cache_type', 'pf_prefix'],
            registry=self._registry
        )
        self.cache_misses = Counter(
            'rag_cache_misses_total', 
            'Total cache misses',
            ['cache_type', 'pf_prefix'],
            registry=self._registry
        )
        self.cache_evictions = Counter(
            'rag_cache_evictions_total',
            'Total cache evictions',
            ['cache_type', 'reason'],
            registry=self._registry
        )
        self.singleflight_dedup = Counter(
            'rag_singleflight_dedup_total',
            'Total deduplicated requests via singleflight',
            ['cache_type'],
            registry=self._registry
        )
        
        # Histograms
        self.cache_operation_latency = Histogram(
            'rag_cache_operation_seconds',
            'Cache operation latency in seconds',
            ['cache_type', 'operation'],
            buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
            registry=self._registry
        )
        self.compute_latency = Histogram(
            'rag_cache_compute_seconds',
            'Time spent computing values on cache miss',
            ['cache_type'],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self._registry
        )
        
        # Gauges
        self.cache_size = Gauge(
            'rag_cache_entries',
            'Current number of cache entries',
            ['cache_type'],
            registry=self._registry
        )
        self.cache_memory_bytes = Gauge(
            'rag_cache_memory_bytes',
            'Current cache memory usage in bytes',
            ['cache_type'],
            registry=self._registry
        )
        self.cache_hit_rate = Gauge(
            'rag_cache_hit_rate',
            'Current cache hit rate (0-1)',
            ['cache_type'],
            registry=self._registry
        )
    
    def _init_noop(self) -> None:
        """Initialize no-op metrics when prometheus_client is not available."""
        self.cache_hits = _NoOpCounter()
        self.cache_misses = _NoOpCounter()
        self.cache_evictions = _NoOpCounter()
        self.singleflight_dedup = _NoOpCounter()
        self.cache_operation_latency = _NoOpHistogram()
        self.compute_latency = _NoOpHistogram()
        self.cache_size = _NoOpGauge()
        self.cache_memory_bytes = _NoOpGauge()
        self.cache_hit_rate = _NoOpGauge()
    
    def record_hit(self, cache_type: str, pf_prefix: str = "") -> None:
        """Record a cache hit."""
        self.cache_hits.labels(cache_type=cache_type, pf_prefix=pf_prefix[:8]).inc()
    
    def record_miss(self, cache_type: str, pf_prefix: str = "") -> None:
        """Record a cache miss."""
        self.cache_misses.labels(cache_type=cache_type, pf_prefix=pf_prefix[:8]).inc()
    
    def record_eviction(self, cache_type: str, reason: str = "lru") -> None:
        """Record a cache eviction."""
        self.cache_evictions.labels(cache_type=cache_type, reason=reason).inc()
    
    def record_singleflight_dedup(self, cache_type: str) -> None:
        """Record a deduplicated request."""
        self.singleflight_dedup.labels(cache_type=cache_type).inc()
    
    def observe_operation_latency(self, cache_type: str, operation: str, seconds: float) -> None:
        """Record cache operation latency."""
        self.cache_operation_latency.labels(cache_type=cache_type, operation=operation).observe(seconds)
    
    def observe_compute_latency(self, cache_type: str, seconds: float) -> None:
        """Record compute latency on cache miss."""
        self.compute_latency.labels(cache_type=cache_type).observe(seconds)
    
    def set_cache_size(self, cache_type: str, size: int) -> None:
        """Set current cache size."""
        self.cache_size.labels(cache_type=cache_type).set(size)
    
    def set_cache_memory(self, cache_type: str, bytes_used: int) -> None:
        """Set current cache memory usage."""
        self.cache_memory_bytes.labels(cache_type=cache_type).set(bytes_used)
    
    def set_hit_rate(self, cache_type: str, rate: float) -> None:
        """Set current hit rate (0-1)."""
        self.cache_hit_rate.labels(cache_type=cache_type).set(rate)
    
    @contextmanager
    def time_operation(self, cache_type: str, operation: str) -> Generator[None, None, None]:
        """Context manager to time a cache operation."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.observe_operation_latency(cache_type, operation, elapsed)


# No-op implementations for when prometheus_client is not installed
class _NoOpLabeled:
    def labels(self, **kwargs: Any) -> "_NoOpLabeled":
        return self


class _NoOpCounter(_NoOpLabeled):
    def inc(self, amount: float = 1) -> None:
        pass


class _NoOpHistogram(_NoOpLabeled):
    def observe(self, amount: float) -> None:
        pass


class _NoOpGauge(_NoOpLabeled):
    def set(self, value: float) -> None:
        pass
    
    def inc(self, amount: float = 1) -> None:
        pass
    
    def dec(self, amount: float = 1) -> None:
        pass


def get_metrics() -> CacheMetrics:
    """Get the global cache metrics instance."""
    global _metrics
    if _metrics is None:
        _metrics = CacheMetrics()
    return _metrics


def reset_metrics() -> None:
    """Reset global metrics. Useful for testing."""
    global _metrics
    _metrics = None

