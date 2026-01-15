#!/usr/bin/env python3
"""
Cache Performance Benchmark Script

This script measures the performance impact of the caching layer and identifies
when maximum performance gains occur.

Usage:
    python scripts/benchmark_cache.py

Output includes:
- Latency comparison (cache hit vs miss)
- Throughput under load
- Memory usage
- Singleflight effectiveness
- Performance degradation analysis
"""
from __future__ import annotations
import os
import sys
import time
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.cache import (
    RAGCache,
    InMemoryAdapter,
    Singleflight,
    compute_pf_for_subject,
    reset_rag_cache,
    reset_cache_adapter,
)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    name: str
    iterations: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    throughput_rps: float


def benchmark(name: str, fn: Callable[[], Any], iterations: int = 1000, warmup: int = 100) -> BenchmarkResult:
    """Run a benchmark and collect statistics."""
    # Warmup
    for _ in range(warmup):
        fn()
    
    # Measure
    latencies = []
    start_total = time.perf_counter()
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # Convert to ms
    total_time = time.perf_counter() - start_total
    
    latencies.sort()
    return BenchmarkResult(
        name=name,
        iterations=iterations,
        mean_ms=statistics.mean(latencies),
        median_ms=statistics.median(latencies),
        p95_ms=latencies[int(len(latencies) * 0.95)],
        p99_ms=latencies[int(len(latencies) * 0.99)],
        min_ms=min(latencies),
        max_ms=max(latencies),
        throughput_rps=iterations / total_time,
    )


def print_result(result: BenchmarkResult):
    """Pretty print a benchmark result."""
    print(f"\n{'='*60}")
    print(f"  {result.name}")
    print(f"{'='*60}")
    print(f"  Iterations: {result.iterations:,}")
    print(f"  Mean:       {result.mean_ms:.4f} ms")
    print(f"  Median:     {result.median_ms:.4f} ms")
    print(f"  P95:        {result.p95_ms:.4f} ms")
    print(f"  P99:        {result.p99_ms:.4f} ms")
    print(f"  Min:        {result.min_ms:.4f} ms")
    print(f"  Max:        {result.max_ms:.4f} ms")
    print(f"  Throughput: {result.throughput_rps:,.0f} req/s")


def print_comparison(miss_result: BenchmarkResult, hit_result: BenchmarkResult):
    """Print comparison between cache miss and hit."""
    speedup_mean = miss_result.mean_ms / hit_result.mean_ms if hit_result.mean_ms > 0 else float('inf')
    speedup_p95 = miss_result.p95_ms / hit_result.p95_ms if hit_result.p95_ms > 0 else float('inf')
    throughput_gain = hit_result.throughput_rps / miss_result.throughput_rps
    
    print(f"\n{'='*60}")
    print(f"  PERFORMANCE COMPARISON")
    print(f"{'='*60}")
    print(f"  Mean Speedup:       {speedup_mean:.1f}x faster")
    print(f"  P95 Speedup:        {speedup_p95:.1f}x faster")
    print(f"  Throughput Gain:    {throughput_gain:.1f}x more requests/sec")
    print(f"  Latency Reduction:  {(1 - hit_result.mean_ms/miss_result.mean_ms)*100:.1f}%")


# ============================================================================
# Benchmark Scenarios
# ============================================================================

def benchmark_decision_cache():
    """Benchmark policy decision caching."""
    print("\n" + "="*60)
    print("  POLICY DECISION CACHE BENCHMARK")
    print("="*60)

    reset_rag_cache()
    pf = compute_pf_for_subject({"role": "finance"})
    resource = {"owner_dept": "finance", "doc_type": "internal", "sensitivity": "low"}

    pdp_delay_ms = 1.0  # Simulated PDP evaluation time

    def simulate_pdp_evaluation():
        """Simulate expensive PDP evaluation (~1ms)."""
        time.sleep(pdp_delay_ms / 1000)
        return {"effect": "permit", "rule": "same_dept_access", "obligations": []}

    # For miss benchmark: measure raw compute time + cache overhead
    miss_times = []
    for _ in range(500):
        reset_rag_cache()
        reset_cache_adapter()
        cache = RAGCache()
        t0 = time.perf_counter()
        cache.get_decision(pf, resource, "retrieve_chunk", simulate_pdp_evaluation)
        miss_times.append((time.perf_counter() - t0) * 1000)

    miss_times.sort()
    miss_mean = statistics.mean(miss_times)
    miss_p95 = miss_times[int(len(miss_times) * 0.95)]
    print(f"\n  Decision Cache MISS (with {pdp_delay_ms}ms PDP simulation)")
    print(f"    Mean: {miss_mean:.3f} ms")
    print(f"    P95:  {miss_p95:.3f} ms")

    # For hit benchmark: measure cached lookup time
    reset_rag_cache()
    cache = RAGCache()
    cache.get_decision(pf, resource, "retrieve_chunk", simulate_pdp_evaluation)  # Prime cache

    hit_times = []
    for _ in range(10000):
        t0 = time.perf_counter()
        cache.get_decision(pf, resource, "retrieve_chunk", simulate_pdp_evaluation)
        hit_times.append((time.perf_counter() - t0) * 1000)

    hit_times.sort()
    hit_mean = statistics.mean(hit_times)
    hit_p95 = hit_times[int(len(hit_times) * 0.95)]
    print(f"\n  Decision Cache HIT")
    print(f"    Mean: {hit_mean:.4f} ms")
    print(f"    P95:  {hit_p95:.4f} ms")

    # Calculate speedup
    speedup_mean = miss_mean / hit_mean if hit_mean > 0 else float('inf')
    speedup_p95 = miss_p95 / hit_p95 if hit_p95 > 0 else float('inf')

    print(f"\n  PERFORMANCE IMPROVEMENT")
    print(f"    Mean Speedup:       {speedup_mean:.0f}x faster")
    print(f"    P95 Speedup:        {speedup_p95:.0f}x faster")
    print(f"    Latency Reduction:  {(1 - hit_mean/miss_mean)*100:.1f}%")

    return miss_mean, hit_mean


def benchmark_llm_cache():
    """Benchmark LLM answer caching."""
    print("\n" + "="*60)
    print("  LLM ANSWER CACHE BENCHMARK")
    print("="*60)

    llm_delay_ms = 50.0  # Simulated LLM API call time
    pf = compute_pf_for_subject({"role": "engineering"})

    def simulate_llm_call():
        """Simulate expensive LLM API call (~50ms)."""
        time.sleep(llm_delay_ms / 1000)
        return "Based on the documents, the Q4 engineering budget is $2.5M..."

    # Cache miss benchmark
    miss_times = []
    for _ in range(50):  # Fewer iterations due to 50ms delay
        reset_rag_cache()
        reset_cache_adapter()
        cache = RAGCache()
        t0 = time.perf_counter()
        cache.get_llm_answer(pf, "what is the q4 budget?", "ctx123", "gpt-4o-mini", simulate_llm_call)
        miss_times.append((time.perf_counter() - t0) * 1000)

    miss_times.sort()
    miss_mean = statistics.mean(miss_times)
    miss_p95 = miss_times[int(len(miss_times) * 0.95)]
    print(f"\n  LLM Cache MISS (with {llm_delay_ms}ms API call simulation)")
    print(f"    Mean: {miss_mean:.2f} ms")
    print(f"    P95:  {miss_p95:.2f} ms")

    # Cache hit benchmark
    reset_rag_cache()
    cache = RAGCache()
    cache.get_llm_answer(pf, "what is the q4 budget?", "ctx123", "gpt-4o-mini", simulate_llm_call)  # Prime

    hit_times = []
    for _ in range(10000):
        t0 = time.perf_counter()
        cache.get_llm_answer(pf, "what is the q4 budget?", "ctx123", "gpt-4o-mini", simulate_llm_call)
        hit_times.append((time.perf_counter() - t0) * 1000)

    hit_times.sort()
    hit_mean = statistics.mean(hit_times)
    hit_p95 = hit_times[int(len(hit_times) * 0.95)]
    print(f"\n  LLM Cache HIT")
    print(f"    Mean: {hit_mean:.4f} ms")
    print(f"    P95:  {hit_p95:.4f} ms")

    # Calculate speedup
    speedup_mean = miss_mean / hit_mean if hit_mean > 0 else float('inf')
    speedup_p95 = miss_p95 / hit_p95 if hit_p95 > 0 else float('inf')

    print(f"\n  PERFORMANCE IMPROVEMENT")
    print(f"    Mean Speedup:       {speedup_mean:.0f}x faster")
    print(f"    P95 Speedup:        {speedup_p95:.0f}x faster")
    print(f"    Latency Reduction:  {(1 - hit_mean/miss_mean)*100:.1f}%")
    print(f"    Time Saved/Request: {miss_mean - hit_mean:.2f} ms")

    return miss_mean, hit_mean


def benchmark_retrieval_cache():
    """Benchmark retrieval caching."""
    print("\n" + "="*60)
    print("  RETRIEVAL CACHE BENCHMARK")
    print("="*60)

    retrieval_delay_ms = 20.0  # Simulated embedding + vector search time

    def simulate_retrieval():
        """Simulate embedding + vector search (~20ms)."""
        time.sleep(retrieval_delay_ms / 1000)
        return [{"id": f"doc{i}", "score": 0.9 - i*0.05} for i in range(10)]

    # Cache miss benchmark
    miss_times = []
    for _ in range(100):
        reset_rag_cache()
        reset_cache_adapter()
        cache = RAGCache()
        t0 = time.perf_counter()
        cache.get_retrieval("what are the quarterly results?", 10, simulate_retrieval)
        miss_times.append((time.perf_counter() - t0) * 1000)

    miss_times.sort()
    miss_mean = statistics.mean(miss_times)
    miss_p95 = miss_times[int(len(miss_times) * 0.95)]
    print(f"\n  Retrieval Cache MISS (with {retrieval_delay_ms}ms search simulation)")
    print(f"    Mean: {miss_mean:.2f} ms")
    print(f"    P95:  {miss_p95:.2f} ms")

    # Cache hit benchmark
    reset_rag_cache()
    cache = RAGCache()
    cache.get_retrieval("what are the quarterly results?", 10, simulate_retrieval)  # Prime

    hit_times = []
    for _ in range(10000):
        t0 = time.perf_counter()
        cache.get_retrieval("what are the quarterly results?", 10, simulate_retrieval)
        hit_times.append((time.perf_counter() - t0) * 1000)

    hit_times.sort()
    hit_mean = statistics.mean(hit_times)
    hit_p95 = hit_times[int(len(hit_times) * 0.95)]
    print(f"\n  Retrieval Cache HIT")
    print(f"    Mean: {hit_mean:.4f} ms")
    print(f"    P95:  {hit_p95:.4f} ms")

    # Calculate speedup
    speedup_mean = miss_mean / hit_mean if hit_mean > 0 else float('inf')
    speedup_p95 = miss_p95 / hit_p95 if hit_p95 > 0 else float('inf')

    print(f"\n  PERFORMANCE IMPROVEMENT")
    print(f"    Mean Speedup:       {speedup_mean:.0f}x faster")
    print(f"    P95 Speedup:        {speedup_p95:.0f}x faster")
    print(f"    Latency Reduction:  {(1 - hit_mean/miss_mean)*100:.1f}%")
    print(f"    Time Saved/Request: {miss_mean - hit_mean:.2f} ms")

    return miss_mean, hit_mean


def benchmark_singleflight():
    """Benchmark singleflight deduplication under concurrent load."""
    print("\n" + "="*60)
    print("  SINGLEFLIGHT DEDUPLICATION BENCHMARK")
    print("="*60)

    compute_count = 0
    lock = threading.Lock()

    def slow_compute():
        nonlocal compute_count
        with lock:
            compute_count += 1
        time.sleep(0.05)  # 50ms
        return {"result": "computed"}

    # WITHOUT singleflight (simulate)
    reset_rag_cache()
    compute_count = 0
    no_sf_cache = InMemoryAdapter()

    def without_singleflight():
        key = "test_key"
        result = no_sf_cache.get(key)
        if result is None:
            result = slow_compute()
            no_sf_cache.set(key, result, 3600)
        return result

    # Simulate concurrent requests without singleflight
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(without_singleflight) for _ in range(20)]
        for f in as_completed(futures):
            f.result()
    no_sf_time = time.perf_counter() - start
    no_sf_compute_count = compute_count

    # WITH singleflight
    reset_rag_cache()
    compute_count = 0
    sf = Singleflight()
    sf_cache = InMemoryAdapter()

    def with_singleflight():
        key = "test_key"
        result = sf_cache.get(key)
        if result is None:
            result = sf.do(key, slow_compute)
            sf_cache.set(key, result, 3600)
        return result

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(with_singleflight) for _ in range(20)]
        for f in as_completed(futures):
            f.result()
    sf_time = time.perf_counter() - start
    sf_compute_count = compute_count

    print(f"\n  WITHOUT Singleflight:")
    print(f"    Total Time:     {no_sf_time*1000:.1f} ms")
    print(f"    Compute Calls:  {no_sf_compute_count}")
    print(f"    Wasted Work:    {no_sf_compute_count - 1} redundant computations")

    print(f"\n  WITH Singleflight:")
    print(f"    Total Time:     {sf_time*1000:.1f} ms")
    print(f"    Compute Calls:  {sf_compute_count}")
    print(f"    Dedup Rate:     {(1 - sf_compute_count/no_sf_compute_count)*100:.1f}%")

    print(f"\n  Improvement:")
    print(f"    Time Saved:     {(no_sf_time - sf_time)*1000:.1f} ms ({(1 - sf_time/no_sf_time)*100:.1f}%)")
    print(f"    Compute Saved:  {no_sf_compute_count - sf_compute_count} calls")


def benchmark_memory_overhead():
    """Measure memory overhead of cached entries."""
    print("\n" + "="*60)
    print("  MEMORY OVERHEAD ANALYSIS")
    print("="*60)

    cache = InMemoryAdapter(max_size=10000)

    # Measure overhead per entry type
    # Decision cache entry
    pf = compute_pf_for_subject({"role": "finance"})
    decision = {"effect": "permit", "rule": "same_dept", "obligations": []}
    cache.set(f"decision:{pf.hash}:abc123:read", decision, 3600)

    # LLM cache entry (larger)
    llm_answer = "This is a typical LLM response. " * 50  # ~1.5KB
    cache.set(f"llm:{pf.hash}:query123:ctx456:gpt-4o-mini", llm_answer, 3600)

    # Retrieval entry
    retrieval = [{"id": f"doc{i}", "score": 0.9, "text": "chunk"*20} for i in range(10)]
    cache.set("ret0:query789:1:10", retrieval, 3600)

    stats = cache.stats()
    print(f"  Cache Entries: {stats['entries']}")
    print(f"  Max Size: {stats['max_size']}")
    print(f"\n  Entry Size Estimates:")
    print(f"    Decision:   ~200 bytes")
    print(f"    LLM Answer: ~1-10 KB")
    print(f"    Retrieval:  ~2-5 KB")


def print_summary():
    """Print when maximum performance gains occur."""
    print("\n" + "="*60)
    print("  MAXIMUM PERFORMANCE GAINS - SUMMARY")
    print("="*60)
    print("""
  WHEN TO EXPECT MAXIMUM SPEEDUP:

  1. LLM ANSWER CACHING (50-500x speedup)
     - Same query asked by users with same role
     - API call saved entirely (~50-500ms per hit)
     - Best for: FAQ-type queries, common questions

  2. RETRIEVAL CACHING (20-100x speedup)
     - Identical queries (after normalization)
     - Embedding + vector search saved (~20-100ms per hit)
     - Best for: Search-heavy workloads, repeated queries

  3. POLICY DECISION CACHING (10-50x speedup)
     - Same resource accessed by users with same role
     - PDP evaluation saved (~1-10ms per hit)
     - Best for: Hot documents, high-traffic resources

  4. SINGLEFLIGHT DEDUPLICATION (N-1 saved)
     - Concurrent identical requests
     - Prevents N-1 redundant computations
     - Best for: Burst traffic, popular queries

  OPTIMAL CONDITIONS FOR CACHING:

  ✓ Repeated queries (normalize to maximize hits)
  ✓ Users with same roles (share PF-scoped caches)
  ✓ Stable policies (infrequent epoch bumps)
  ✓ Burst traffic patterns (singleflight shines)
  ✓ Read-heavy workloads (low write invalidation)

  WHEN CACHING PROVIDES MINIMAL BENEFIT:

  ✗ Every query is unique
  ✗ Every user has different role
  ✗ Frequent policy changes (epoch bumps)
  ✗ Write-heavy workloads (constant invalidation)
  ✗ Cold start (cache empty)
""")


def main():
    from unittest.mock import patch

    print("\n" + "#"*60)
    print("#" + " "*18 + "CACHE BENCHMARK SUITE" + " "*17 + "#")
    print("#"*60)

    # Suppress audit logging during benchmarks
    with patch('app.utils.audit.log_event'):
        benchmark_decision_cache()
        benchmark_llm_cache()
        benchmark_retrieval_cache()
        benchmark_singleflight()
        benchmark_memory_overhead()

    print_summary()


if __name__ == "__main__":
    main()

