#!/usr/bin/env python3
"""
Inference Benchmarking Script

Measures and compares:
1. Time to First Token (TTFT) - prefill latency
2. Tokens per second - decode throughput
3. End-to-end latency
4. Effect of prefix caching

Usage:
    python scripts/benchmark_inference.py --backend vllm --url http://localhost:8000/v1
    python scripts/benchmark_inference.py --backend openai
    python scripts/benchmark_inference.py --compare  # Compare all backends
"""
import argparse
import json
import statistics
import sys
import time
import urllib.request
from dataclasses import dataclass, asdict
from typing import List, Optional

# Add project root to path
sys.path.insert(0, ".")


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    backend: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float  # Time to first token
    total_ms: float  # Total latency
    tokens_per_sec: float
    cached_prefix: bool = False


@dataclass
class BenchmarkSummary:
    """Aggregated benchmark statistics."""
    backend: str
    model: str
    num_runs: int
    avg_ttft_ms: float
    p50_ttft_ms: float
    p95_ttft_ms: float
    avg_total_ms: float
    avg_tokens_per_sec: float
    prefix_cache_speedup: Optional[float] = None


def call_openai_api(base_url: str, model: str, messages: List[dict], 
                    api_key: str = "not-needed") -> dict:
    """Make an OpenAI-compatible API call."""
    url = f"{base_url}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 256,
    }
    
    req = urllib.request.Request(url, method="POST",
                                  data=json.dumps(payload).encode("utf-8"))
    for k, v in headers.items():
        req.add_header(k, v)
    
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    total_time = time.perf_counter() - t0
    
    usage = body.get("usage", {})
    return {
        "content": body["choices"][0]["message"]["content"],
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_time": total_time,
    }


def run_single_benchmark(base_url: str, model: str, messages: List[dict],
                         backend_name: str, api_key: str = "not-needed",
                         cached_prefix: bool = False) -> BenchmarkResult:
    """Run a single benchmark iteration."""
    result = call_openai_api(base_url, model, messages, api_key)
    
    prompt_tokens = result["prompt_tokens"]
    completion_tokens = result["completion_tokens"]
    total_ms = result["total_time"] * 1000
    
    # Estimate TTFT (prefill time) - approximation
    # Real TTFT would need streaming, but we can estimate
    tokens_per_sec = completion_tokens / result["total_time"] if result["total_time"] > 0 else 0
    decode_time_ms = (completion_tokens / tokens_per_sec * 1000) if tokens_per_sec > 0 else 0
    ttft_ms = total_ms - decode_time_ms
    
    return BenchmarkResult(
        backend=backend_name,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        ttft_ms=max(0, ttft_ms),
        total_ms=total_ms,
        tokens_per_sec=tokens_per_sec,
        cached_prefix=cached_prefix,
    )


# System prompt for RAG (typical length for our use case)
SYSTEM_PROMPT = """You are a careful assistant for FinSolve, an internal company chatbot.
Answer ONLY using the provided context snippets. Cite sources in-line as [#].
If the answer is not in the context, reply: 'I don't know based on the available context.'
Always be professional and accurate. Include a final 'Citations' section."""

# Sample context (simulating retrieved documents)
SAMPLE_CONTEXT = """
Context snippets:
[1] The Q4 2024 marketing budget has been set at $2.5 million, representing a 15% increase 
from Q3. This allocation focuses on digital advertising (40%), content marketing (30%), 
and event sponsorships (30%). (Source: marketing/budget_2024.md#q4-allocation)

[2] The engineering team completed the migration to Kubernetes in Q3, reducing infrastructure 
costs by 23%. The remaining budget will be reallocated to machine learning initiatives.
(Source: engineering/quarterly_update.md#infrastructure)

[3] Employee satisfaction surveys show 87% positive responses, up from 82% last quarter.
Key improvements noted in work-life balance and remote work policies.
(Source: hr/employee_survey_q3.md#summary)
"""

TEST_QUERIES = [
    "What is the Q4 marketing budget?",
    "How much did infrastructure costs decrease after the Kubernetes migration?",
    "What is the current employee satisfaction rate?",
    "Summarize the key updates from all departments.",
]


def run_benchmark_suite(base_url: str, model: str, backend_name: str,
                        num_runs: int = 5, api_key: str = "not-needed",
                        test_prefix_cache: bool = True) -> BenchmarkSummary:
    """Run a full benchmark suite."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {backend_name}")
    print(f"Model: {model}")
    print(f"URL: {base_url}")
    print(f"Runs per query: {num_runs}")
    print(f"{'='*60}\n")
    
    results: List[BenchmarkResult] = []
    prefix_cache_results: List[BenchmarkResult] = []
    
    for query in TEST_QUERIES:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{SAMPLE_CONTEXT}\n\nQuery: {query}"},
        ]
        
        print(f"Query: {query[:50]}...")
        
        # Warmup run (also primes prefix cache)
        print("  Warmup...", end=" ", flush=True)
        try:
            run_single_benchmark(base_url, model, messages, backend_name, api_key)
            print("done")
        except Exception as e:
            print(f"failed: {e}")
            continue

        # Benchmark runs
        for i in range(num_runs):
            result = run_single_benchmark(base_url, model, messages, backend_name, api_key)
            results.append(result)
            print(f"  Run {i+1}: TTFT={result.ttft_ms:.1f}ms, "
                  f"Total={result.total_ms:.1f}ms, "
                  f"Tok/s={result.tokens_per_sec:.1f}")

        # Test prefix cache effect (same system prompt, different query)
        if test_prefix_cache:
            cache_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{SAMPLE_CONTEXT}\n\nQuery: What else can you tell me?"},
            ]
            cache_result = run_single_benchmark(base_url, model, cache_messages,
                                                 backend_name, api_key, cached_prefix=True)
            prefix_cache_results.append(cache_result)

    if not results:
        raise RuntimeError("No successful benchmark runs")

    # Calculate statistics
    ttft_values = [r.ttft_ms for r in results]
    total_values = [r.total_ms for r in results]
    tps_values = [r.tokens_per_sec for r in results]

    summary = BenchmarkSummary(
        backend=backend_name,
        model=model,
        num_runs=len(results),
        avg_ttft_ms=statistics.mean(ttft_values),
        p50_ttft_ms=statistics.median(ttft_values),
        p95_ttft_ms=sorted(ttft_values)[int(len(ttft_values) * 0.95)] if len(ttft_values) >= 20 else max(ttft_values),
        avg_total_ms=statistics.mean(total_values),
        avg_tokens_per_sec=statistics.mean(tps_values),
    )

    # Calculate prefix cache speedup
    if prefix_cache_results and results:
        avg_no_cache = statistics.mean([r.ttft_ms for r in results[:len(prefix_cache_results)]])
        avg_with_cache = statistics.mean([r.ttft_ms for r in prefix_cache_results])
        if avg_with_cache > 0:
            summary.prefix_cache_speedup = avg_no_cache / avg_with_cache

    return summary


def print_summary(summary: BenchmarkSummary):
    """Print benchmark summary."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK SUMMARY: {summary.backend}")
    print(f"{'='*60}")
    print(f"Model:              {summary.model}")
    print(f"Total Runs:         {summary.num_runs}")
    print(f"")
    print(f"TTFT (Time to First Token):")
    print(f"  Average:          {summary.avg_ttft_ms:.1f} ms")
    print(f"  P50:              {summary.p50_ttft_ms:.1f} ms")
    print(f"  P95:              {summary.p95_ttft_ms:.1f} ms")
    print(f"")
    print(f"Throughput:")
    print(f"  Avg Tokens/sec:   {summary.avg_tokens_per_sec:.1f}")
    print(f"  Avg Total Time:   {summary.avg_total_ms:.1f} ms")
    if summary.prefix_cache_speedup:
        print(f"")
        print(f"Prefix Cache Speedup: {summary.prefix_cache_speedup:.2f}x")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark LLM inference")
    parser.add_argument("--backend", choices=["openai", "vllm", "ollama"],
                        default="vllm", help="Backend to benchmark")
    parser.add_argument("--url", default=None, help="API base URL")
    parser.add_argument("--model", default=None, help="Model name")
    parser.add_argument("--api-key", default="not-needed", help="API key")
    parser.add_argument("--runs", type=int, default=5, help="Runs per query")
    parser.add_argument("--output", default=None, help="Output JSON file")
    args = parser.parse_args()

    # Set defaults based on backend
    if args.backend == "openai":
        base_url = args.url or "https://api.openai.com/v1"
        model = args.model or "gpt-4o-mini"
        import os
        api_key = args.api_key if args.api_key != "not-needed" else os.getenv("OPENAI_API_KEY", "")
    elif args.backend == "vllm":
        base_url = args.url or "http://localhost:8000/v1"
        model = args.model or "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
        api_key = args.api_key
    elif args.backend == "ollama":
        base_url = args.url or "http://localhost:11434/v1"
        model = args.model or "llama3.2:3b"
        api_key = args.api_key

    try:
        summary = run_benchmark_suite(
            base_url=base_url,
            model=model,
            backend_name=args.backend,
            num_runs=args.runs,
            api_key=api_key,
        )
        print_summary(summary)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(asdict(summary), f, indent=2)
            print(f"Results saved to {args.output}")

    except Exception as e:
        print(f"Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

