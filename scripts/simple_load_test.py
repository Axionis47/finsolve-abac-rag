#!/usr/bin/env python3
"""
Simple load test for cache performance without external dependencies.

Usage:
    python3 scripts/simple_load_test.py [--host HOST] [--users N] [--duration SECS]

Example:
    python3 scripts/simple_load_test.py --host http://localhost:8000 --users 10 --duration 30
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


@dataclass
class LoadTestResult:
    """Results from a load test run."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    latencies_ms: List[float] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0
    errors: List[str] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    
    @property
    def duration_secs(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def requests_per_second(self) -> float:
        if self.duration_secs > 0:
            return self.total_requests / self.duration_secs
        return 0.0
    
    @property
    def success_rate(self) -> float:
        if self.total_requests > 0:
            return (self.successful_requests / self.total_requests) * 100
        return 0.0
    
    def summary(self) -> Dict[str, Any]:
        latencies = self.latencies_ms or [0]
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate_pct": round(self.success_rate, 2),
            "requests_per_second": round(self.requests_per_second, 2),
            "duration_secs": round(self.duration_secs, 2),
            "latency_ms": {
                "min": round(min(latencies), 2),
                "max": round(max(latencies), 2),
                "mean": round(statistics.mean(latencies), 2),
                "median": round(statistics.median(latencies), 2),
                "p95": round(sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0], 2),
                "p99": round(sorted(latencies)[int(len(latencies) * 0.99)] if len(latencies) > 1 else latencies[0], 2),
            },
        }


SAMPLE_QUERIES = [
    "What is the company vacation policy?",
    "How do I submit an expense report?",
    "What are the security guidelines?",
    "Tell me about the benefits package",
    "What is the remote work policy?",
]

TEST_USERS = [
    {"user_id": "alice", "role": "engineer", "department": "engineering", "clearance": "internal"},
    {"user_id": "bob", "role": "manager", "department": "engineering", "clearance": "confidential"},
]


def make_request(host: str, query: str, user: Dict[str, str]) -> tuple[bool, float, str]:
    """Make a single request and return (success, latency_ms, error)."""
    url = f"{host}/query"
    payload = json.dumps({"query": query, "subject": user}).encode('utf-8')
    
    req = Request(url, data=payload, headers={"Content-Type": "application/json"})
    
    start = time.perf_counter()
    try:
        with urlopen(req, timeout=10) as response:
            latency_ms = (time.perf_counter() - start) * 1000
            return True, latency_ms, ""
    except HTTPError as e:
        latency_ms = (time.perf_counter() - start) * 1000
        if e.code == 403:  # Access denied is valid
            return True, latency_ms, ""
        return False, latency_ms, f"HTTP {e.code}"
    except URLError as e:
        latency_ms = (time.perf_counter() - start) * 1000
        return False, latency_ms, str(e.reason)
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        return False, latency_ms, str(e)


def run_user_workload(host: str, duration_secs: float, user_id: int, result: LoadTestResult) -> None:
    """Run workload for a single simulated user."""
    user = TEST_USERS[user_id % len(TEST_USERS)]
    end_time = time.time() + duration_secs
    
    while time.time() < end_time:
        query = random.choice(SAMPLE_QUERIES)
        success, latency_ms, error = make_request(host, query, user)
        
        result.total_requests += 1
        result.latencies_ms.append(latency_ms)
        
        if success:
            result.successful_requests += 1
        else:
            result.failed_requests += 1
            if error:
                result.errors.append(error)
        
        # Small delay between requests
        time.sleep(random.uniform(0.1, 0.5))


def run_load_test(host: str, num_users: int, duration_secs: float) -> LoadTestResult:
    """Run load test with multiple concurrent users."""
    result = LoadTestResult()
    result.start_time = time.time()
    
    print(f"Starting load test: {num_users} users, {duration_secs}s duration")
    print(f"Target: {host}")
    print("-" * 50)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_users) as executor:
        futures = [
            executor.submit(run_user_workload, host, duration_secs, i, result)
            for i in range(num_users)
        ]
        concurrent.futures.wait(futures)
    
    result.end_time = time.time()
    return result


def main():
    parser = argparse.ArgumentParser(description="Simple load test for RAG cache")
    parser.add_argument("--host", default="http://localhost:8000", help="Target host URL")
    parser.add_argument("--users", type=int, default=5, help="Number of concurrent users")
    parser.add_argument("--duration", type=float, default=30.0, help="Test duration in seconds")
    args = parser.parse_args()
    
    result = run_load_test(args.host, args.users, args.duration)
    summary = result.summary()
    
    print("\n" + "=" * 50)
    print("LOAD TEST RESULTS")
    print("=" * 50)
    print(json.dumps(summary, indent=2))
    
    if result.errors:
        print(f"\nUnique errors ({len(set(result.errors))}):")
        for err in set(result.errors)[:5]:
            print(f"  - {err}")


if __name__ == "__main__":
    main()

