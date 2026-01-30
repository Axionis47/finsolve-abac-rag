#!/usr/bin/env python3
"""
Locust-based load testing for ABAC RAG cache performance.

Usage:
    # Install locust first: pip install locust
    
    # Run with web UI:
    locust -f scripts/load_test.py --host=http://localhost:8000
    
    # Run headless:
    locust -f scripts/load_test.py --host=http://localhost:8000 \
           --headless -u 50 -r 10 -t 60s --csv=results/load_test

Scenarios:
    1. Warm cache: Repeated queries that should hit cache
    2. Cold cache: Unique queries that miss cache
    3. Mixed workload: Realistic mix of cache hits and misses
    4. Multi-user: Different users with different roles
"""
from __future__ import annotations

import json
import random
import string
import time
from typing import Any, Dict, List

try:
    from locust import HttpUser, task, between, events
    from locust.runners import MasterRunner
    HAS_LOCUST = True
except ImportError:
    HAS_LOCUST = False
    # Provide stub classes for import
    class HttpUser:  # type: ignore
        pass
    def task(weight: int = 1):  # type: ignore
        def decorator(func):
            return func
        return decorator
    def between(min_wait: float, max_wait: float):  # type: ignore
        return 1.0


# Test data
SAMPLE_QUERIES = [
    "What is the company vacation policy?",
    "How do I submit an expense report?",
    "What are the security guidelines?",
    "Tell me about the benefits package",
    "What is the remote work policy?",
    "How do I request time off?",
    "What are the performance review criteria?",
    "Explain the promotion process",
    "What training programs are available?",
    "How do I report a compliance issue?",
]

TEST_USERS = [
    {"user_id": "alice", "role": "engineer", "department": "engineering", "clearance": "internal"},
    {"user_id": "bob", "role": "manager", "department": "engineering", "clearance": "confidential"},
    {"user_id": "carol", "role": "hr_admin", "department": "hr", "clearance": "restricted"},
    {"user_id": "dave", "role": "analyst", "department": "finance", "clearance": "internal"},
    {"user_id": "eve", "role": "executive", "department": "executive", "clearance": "top_secret"},
]


class RAGCacheUser(HttpUser):
    """Simulates a user making RAG queries."""
    
    wait_time = between(0.5, 2.0)  # Wait 0.5-2 seconds between requests
    
    def on_start(self):
        """Initialize user with random identity."""
        self.user_data = random.choice(TEST_USERS)
        self.query_history: List[str] = []
    
    @task(10)
    def warm_cache_query(self):
        """Query that should hit cache (repeated query)."""
        # Use a fixed query for this user to maximize cache hits
        query = SAMPLE_QUERIES[hash(self.user_data["user_id"]) % len(SAMPLE_QUERIES)]
        self._make_query(query, name="warm_cache")
    
    @task(3)
    def cold_cache_query(self):
        """Query that should miss cache (unique query)."""
        # Generate a unique query
        suffix = ''.join(random.choices(string.ascii_lowercase, k=8))
        query = f"What is the policy on {suffix}?"
        self._make_query(query, name="cold_cache")
    
    @task(5)
    def mixed_query(self):
        """Realistic mix - sometimes repeat, sometimes new."""
        if self.query_history and random.random() < 0.7:
            # 70% chance to repeat a previous query
            query = random.choice(self.query_history)
        else:
            query = random.choice(SAMPLE_QUERIES)
            self.query_history.append(query)
            # Keep history bounded
            if len(self.query_history) > 20:
                self.query_history.pop(0)
        self._make_query(query, name="mixed")
    
    @task(2)
    def check_cache_stats(self):
        """Check cache statistics endpoint."""
        with self.client.get("/cache/stats", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status {response.status_code}")
    
    def _make_query(self, query: str, name: str = "query"):
        """Make a RAG query with the current user context."""
        payload = {
            "query": query,
            "subject": self.user_data,
        }
        
        with self.client.post(
            "/query",
            json=payload,
            name=f"/query [{name}]",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    # Check if response has expected fields
                    if "answer" in data or "response" in data:
                        response.success()
                    else:
                        response.failure("Missing answer in response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            elif response.status_code == 403:
                # Access denied is a valid response
                response.success()
            else:
                response.failure(f"Status {response.status_code}")


class HighVolumeUser(HttpUser):
    """Simulates high-volume automated queries."""
    
    wait_time = between(0.1, 0.5)  # Very fast queries
    
    def on_start(self):
        self.user_data = {"user_id": "bot", "role": "engineer", "department": "engineering", "clearance": "internal"}
    
    @task
    def rapid_query(self):
        """Rapid-fire queries to stress test cache."""
        query = random.choice(SAMPLE_QUERIES)
        payload = {"query": query, "subject": self.user_data}
        
        with self.client.post("/query", json=payload, name="/query [rapid]", catch_response=True) as response:
            if response.status_code in (200, 403):
                response.success()
            else:
                response.failure(f"Status {response.status_code}")

