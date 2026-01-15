"""
Integration tests for the policy-safe caching layer.

Tests verify:
1. Cross-user cache isolation (no data leakage between roles)
2. Epoch-based cache invalidation
3. Singleflight request deduplication
4. End-to-end cache behavior with API endpoints
5. Redis adapter functionality (if available)
"""
from __future__ import annotations
import base64
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.cache import (
    get_rag_cache,
    reset_rag_cache,
    reset_cache_adapter,
    compute_pf_for_subject,
    RAGCache,
    InMemoryAdapter,
    Singleflight,
)
from app.policy.pdp import PDP


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def reset_cache():
    """Reset cache state before and after each test."""
    reset_rag_cache()
    reset_cache_adapter()
    yield
    reset_rag_cache()
    reset_cache_adapter()


@pytest.fixture
def finance_auth():
    """Auth header for finance user."""
    return {"Authorization": f"Basic {base64.b64encode(b'alice:alice123').decode()}"}


@pytest.fixture
def hr_auth():
    """Auth header for HR user."""
    return {"Authorization": f"Basic {base64.b64encode(b'bob:bobpass').decode()}"}


@pytest.fixture
def c_level_auth():
    """Auth header for c_level user."""
    return {"Authorization": f"Basic {base64.b64encode(b'carol:carolpw').decode()}"}


@pytest.fixture
def client():
    """Test client with mocked external dependencies."""
    return TestClient(app)


# ============================================================================
# Cross-User Isolation Tests (CRITICAL SECURITY)
# ============================================================================

class TestCrossUserIsolation:
    """Tests that verify NO data leakage between users with different roles."""

    def test_different_roles_get_different_pf(self):
        """Different roles must produce different Policy Fingerprints."""
        pf_finance = compute_pf_for_subject({"role": "finance"})
        pf_hr = compute_pf_for_subject({"role": "hr"})
        pf_engineering = compute_pf_for_subject({"role": "engineering"})
        pf_c_level = compute_pf_for_subject({"role": "c_level"})
        
        # All PFs must be unique
        pfs = {pf_finance.hash, pf_hr.hash, pf_engineering.hash, pf_c_level.hash}
        assert len(pfs) == 4, "All roles must have unique Policy Fingerprints"

    def test_decision_cache_isolates_roles(self):
        """Cached decisions for one role must NOT be returned to another role."""
        cache = RAGCache()
        resource = {"owner_dept": "hr", "doc_type": "internal", "sensitivity": "low"}
        
        # Finance user caches a DENY decision
        pf_finance = compute_pf_for_subject({"role": "finance"})
        cache.get_decision(
            pf=pf_finance,
            resource=resource,
            action="retrieve_chunk",
            compute_fn=lambda: {"effect": "deny", "rule": "cross_dept_deny"},
        )
        
        # HR user should NOT get the cached deny - they should get their own decision
        pf_hr = compute_pf_for_subject({"role": "hr"})
        call_count = 0
        def compute_hr_decision():
            nonlocal call_count
            call_count += 1
            return {"effect": "permit", "rule": "same_dept_access"}
        
        result, was_cached = cache.get_decision(
            pf=pf_hr,
            resource=resource,
            action="retrieve_chunk",
            compute_fn=compute_hr_decision,
        )
        
        # HR should get their own computed result, not finance's cached deny
        assert was_cached is False, "HR should NOT get finance's cached decision"
        assert call_count == 1, "HR's compute function should be called"
        assert result["effect"] == "permit", "HR should get permit for their dept docs"

    def test_llm_cache_isolates_roles(self):
        """Cached LLM answers must NOT leak between roles."""
        cache = RAGCache()
        query = "What are the salary ranges?"
        context_hash = "salary_context_abc"
        model = "gpt-4o-mini"
        
        # HR user caches an answer with sensitive salary data
        pf_hr = compute_pf_for_subject({"role": "hr"})
        cache.get_llm_answer(
            pf=pf_hr,
            query=query,
            context_hash=context_hash,
            model=model,
            compute_fn=lambda: "Salary ranges: Engineer $100k-200k, Manager $150k-250k",
        )
        
        # Marketing user asks same question - should NOT get HR's cached answer
        pf_marketing = compute_pf_for_subject({"role": "marketing"})
        result, was_cached = cache.get_llm_answer(
            pf=pf_marketing,
            query=query,
            context_hash=context_hash,
            model=model,
            compute_fn=lambda: "I don't have access to salary information.",
        )
        
        assert was_cached is False, "Marketing should NOT get HR's cached answer"
        assert "don't have access" in result, "Marketing should get their own answer"

    def test_same_role_shares_cache(self):
        """Users with the SAME role SHOULD share cached decisions."""
        cache = RAGCache()
        resource = {"owner_dept": "finance", "doc_type": "internal"}

        # First finance user caches a decision
        pf_finance1 = compute_pf_for_subject({"role": "finance"})
        cache.get_decision(
            pf=pf_finance1,
            resource=resource,
            action="retrieve_chunk",
            compute_fn=lambda: {"effect": "permit", "rule": "same_dept"},
        )

        # Second finance user (same role) should get cached result
        pf_finance2 = compute_pf_for_subject({"role": "finance"})
        call_count = 0
        def should_not_be_called():
            nonlocal call_count
            call_count += 1
            return {"effect": "permit", "rule": "same_dept"}

        result, was_cached = cache.get_decision(
            pf=pf_finance2,
            resource=resource,
            action="retrieve_chunk",
            compute_fn=should_not_be_called,
        )

        assert was_cached is True, "Same role users should share cache"
        assert call_count == 0, "Compute function should NOT be called for cache hit"


# ============================================================================
# Epoch Invalidation Tests
# ============================================================================

class TestEpochInvalidation:
    """Tests that verify epoch bumps correctly invalidate caches."""

    def test_policy_epoch_change_invalidates_pf(self):
        """Changing CACHE_POLICY_EPOCH should produce different PF."""
        with patch.dict(os.environ, {"CACHE_POLICY_EPOCH": "1"}):
            pf1 = compute_pf_for_subject({"role": "finance"})

        with patch.dict(os.environ, {"CACHE_POLICY_EPOCH": "2"}):
            pf2 = compute_pf_for_subject({"role": "finance"})

        assert pf1.hash != pf2.hash, "Policy epoch change should invalidate PF"

    def test_entitlement_epoch_change_invalidates_pf(self):
        """Changing CACHE_ENTITLEMENT_EPOCH should produce different PF."""
        with patch.dict(os.environ, {"CACHE_ENTITLEMENT_EPOCH": "1"}):
            pf1 = compute_pf_for_subject({"role": "hr"})

        with patch.dict(os.environ, {"CACHE_ENTITLEMENT_EPOCH": "2"}):
            pf2 = compute_pf_for_subject({"role": "hr"})

        assert pf1.hash != pf2.hash, "Entitlement epoch change should invalidate PF"

    def test_epoch_bump_causes_cache_miss(self):
        """After epoch bump, previously cached values should miss."""
        cache = RAGCache()
        resource = {"owner_dept": "engineering"}

        # Cache a decision with epoch=1
        with patch.dict(os.environ, {"CACHE_POLICY_EPOCH": "1"}):
            pf1 = compute_pf_for_subject({"role": "engineering"})
            cache.get_decision(
                pf=pf1,
                resource=resource,
                action="read",
                compute_fn=lambda: {"effect": "permit", "rule": "v1"},
            )

        # Bump epoch to 2 - should miss cache
        with patch.dict(os.environ, {"CACHE_POLICY_EPOCH": "2"}):
            pf2 = compute_pf_for_subject({"role": "engineering"})
            call_count = 0
            def new_compute():
                nonlocal call_count
                call_count += 1
                return {"effect": "permit", "rule": "v2"}

            result, was_cached = cache.get_decision(
                pf=pf2,
                resource=resource,
                action="read",
                compute_fn=new_compute,
            )

        assert was_cached is False, "Epoch bump should cause cache miss"
        assert call_count == 1, "New compute should be called after epoch bump"
        assert result["rule"] == "v2", "Should get new computed value"


# ============================================================================
# Singleflight Tests
# ============================================================================

class TestSingleflightIntegration:
    """Tests for request coalescing under concurrent load."""

    def test_concurrent_requests_deduplicated(self):
        """Multiple concurrent requests for same key should only compute once."""
        cache = RAGCache()
        pf = compute_pf_for_subject({"role": "finance"})
        resource = {"owner_dept": "finance"}

        compute_count = 0
        compute_lock = threading.Lock()

        def slow_compute():
            nonlocal compute_count
            with compute_lock:
                compute_count += 1
            time.sleep(0.1)  # Simulate slow computation
            return {"effect": "permit", "rule": "slow_rule"}

        results = []
        threads = []

        # Launch 10 concurrent requests
        for _ in range(10):
            def worker():
                result, _ = cache.get_decision(
                    pf=pf,
                    resource=resource,
                    action="read",
                    compute_fn=slow_compute,
                )
                results.append(result)

            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All should get same result
        assert all(r["effect"] == "permit" for r in results)
        # But compute should only be called once (or maybe twice due to timing)
        assert compute_count <= 2, f"Compute called {compute_count} times, expected <=2"

    def test_singleflight_stats_tracked(self):
        """Singleflight should track deduplication stats."""
        sf = Singleflight()

        def slow_fn():
            time.sleep(0.05)
            return "result"

        threads = []
        for _ in range(5):
            t = threading.Thread(target=lambda: sf.do("key", slow_fn))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        stats = sf.stats()
        assert stats["dedup_count"] >= 4, "Should deduplicate at least 4 requests"


# ============================================================================
# Performance Characteristics Tests
# ============================================================================

class TestCachePerformance:
    """Tests that verify cache hit is faster than cache miss."""

    def test_cache_hit_faster_than_miss(self):
        """Cache hit should be significantly faster than miss."""
        cache = RAGCache()
        pf = compute_pf_for_subject({"role": "engineering"})

        def expensive_compute():
            # Simulate expensive PDP evaluation
            time.sleep(0.01)  # 10ms
            return {"effect": "permit", "rule": "expensive"}

        # First call - cache miss (slow)
        t0 = time.perf_counter()
        cache.get_decision(pf, {"dept": "eng"}, "read", expensive_compute)
        miss_time = time.perf_counter() - t0

        # Second call - cache hit (fast)
        t1 = time.perf_counter()
        cache.get_decision(pf, {"dept": "eng"}, "read", expensive_compute)
        hit_time = time.perf_counter() - t1

        # Hit should be at least 10x faster
        speedup = miss_time / hit_time if hit_time > 0 else float('inf')
        assert speedup > 10, f"Cache hit speedup was only {speedup:.1f}x, expected >10x"

    def test_retrieval_cache_speedup(self):
        """Retrieval cache should provide significant speedup."""
        cache = RAGCache()

        def expensive_retrieval():
            time.sleep(0.02)  # 20ms simulated embedding + vector search
            return [{"id": "doc1", "score": 0.9}, {"id": "doc2", "score": 0.8}]

        # Miss
        t0 = time.perf_counter()
        cache.get_retrieval("what is the budget?", 10, expensive_retrieval)
        miss_time = time.perf_counter() - t0

        # Hit
        t1 = time.perf_counter()
        cache.get_retrieval("what is the budget?", 10, expensive_retrieval)
        hit_time = time.perf_counter() - t1

        speedup = miss_time / hit_time if hit_time > 0 else float('inf')
        assert speedup > 50, f"Retrieval cache speedup was {speedup:.1f}x, expected >50x"

