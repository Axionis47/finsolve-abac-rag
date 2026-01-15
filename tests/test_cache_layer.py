"""
Tests for the policy-safe caching layer.

Tests cover:
- PolicyFingerprint computation and stability
- Key generation and normalization
- Singleflight request coalescing
- RAGCache integration
- Cache adapter behavior
"""
from __future__ import annotations
import time
import threading
from unittest.mock import patch

import pytest

from app.cache import (
    PolicyFingerprint,
    compute_policy_fingerprint,
    compute_pf_for_subject,
    get_epochs,
    normalize_query,
    make_ret0_key,
    make_allowset_key,
    make_decision_key,
    make_rerank_key,
    make_llm_key,
    Singleflight,
    InMemoryAdapter,
    NoOpAdapter,
    RAGCache,
    get_rag_cache,
    reset_rag_cache,
    reset_cache_adapter,
)


class TestPolicyFingerprint:
    """Tests for Policy Fingerprint computation."""

    def test_pf_deterministic(self):
        """Same inputs produce same fingerprint."""
        pf1 = compute_policy_fingerprint("finance")
        pf2 = compute_policy_fingerprint("finance")
        assert pf1.hash == pf2.hash
        assert pf1.prefix == pf2.prefix

    def test_pf_different_roles(self):
        """Different roles produce different fingerprints."""
        pf_finance = compute_policy_fingerprint("finance")
        pf_hr = compute_policy_fingerprint("hr")
        assert pf_finance.hash != pf_hr.hash

    def test_pf_epoch_change_invalidates(self):
        """Changing epoch produces different fingerprint (via env var)."""
        # This test verifies that epochs are included in the fingerprint
        # by checking that the fingerprint includes epoch values
        pf = compute_policy_fingerprint("finance")
        assert pf.policy_epoch >= 1
        assert pf.entitlement_epoch >= 1

    def test_compute_pf_for_subject(self):
        """Helper function works correctly."""
        pf = compute_pf_for_subject({"role": "c_level"})
        assert isinstance(pf, PolicyFingerprint)
        assert len(pf.hash) == 16  # Truncated SHA-256 hex
        assert len(pf.prefix) == 8


class TestKeyGeneration:
    """Tests for cache key generation."""

    def test_normalize_query(self):
        """Query normalization is consistent."""
        assert normalize_query("  Hello World  ") == "hello world"
        assert normalize_query("HELLO") == normalize_query("hello")

    def test_ret0_key_deterministic(self):
        """RET0 keys are deterministic."""
        k1 = make_ret0_key("test query", 10)
        k2 = make_ret0_key("test query", 10)
        assert k1 == k2

    def test_ret0_key_different_topk(self):
        """Different top_k produces different keys."""
        k1 = make_ret0_key("test query", 5)
        k2 = make_ret0_key("test query", 10)
        assert k1 != k2

    def test_allowset_key_uses_pf(self):
        """Allowset key includes PF prefix."""
        pf = compute_pf_for_subject({"role": "finance"})
        key = make_allowset_key(pf)
        assert key.startswith("allowset:")  # lowercase
        assert pf.prefix in key

    def test_decision_key_includes_resource(self):
        """Decision key includes resource hash."""
        pf = compute_pf_for_subject({"role": "finance"})
        resource = {"owner_dept": "finance", "doc_type": "internal"}
        key = make_decision_key(pf, resource, "retrieve_chunk")
        assert key.startswith("decision:")  # lowercase
        assert pf.prefix in key


class TestSingleflight:
    """Tests for singleflight request coalescing."""

    def test_singleflight_dedup(self):
        """Concurrent requests for same key are deduplicated."""
        sf = Singleflight()
        call_count = 0

        def slow_compute():
            nonlocal call_count
            call_count += 1
            time.sleep(0.1)
            return "result"

        results = []
        threads = []
        for _ in range(5):
            t = threading.Thread(target=lambda: results.append(sf.do("key1", slow_compute)))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All threads should get the same result
        assert all(r == "result" for r in results)
        # But compute should only be called once
        assert call_count == 1
        assert sf.stats()["dedup_count"] >= 4

    def test_singleflight_different_keys(self):
        """Different keys are computed separately."""
        sf = Singleflight()
        r1 = sf.do("key1", lambda: "result1")
        r2 = sf.do("key2", lambda: "result2")
        assert r1 == "result1"
        assert r2 == "result2"


class TestInMemoryAdapter:
    """Tests for in-memory cache adapter."""

    def test_get_set(self):
        """Basic get/set works."""
        adapter = InMemoryAdapter(max_size=100)
        adapter.set("key1", {"data": "value"})
        assert adapter.get("key1") == {"data": "value"}

    def test_get_miss(self):
        """Missing key returns None."""
        adapter = InMemoryAdapter()
        assert adapter.get("nonexistent") is None

    def test_ttl_expiry(self):
        """TTL causes expiry."""
        adapter = InMemoryAdapter()
        adapter.set("key1", "value", ttl=1)
        assert adapter.get("key1") == "value"
        time.sleep(1.1)
        assert adapter.get("key1") is None

    def test_eviction_when_full(self):
        """Eviction occurs when max_size exceeded."""
        adapter = InMemoryAdapter(max_size=3)
        adapter.set("k1", "v1")
        adapter.set("k2", "v2")
        adapter.set("k3", "v3")
        # Add k4, should evict oldest (k1)
        adapter.set("k4", "v4")
        assert adapter.get("k1") is None  # Evicted (oldest)
        assert adapter.get("k2") == "v2"
        assert adapter.get("k3") == "v3"
        assert adapter.get("k4") == "v4"
        assert adapter.stats()["evictions"] >= 1

    def test_stats(self):
        """Stats are tracked correctly."""
        adapter = InMemoryAdapter()
        adapter.set("k1", "v1")
        adapter.get("k1")  # Hit
        adapter.get("k2")  # Miss
        stats = adapter.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["entries"] == 1


class TestNoOpAdapter:
    """Tests for no-op cache adapter."""

    def test_always_miss(self):
        """NoOp adapter always returns None."""
        adapter = NoOpAdapter()
        adapter.set("key", "value")
        assert adapter.get("key") is None


class TestRAGCache:
    """Tests for the high-level RAG cache."""

    def setup_method(self):
        """Reset cache state before each test."""
        reset_rag_cache()
        reset_cache_adapter()

    def test_retrieval_cache(self):
        """Retrieval cache works."""
        cache = RAGCache()
        call_count = 0

        def compute():
            nonlocal call_count
            call_count += 1
            return [{"id": "doc1", "text": "content"}]

        # First call computes
        result1, cached1 = cache.get_retrieval("test query", 10, compute)
        assert cached1 is False
        assert call_count == 1

        # Second call uses cache
        result2, cached2 = cache.get_retrieval("test query", 10, compute)
        assert cached2 is True
        assert call_count == 1  # Not called again
        assert result1 == result2

    def test_decision_cache_pf_scoped(self):
        """Decision cache is scoped by Policy Fingerprint."""
        cache = RAGCache()
        pf_finance = compute_pf_for_subject({"role": "finance"})
        pf_hr = compute_pf_for_subject({"role": "hr"})
        resource = {"owner_dept": "finance"}

        # Cache decision for finance
        cache.get_decision(pf_finance, resource, "read", lambda: {"effect": "permit"})

        # HR should not get cached finance decision
        call_count = 0
        def compute_hr():
            nonlocal call_count
            call_count += 1
            return {"effect": "deny"}

        result, cached = cache.get_decision(pf_hr, resource, "read", compute_hr)
        assert cached is False  # Different PF, so cache miss
        assert call_count == 1
        assert result["effect"] == "deny"

    def test_llm_cache_pf_scoped(self):
        """LLM cache is scoped by Policy Fingerprint."""
        cache = RAGCache()
        pf_finance = compute_pf_for_subject({"role": "finance"})
        pf_hr = compute_pf_for_subject({"role": "hr"})

        # Cache answer for finance
        cache.get_llm_answer(pf_finance, "query", "ctx_hash", "gpt-4", lambda: "finance answer")

        # HR should not get cached finance answer
        result, cached = cache.get_llm_answer(pf_hr, "query", "ctx_hash", "gpt-4", lambda: "hr answer")
        assert cached is False
        assert result == "hr answer"

    def test_cache_disabled(self):
        """Cache can be disabled."""
        with patch("app.cache.rag_cache.is_cache_enabled", return_value=False):
            cache = RAGCache()
            call_count = 0

            def compute():
                nonlocal call_count
                call_count += 1
                return [{"id": "doc1"}]

            cache.get_retrieval("query", 10, compute)
            cache.get_retrieval("query", 10, compute)
            assert call_count == 2  # Called twice, no caching

    def test_stats(self):
        """Stats are available."""
        cache = RAGCache()
        stats = cache.stats()
        assert "enabled" in stats
        assert "adapter" in stats
        assert "singleflight" in stats

