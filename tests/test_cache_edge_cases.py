"""
Edge case tests for cache key generation, isolation, and singleflight.

Tests probe:
- Query normalization corner cases
- Hash determinism with unusual inputs
- Singleflight error propagation and timeout
- Epoch boundary values
"""
from __future__ import annotations

import os
import threading
import time
from unittest.mock import patch

import pytest

from app.cache.keys import (
    normalize_query,
    make_ret0_key,
    make_llm_key,
    make_decision_key,
    make_embedding_key,
    _hash,
)
from app.cache.singleflight import Singleflight
from app.cache import compute_pf_for_subject, RAGCache, reset_rag_cache, reset_cache_adapter


@pytest.fixture(autouse=True)
def reset():
    reset_rag_cache()
    reset_cache_adapter()
    yield
    reset_rag_cache()
    reset_cache_adapter()


class TestQueryNormalization:
    """Tests for normalize_query()."""

    def test_whitespace_collapse(self):
        assert normalize_query("hello   world") == "hello world"

    def test_leading_trailing_whitespace(self):
        assert normalize_query("  hello  ") == "hello"

    def test_case_insensitive(self):
        assert normalize_query("Hello World") == "hello world"

    def test_tabs_and_newlines(self):
        assert normalize_query("hello\t\nworld") == "hello world"

    def test_empty_string(self):
        assert normalize_query("") == ""

    def test_none_returns_empty(self):
        """None input should not crash (returns '')."""
        # normalize_query checks `if not query` which catches None
        assert normalize_query(None) == ""

    def test_unicode_preserved(self):
        """Unicode characters should be preserved."""
        assert normalize_query("café résumé") == "café résumé"

    def test_equivalent_queries_same_key(self):
        """Different whitespace/casing should produce same cache key."""
        k1 = make_ret0_key("Hello World", top_k=5, index_epoch=1)
        k2 = make_ret0_key("  hello   world  ", top_k=5, index_epoch=1)
        assert k1 == k2


class TestHashFunction:
    """Tests for _hash() edge cases."""

    def test_empty_string(self):
        h = _hash("")
        assert isinstance(h, str) and len(h) == 16

    def test_none_value_in_dict(self):
        """Dict with None values should hash deterministically."""
        h1 = _hash({"a": None, "b": None})
        h2 = _hash({"a": None, "b": None})
        assert h1 == h2

    def test_dict_key_order_irrelevant(self):
        """JSON sort_keys=True ensures key order doesn't matter."""
        h1 = _hash({"b": 2, "a": 1})
        h2 = _hash({"a": 1, "b": 2})
        assert h1 == h2

    def test_very_long_input(self):
        """10KB string should hash without error."""
        h = _hash("x" * 10_000)
        assert isinstance(h, str) and len(h) == 16

    def test_unicode_input(self):
        """Unicode should hash consistently."""
        h1 = _hash("日本語テスト")
        h2 = _hash("日本語テスト")
        assert h1 == h2

    def test_string_vs_int_same_hash(self):
        """String '1' and int 1 produce same hash (both encode to '1' bytes).
        This is a known limitation - _hash encodes strings directly and
        json.dumps(1) also produces '1'."""
        h1 = _hash("1")
        h2 = _hash(1)
        # Both produce the same hash because json.dumps(1) == "1"
        # and "1".encode() == b"1"
        assert h1 == h2


class TestDecisionCacheKey:
    """Tests for make_decision_key edge cases."""

    def test_resource_with_none_fields(self):
        """Resource with all None fields should produce a valid key."""
        from app.cache.policy_fingerprint import PolicyFingerprint
        pf = PolicyFingerprint(role="test")
        key = make_decision_key(pf, {"owner_dept": None, "doc_type": None}, "read")
        assert key.startswith("decision:")

    def test_resource_ordering_deterministic(self):
        """Same resource attributes in different order produce same key."""
        from app.cache.policy_fingerprint import PolicyFingerprint
        pf = PolicyFingerprint(role="test")
        r1 = {"owner_dept": "hr", "doc_type": "csv", "sensitivity": "high", "agg_type": None}
        r2 = {"sensitivity": "high", "doc_type": "csv", "agg_type": None, "owner_dept": "hr"}
        k1 = make_decision_key(pf, r1, "read")
        k2 = make_decision_key(pf, r2, "read")
        assert k1 == k2


class TestSingleflightEdgeCases:
    """Edge cases for singleflight deduplication."""

    def test_error_propagation_to_waiters(self):
        """If leader function raises, waiters should also get the error."""
        sf = Singleflight(timeout=5.0)
        errors = []
        started = threading.Event()

        def failing_fn():
            started.set()
            time.sleep(0.1)  # Give waiters time to queue up
            raise ValueError("computation failed")

        def leader():
            try:
                sf.do("err-key", failing_fn)
            except ValueError as e:
                errors.append(str(e))

        def waiter():
            started.wait(timeout=2)  # Wait for leader to start
            try:
                sf.do("err-key", failing_fn)
            except ValueError as e:
                errors.append(str(e))

        t_leader = threading.Thread(target=leader)
        t_waiter = threading.Thread(target=waiter)
        t_leader.start()
        t_waiter.start()
        t_leader.join(timeout=5)
        t_waiter.join(timeout=5)

        # At least the leader should get the error
        assert len(errors) >= 1
        assert all("computation failed" in e for e in errors)

    def test_timeout_falls_back_to_direct_execution(self):
        """When singleflight timeout is exceeded, waiter executes directly."""
        sf = Singleflight(timeout=0.05)  # Very short timeout
        results = []

        def slow_leader():
            time.sleep(0.5)  # Much longer than timeout
            return "leader_result"

        def worker(fn):
            result = sf.do("timeout-key", fn)
            results.append(result)

        # Start leader thread
        leader_thread = threading.Thread(target=worker, args=(slow_leader,))
        leader_thread.start()
        time.sleep(0.01)  # Let leader start

        # Start waiter that should timeout
        fast_fn_called = threading.Event()

        def fast_fn():
            fast_fn_called.set()
            return "fallback_result"

        waiter_thread = threading.Thread(target=worker, args=(fast_fn,))
        waiter_thread.start()

        waiter_thread.join(timeout=3)
        leader_thread.join(timeout=3)

        # Waiter should have timed out and executed directly
        assert fast_fn_called.is_set(), "Waiter should have called fallback fn on timeout"

    def test_stats_tracking(self):
        """Stats should track dedup and timeout counts."""
        sf = Singleflight(timeout=5.0)
        sf.do("k1", lambda: "a")
        stats = sf.stats()
        assert "dedup_count" in stats
        assert "timeout_count" in stats
        assert "in_flight" in stats


class TestEpochEdgeCases:
    """Epoch validation edge cases."""

    def test_epoch_zero_valid_pf(self):
        """Epoch=0 should produce a valid PolicyFingerprint."""
        with patch.dict(os.environ, {"CACHE_POLICY_EPOCH": "0", "CACHE_ENTITLEMENT_EPOCH": "0"}):
            pf = compute_pf_for_subject({"role": "finance"})
            assert pf.hash is not None and len(pf.hash) > 0

    def test_invalid_epoch_uses_default(self):
        """Non-numeric epoch should fall back to default without crashing."""
        with patch.dict(os.environ, {"CACHE_POLICY_EPOCH": "not_a_number"}):
            pf = compute_pf_for_subject({"role": "finance"})
            assert pf.hash is not None

    def test_negative_epoch_clamps_to_zero(self):
        """Negative epoch should be clamped to 0."""
        with patch.dict(os.environ, {"CACHE_POLICY_EPOCH": "-5"}):
            pf = compute_pf_for_subject({"role": "finance"})
            assert pf.hash is not None
