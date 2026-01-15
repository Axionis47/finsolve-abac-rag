"""
RAG Pipeline Cache Service.

Provides high-level caching functions for the RAG pipeline:
- Retrieval cache (RET0)
- Allowed-set cache (ALLOWSET) 
- Policy decision cache (DECISION)
- Rerank cache (RR)
- LLM answer cache

All policy-sensitive caches are scoped by Policy Fingerprint.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from app.cache.base import CacheAdapter, get_cache_adapter
from app.cache.policy_fingerprint import PolicyFingerprint, compute_pf_for_subject
from app.cache.keys import (
    make_ret0_key,
    make_allowset_key,
    make_decision_key,
    make_rerank_key,
    make_llm_key,
)
from app.cache.singleflight import get_singleflight
from app.utils.config import get_cache_ttl, is_cache_enabled
from app.utils.audit import log_event


class RAGCache:
    """
    High-level cache interface for RAG pipeline.
    
    Provides policy-safe caching with proper key scoping,
    singleflight deduplication, and observability.
    """

    def __init__(self, adapter: Optional[CacheAdapter] = None):
        self._adapter = adapter or get_cache_adapter()
        self._singleflight = get_singleflight()
        self._enabled = is_cache_enabled()

    # -------------------------------------------------------------------------
    # Unfiltered Retrieval Cache (RET0) - Safe to cache globally
    # -------------------------------------------------------------------------

    def get_retrieval(
        self,
        query: str,
        top_k: int,
        compute_fn: Callable[[], List[Dict[str, Any]]],
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Get or compute unfiltered retrieval results.
        
        IMPORTANT: Results MUST be filtered by policy after retrieval.
        
        Returns:
            Tuple of (results, was_cached)
        """
        if not self._enabled:
            return compute_fn(), False
        
        key = make_ret0_key(query, top_k)
        ttl = get_cache_ttl("retrieval")
        
        cached = self._adapter.get(key)
        if cached is not None:
            log_event(None, "cache.hit", {"cache": "ret0", "key_prefix": key[:20]})
            return cached, True
        
        # Use singleflight to prevent stampede
        result = self._singleflight.do(key, compute_fn)
        self._adapter.set(key, result, ttl)
        log_event(None, "cache.miss", {"cache": "ret0", "key_prefix": key[:20]})
        return result, False

    # -------------------------------------------------------------------------
    # Allowed-Set Cache (ALLOWSET) - Scoped by Policy Fingerprint
    # -------------------------------------------------------------------------

    def get_allowed_set(
        self,
        pf: PolicyFingerprint,
        compute_fn: Callable[[], Set[str]],
    ) -> Tuple[Set[str], bool]:
        """
        Get or compute the set of document IDs the user can access.
        
        Returns:
            Tuple of (allowed_doc_ids, was_cached)
        """
        if not self._enabled:
            return compute_fn(), False
        
        key = make_allowset_key(pf)
        ttl = get_cache_ttl("decision")
        
        cached = self._adapter.get(key)
        if cached is not None:
            log_event(None, "cache.hit", {"cache": "allowset", "pf": pf.prefix})
            return set(cached), True
        
        result = self._singleflight.do(key, compute_fn)
        # Store as list for JSON serialization
        self._adapter.set(key, list(result), ttl)
        log_event(None, "cache.miss", {"cache": "allowset", "pf": pf.prefix})
        return result, False

    # -------------------------------------------------------------------------
    # Policy Decision Cache (DECISION) - Scoped by PF + resource
    # -------------------------------------------------------------------------

    def get_decision(
        self,
        pf: PolicyFingerprint,
        resource: Dict[str, Any],
        action: str,
        compute_fn: Callable[[], Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], bool]:
        """
        Get or compute a policy decision.
        
        Returns:
            Tuple of (decision_dict, was_cached)
        """
        if not self._enabled:
            return compute_fn(), False
        
        key = make_decision_key(pf, resource, action)
        ttl = get_cache_ttl("decision")
        
        cached = self._adapter.get(key)
        if cached is not None:
            return cached, True

        # Use singleflight to prevent stampede
        result = self._singleflight.do(key, compute_fn)
        self._adapter.set(key, result, ttl)
        return result, False

    # -------------------------------------------------------------------------
    # Rerank Cache (RR) - Safe to cache globally (ordering only)
    # -------------------------------------------------------------------------

    def get_rerank(
        self,
        query: str,
        candidate_ids: List[str],
        reranker_version: str,
        compute_fn: Callable[[], List[int]],
    ) -> Tuple[List[int], bool]:
        """
        Get or compute reranking order.

        Returns:
            Tuple of (ordered_indices, was_cached)
        """
        if not self._enabled:
            return compute_fn(), False

        key = make_rerank_key(query, candidate_ids, reranker_version)
        ttl = get_cache_ttl("retrieval")

        cached = self._adapter.get(key)
        if cached is not None:
            return cached, True

        result = compute_fn()
        self._adapter.set(key, result, ttl)
        return result, False

    # -------------------------------------------------------------------------
    # LLM Answer Cache - MUST be scoped by Policy Fingerprint
    # -------------------------------------------------------------------------

    def get_llm_answer(
        self,
        pf: PolicyFingerprint,
        query: str,
        context_hash: str,
        model: str,
        compute_fn: Callable[[], str],
    ) -> Tuple[str, bool]:
        """
        Get or compute LLM answer.

        IMPORTANT: This cache is scoped by PF because answers contain
        policy-filtered content.

        Returns:
            Tuple of (answer, was_cached)
        """
        if not self._enabled:
            return compute_fn(), False

        key = make_llm_key(pf, query, context_hash, model)
        ttl = get_cache_ttl("llm")

        cached = self._adapter.get(key)
        if cached is not None:
            log_event(None, "cache.hit", {"cache": "llm", "pf": pf.prefix})
            return cached, True

        result = self._singleflight.do(key, compute_fn)
        self._adapter.set(key, result, ttl)
        log_event(None, "cache.miss", {"cache": "llm", "pf": pf.prefix})
        return result, False

    # -------------------------------------------------------------------------
    # Statistics and Management
    # -------------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "enabled": self._enabled,
            "adapter": self._adapter.stats(),
            "singleflight": self._singleflight.stats(),
        }

    def clear(self) -> None:
        """Clear all cached entries."""
        self._adapter.clear()


# Global RAG cache instance
_rag_cache: Optional[RAGCache] = None


def get_rag_cache() -> RAGCache:
    """Get the global RAG cache instance."""
    global _rag_cache
    if _rag_cache is None:
        _rag_cache = RAGCache()
    return _rag_cache


def reset_rag_cache() -> None:
    """Reset the global RAG cache. Useful for testing."""
    global _rag_cache
    if _rag_cache is not None:
        _rag_cache.clear()
    _rag_cache = None

