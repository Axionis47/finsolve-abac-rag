"""
Policy-Safe Caching Layer for ABAC RAG System.

This module provides:
- CacheAdapter interface with InMemory and Redis implementations
- Policy Fingerprint (PF) computation for safe cache scoping
- Singleflight request coalescing to prevent stampedes
- Key normalization and schema helpers
- RAGCache high-level interface for pipeline caching

Safety Invariant:
    Cached items must either be (a) content-neutral and re-filtered before use,
    OR (b) keyed by a Policy Fingerprint (PF) that captures all subject attributes
    affecting access decisions.
"""

from app.cache.base import (
    CacheAdapter,
    InMemoryAdapter,
    NoOpAdapter,
    get_cache_adapter,
    reset_cache_adapter,
)
from app.cache.policy_fingerprint import (
    PolicyFingerprint,
    compute_policy_fingerprint,
    compute_pf_for_subject,
    get_epochs,
    Epochs,
)
from app.cache.keys import (
    normalize_query,
    make_ret0_key,
    make_allowset_key,
    make_decision_key,
    make_rerank_key,
    make_llm_key,
    make_embedding_key,
)
from app.cache.singleflight import Singleflight, get_singleflight
from app.cache.rag_cache import RAGCache, get_rag_cache, reset_rag_cache

__all__ = [
    # Base adapters
    "CacheAdapter",
    "InMemoryAdapter",
    "NoOpAdapter",
    "get_cache_adapter",
    "reset_cache_adapter",
    # Policy Fingerprint
    "PolicyFingerprint",
    "compute_policy_fingerprint",
    "compute_pf_for_subject",
    "get_epochs",
    "Epochs",
    # Key generation
    "normalize_query",
    "make_ret0_key",
    "make_allowset_key",
    "make_decision_key",
    "make_rerank_key",
    "make_llm_key",
    "make_embedding_key",
    # Singleflight
    "Singleflight",
    "get_singleflight",
    # RAG Cache
    "RAGCache",
    "get_rag_cache",
    "reset_rag_cache",
]

