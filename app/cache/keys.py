"""
Cache key generation and normalization.

Defines the key schema for all cache types:
- RET0: Unfiltered retrieval cache
- ALLOWSET: Allowed document set cache
- DECISION: Policy decision cache
- RR: Rerank cache
- LLM: LLM answer cache
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Optional

from app.cache.policy_fingerprint import PolicyFingerprint, get_epochs


def _hash(data: Any, length: int = 16) -> str:
    """Create a truncated SHA256 hash of data."""
    if isinstance(data, str):
        encoded = data.encode("utf-8")
    else:
        encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def normalize_query(query: str) -> str:
    """
    Normalize a query string for consistent cache keys.
    
    - Lowercase
    - Strip leading/trailing whitespace
    - Collapse multiple spaces to single space
    """
    if not query:
        return ""
    normalized = query.lower().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def make_ret0_key(query: str, top_k: int, index_epoch: Optional[int] = None) -> str:
    """
    Create key for unfiltered retrieval cache (RET0).
    
    Safe to cache globally because results MUST be re-filtered by policy.
    
    Key format: ret0:{query_hash}:{index_epoch}:{top_k}
    """
    if index_epoch is None:
        index_epoch = get_epochs().index
    
    query_norm = normalize_query(query)
    query_hash = _hash(query_norm)
    return f"ret0:{query_hash}:{index_epoch}:{top_k}"


def make_allowset_key(pf: PolicyFingerprint, acl_epoch: Optional[int] = None) -> str:
    """
    Create key for allowed-set cache (ALLOWSET).
    
    Scoped by Policy Fingerprint to ensure user isolation.
    
    Key format: allowset:{pf_hash}:{acl_epoch}
    """
    if acl_epoch is None:
        acl_epoch = get_epochs().acl
    
    return f"allowset:{pf.hash}:{acl_epoch}"


def make_decision_key(
    pf: PolicyFingerprint,
    resource: Dict[str, Any],
    action: str,
) -> str:
    """
    Create key for policy decision cache (DECISION).
    
    Scoped by Policy Fingerprint and resource attributes.
    
    Key format: decision:{pf_hash}:{resource_hash}:{action}
    """
    # Include all policy-relevant resource attributes.
    # agg_type is needed because aggregate sub-types may have different
    # access rules (e.g., headcount vs salary aggregates).
    relevant_attrs = {
        "owner_dept": resource.get("owner_dept"),
        "doc_type": resource.get("doc_type"),
        "sensitivity": resource.get("sensitivity"),
        "agg_type": resource.get("agg_type"),
    }
    resource_hash = _hash(relevant_attrs)
    return f"decision:{pf.hash}:{resource_hash}:{action}"


def make_rerank_key(
    query: str,
    candidate_ids: List[str],
    reranker_version: str = "v1",
) -> str:
    """
    Create key for rerank cache (RR).
    
    Safe to cache globally because it only stores ordering, not content.
    
    Key format: rr:{query_hash}:{ids_hash}:{version}
    """
    query_norm = normalize_query(query)
    query_hash = _hash(query_norm)
    # Sort IDs for consistent hashing regardless of input order
    ids_hash = _hash(sorted(candidate_ids))
    return f"rr:{query_hash}:{ids_hash}:{reranker_version}"


def make_llm_key(
    pf: PolicyFingerprint,
    query: str,
    context_hash: str,
    model: str,
) -> str:
    """
    Create key for LLM answer cache.
    
    MUST be scoped by Policy Fingerprint since answers contain filtered content.
    
    Key format: llm:{pf_hash}:{query_hash}:{context_hash}:{model}
    """
    query_norm = normalize_query(query)
    query_hash = _hash(query_norm)
    # Truncate model name for key brevity
    model_short = model.replace("-", "").replace(".", "")[:12]
    return f"llm:{pf.hash}:{query_hash}:{context_hash}:{model_short}"


def make_embedding_key(text: str, model: str) -> str:
    """
    Create key for embedding cache.
    
    Safe to cache globally (deterministic, no policy data).
    
    Key format: emb:{model}:{text_hash}
    """
    text_hash = _hash(text)
    model_short = model.replace("-", "").replace(".", "")[:12]
    return f"emb:{model_short}:{text_hash}"

