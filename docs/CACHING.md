# Policy-Safe Caching Layer Design

This document describes the caching architecture for the FinSolve ABAC RAG system.
The design ensures **zero data leakage** while improving latency and reducing API costs.

## Table of Contents

1. [Safety Invariant](#safety-invariant)
2. [Threat Model](#threat-model)
3. [Policy Fingerprint (PF)](#policy-fingerprint-pf)
4. [Epoch-Based Invalidation](#epoch-based-invalidation)
5. [Cache Types](#cache-types)
6. [Key Schema & TTLs](#key-schema--ttls)
7. [Cache Store & Concurrency](#cache-store--concurrency)
8. [Observability](#observability)
9. [Acceptance Criteria](#acceptance-criteria)
10. [Rollback Plan](#rollback-plan)

---

## Safety Invariant

> **Cached items must either be (a) content-neutral and re-filtered before use, OR (b) keyed by a Policy Fingerprint (PF) that captures all subject attributes affecting access decisions.**

This invariant ensures no cached data can leak across permission boundaries.

---

## Threat Model

### What Can Leak?

| Artifact | Risk | Mitigation |
|----------|------|------------|
| Document chunks/text | HIGH | Scoped by PF |
| LLM answers | HIGH | Scoped by PF |
| Document IDs | LOW | Global cache OK (re-filter) |
| Embeddings | LOW | Global cache OK (deterministic) |
| Rankings | LOW | Global cache OK |

### Attack Vectors Mitigated

1. **Cross-User Cache Poisoning** → PF-scoped keys prevent serving wrong user's data
2. **Role Escalation** → PF includes role; different roles = different cache entries
3. **Stale Entitlement** → Epoch bump invalidates cache on role changes
4. **Policy Change Bypass** → Policy epoch bump invalidates on policy updates

---

## Policy Fingerprint (PF)

The PF is a SHA256 hash of canonicalized subject attributes:

```python
PF = sha256(canonical_json({
    "role": "engineering",              # Subject's role
    "tenant_id": "default",             # Multi-tenant isolation
    "policy_epoch": 1,                  # Bumped on policy.yaml change
    "entitlement_epoch": 1,             # Bumped on user-role change
    "flags": {
        "hr_aggregate_mode": "disabled",
        "hr_masked_rows_mode": "disabled"
    }
}))
```

### Attributes Included

| Attribute | Source | Why |
|-----------|--------|-----|
| `role` | `subject.role` | All policy rules check this |
| `tenant_id` | Config (default: "default") | Future multi-tenant support |
| `policy_epoch` | Env `CACHE_POLICY_EPOCH` | Invalidate on policy change |
| `entitlement_epoch` | Env `CACHE_ENTITLEMENT_EPOCH` | Invalidate on role assignment |
| `flags.*` | Runtime config | Affects HR data access rules |

### What's NOT Included

- `username` - Not used in policy decisions
- `resource attributes` - Part of cache key separately
- `action` - Part of cache key separately

---

## Epoch-Based Invalidation

Epochs allow cache invalidation without mass deletes. Bumping an epoch changes all derived keys.

| Epoch | Env Variable | Default | When to Bump |
|-------|--------------|---------|--------------|
| Policy | `CACHE_POLICY_EPOCH` | 1 | `docs/policy.yaml` modified |
| Entitlement | `CACHE_ENTITLEMENT_EPOCH` | 1 | User-role assignments change |
| ACL | `CACHE_ACL_EPOCH` | 1 | Document permissions change |
| Index | `CACHE_INDEX_EPOCH` | 1 | Vector/BM25 index rebuilt |

### How to Bump

```bash
# On policy change
export CACHE_POLICY_EPOCH=2

# Or update .env file
echo "CACHE_POLICY_EPOCH=2" >> .env
```

---

## Cache Types

### 1. Unfiltered Retrieval Cache (RET0) ✅ SAFE

Caches raw retrieval candidates **before** policy filtering. Safe because:
- Results are content-neutral (IDs + scores only in key derivation)
- MUST re-apply policy filter after cache hit

```
Key:   ret0:{hash(query_norm)}:{index_epoch}:{top_k}
Value: [(doc_id, score, chunk_ref), ...]
TTL:   1 hour
```

### 2. Allowed-Set Cache (ALLOWSET) ✅ SAFE

Caches the set of document IDs a user can access. Keyed by PF.

```
Key:   allowset:{PF}:{acl_epoch}
Value: Set[doc_id] or compressed bitmap
TTL:   1 hour
```

Usage: `filtered = candidates ∩ allowed_set(PF)`

### 3. Policy Decision Cache (DECISION) ✅ SAFE

Caches individual PDP decisions. Alternative to ALLOWSET for fine-grained access.

```
Key:   decision:{PF}:{hash(resource_attrs)}:{action}
Value: {effect: "permit"|"deny", rule: "rule_name"}
TTL:   1 hour
```

### 4. Rerank Cache (RR) ✅ SAFE

Caches reranking results. Content-neutral (just ordering).

```
Key:   rr:{hash(query_norm)}:{hash(sorted(candidate_ids))}:{reranker_version}
Value: [ordered_indices]
TTL:   1 hour
```

### 5. LLM Answer Cache (LLM) ⚠️ SCOPED

Caches LLM responses. **Must be scoped by PF** since answers contain filtered content.

```
Key:   llm:{PF}:{hash(query)}:{hash(context)}:{model}
Value: answer_text
TTL:   1 hour
```

---

## Key Schema & TTLs

| Cache | Key Pattern | TTL | Size Limit |
|-------|-------------|-----|------------|
| RET0 | `ret0:{query_hash}:{index_epoch}:{top_k}` | 3600s | 100KB |
| ALLOWSET | `allowset:{pf}:{acl_epoch}` | 3600s | 1MB |
| DECISION | `decision:{pf}:{resource_hash}:{action}` | 3600s | 1KB |
| RR | `rr:{query_hash}:{ids_hash}:{version}` | 3600s | 10KB |
| LLM | `llm:{pf}:{query_hash}:{ctx_hash}:{model}` | 3600s | 50KB |
| EMB | `emb:{model}:{text_hash}` | ∞ | 10KB |

### Key Normalization

- **Query normalization**: lowercase, strip whitespace, collapse multiple spaces
- **Hash function**: SHA256, truncated to 16 hex chars for keys
- **Canonical JSON**: Keys sorted, no whitespace, UTF-8 encoded

---

## Cache Store & Concurrency

### Adapter Interface

```python
class CacheAdapter(ABC):
    def get(self, key: str) -> Optional[Any]: ...
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None: ...
    def delete(self, key: str) -> None: ...
    def get_or_compute(self, key: str, compute_fn: Callable, ttl: int) -> Any: ...
```

### Implementations

1. **InMemoryAdapter** (default): Thread-safe dict with TTL tracking
2. **RedisAdapter** (optional): Redis client with JSON serialization

### Stampede Protection (Singleflight)

Prevents multiple concurrent requests from computing the same expensive value:

```python
# Only one request computes; others wait and share result
result = cache.get_or_compute(
    key="allowset:abc123:1",
    compute_fn=lambda: compute_allowed_set(user),
    ttl=3600
)
```

### Configuration

```bash
# Cache backend (memory or redis)
CACHE_BACKEND=memory  # or "redis"
CACHE_REDIS_URL=redis://localhost:6379/0

# Feature flag to disable caching
CACHE_ENABLED=true
```

---

## Observability

### Metrics

| Metric | Type | Labels |
|--------|------|--------|
| `cache_hits_total` | Counter | cache_type |
| `cache_misses_total` | Counter | cache_type |
| `cache_latency_seconds` | Histogram | cache_type, operation |
| `cache_size_bytes` | Gauge | cache_type |
| `cache_evictions_total` | Counter | cache_type |
| `singleflight_dedup_total` | Counter | cache_type |

### Logging

Logs include (never raw attributes):
- PF hash prefix (first 8 chars)
- Epoch values
- Cache operation (hit/miss/set)
- Latency

Example log:
```json
{
  "event": "cache.hit",
  "cache": "allowset",
  "pf_prefix": "a1b2c3d4",
  "epochs": {"policy": 1, "acl": 1},
  "latency_ms": 0.5
}
```

---

## Acceptance Criteria

### Security Tests (Zero Leakage)

1. ✅ Two users with different roles NEVER share cached policy-scoped entries
2. ✅ Bumping policy_epoch invalidates all PF-keyed caches
3. ✅ Bumping entitlement_epoch invalidates affected user's caches
4. ✅ Unfiltered cache (RET0) results are ALWAYS re-filtered

### Performance Tests

1. ✅ Cache hit reduces retrieval latency by >50%
2. ✅ Singleflight prevents duplicate computation under load
3. ✅ Memory usage stays within configured limits

### Property Tests

```python
@given(user_a=user_strategy(), user_b=user_strategy())
def test_no_cross_user_leakage(user_a, user_b):
    assume(user_a.role != user_b.role)
    pf_a = compute_policy_fingerprint(user_a)
    pf_b = compute_policy_fingerprint(user_b)
    assert pf_a != pf_b
    # Cached item for pf_a is never returned for pf_b
```

---

## Rollback Plan

### Feature Flag

All caching can be disabled via environment variable:

```bash
CACHE_ENABLED=false
```

This bypasses all new cache code and falls back to original behavior.

### Gradual Rollout

1. Deploy with `CACHE_ENABLED=false`
2. Enable for internal users: `CACHE_ENABLED=true` on staging
3. Enable for 10% of prod traffic
4. Monitor metrics for anomalies
5. Full rollout

### Emergency Disable

```bash
# Immediate disable
export CACHE_ENABLED=false
# Restart app or send SIGHUP

# Clear all caches (if Redis)
redis-cli FLUSHDB
```

---

## Implementation Checklist

- [x] `app/cache/base.py` - CacheAdapter interface + InMemoryAdapter + NoOpAdapter
- [x] `app/cache/redis_adapter.py` - Redis implementation (optional)
- [x] `app/cache/policy_fingerprint.py` - PF computation with epochs
- [x] `app/cache/singleflight.py` - Request coalescing
- [x] `app/cache/keys.py` - Key generation and normalization
- [x] `app/cache/rag_cache.py` - High-level RAGCache interface
- [x] `app/cache/__init__.py` - Module exports
- [x] Update `app/utils/config.py` - Add epoch/cache config
- [x] Update `app/routes/search.py` - Wire decision caching
- [x] Update `app/routes/chat.py` - Wire decision + LLM caching
- [x] Update `app/routes/admin.py` - Add cache stats/clear endpoints
- [x] Update `app/services/generation.py` - Add PF to LLM cache
- [x] `tests/test_cache_layer.py` - Unit tests for cache module

---

## API Endpoints

### Cache Statistics

```
GET /admin/status
```

Returns cache statistics including:
- Legacy cache stats
- RAG cache stats (hits, misses, hit rate, entries)
- Singleflight dedup count

### Clear RAG Cache

```
POST /admin/rag_cache/clear
```

Clears all policy-safe cached entries. Requires c_level role.

### Get RAG Cache Stats

```
GET /admin/rag_cache/stats
```

Returns detailed RAG cache statistics.

---

## Usage Examples

### Computing Policy Fingerprint

```python
from app.cache import compute_pf_for_subject

# Compute PF for a user
pf = compute_pf_for_subject({"role": "finance"})
print(f"PF hash: {pf.hash}")  # e.g., "a1b2c3d4e5f6g7h8"
print(f"PF prefix: {pf.prefix}")  # e.g., "a1b2c3d4"
```

### Using RAGCache

```python
from app.cache import get_rag_cache, compute_pf_for_subject

cache = get_rag_cache()
pf = compute_pf_for_subject({"role": "engineering"})

# Cache a policy decision
decision, was_cached = cache.get_decision(
    pf=pf,
    resource={"owner_dept": "engineering", "doc_type": "internal"},
    action="retrieve_chunk",
    compute_fn=lambda: pdp.evaluate(subject, resource, action),
)

# Cache an LLM answer (scoped by PF)
answer, was_cached = cache.get_llm_answer(
    pf=pf,
    query="What is the Q4 budget?",
    context_hash="abc123",
    model="gpt-4o-mini",
    compute_fn=lambda: llm.generate(messages),
)
```

### Disabling Cache

```bash
# Disable all caching
export CACHE_ENABLED=false

# Or use Redis backend
export CACHE_BACKEND=redis
export CACHE_REDIS_URL=redis://localhost:6379/0
```

### Invalidating Cache via Epochs

```bash
# After policy.yaml changes
export CACHE_POLICY_EPOCH=2

# After user role assignments change
export CACHE_ENTITLEMENT_EPOCH=2

# After document permissions change
export CACHE_ACL_EPOCH=2

# After index rebuild
export CACHE_INDEX_EPOCH=2
```
