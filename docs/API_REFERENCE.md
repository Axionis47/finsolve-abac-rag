# API Reference

All endpoints require HTTP Basic authentication unless otherwise noted.

Base URL: `http://localhost:8000`

---

## Authentication

All requests must include an `Authorization` header with HTTP Basic credentials:

```
Authorization: Basic base64(username:password)
```

The server resolves the username to a role. The role determines which documents the user can access.

---

## Chat Endpoints

### POST /chat

Ask a question. The system retrieves relevant context, applies policy filtering, and generates an LLM answer with citations.

**Request Body:**

```json
{
  "message": "What is the Q4 marketing budget?",
  "top_k": 5,
  "rerank": false,
  "reranker": null,
  "multihop": false,
  "multihop_auto": true,
  "persist_dir": null,
  "base_dir": null
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `message` | string | (required) | The user's question |
| `top_k` | integer | 5 | Number of context chunks to retrieve |
| `rerank` | boolean | false | Enable second-pass reranking |
| `reranker` | string | null | Reranker backend: "simple", "openai", "openai_llm" |
| `multihop` | boolean | false | Enable multi-hop query decomposition |
| `multihop_auto` | boolean | true | Auto-detect if query needs multi-hop |
| `persist_dir` | string | null | ChromaDB directory (default from config) |
| `base_dir` | string | null | Knowledge base directory (default: resources/data) |

**Response:**

```json
{
  "answer": "The Q4 2024 marketing budget was $2.5M, allocated across...",
  "citations": [
    {
      "source_path": "resources/data/marketing/market_report_q4_2024.md",
      "section_path": "Q4 Market Analysis > Budget Allocation",
      "rule": "same_department_internal_docs"
    }
  ],
  "metrics": {
    "retrieve_ms": 45.2,
    "llm_ms": 1200.5,
    "total_ms": 1250.3
  },
  "correlation_id": "a1b2c3d4e5f6",
  "multihop": null
}
```

**Response Headers:**

```
X-Correlation-ID: a1b2c3d4e5f6
```

---

### POST /chat/stream

Streaming version of /chat using Server-Sent Events (SSE). Tokens are sent as they are generated.

**Request Body:** Same as POST /chat.

**Response:** `text/event-stream`

Events are sent in this order:

1. **Metadata event** (first):
```
data: {"type": "metadata", "citations": [...], "correlation_id": "abc123"}
```

2. **Token events** (streamed):
```
data: {"type": "token", "content": "The"}
data: {"type": "token", "content": " Q4"}
data: {"type": "token", "content": " budget"}
```

3. **Done event** (last):
```
data: {"type": "done"}
```

4. **Error event** (on failure):
```
data: {"type": "error", "message": "Connection timeout"}
```

---

## Search Endpoints

### POST /search

Sparse keyword search using BM25 only.

**Request Body:**

```json
{
  "query": "marketing ROI",
  "top_k": 5
}
```

**Response:**

```json
{
  "results": [
    {
      "text": "The Q1 2024 marketing campaign achieved 3.2x ROI...",
      "source_path": "resources/data/marketing/marketing_report_q1_2024.md",
      "section_path": "Campaign Performance > ROI Analysis",
      "score": 12.5
    }
  ],
  "metrics": {
    "search_ms": 3.2,
    "total_ms": 5.1
  },
  "correlation_id": "d4e5f6a7"
}
```

---

### POST /search/dense

Dense vector search using ChromaDB embeddings only.

**Request Body:**

```json
{
  "query": "employee benefits and vacation policy",
  "top_k": 5,
  "rerank": true,
  "reranker": "simple"
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | string | (required) | Search query |
| `top_k` | integer | 5 | Number of results to return |
| `rerank` | boolean | false | Enable reranking |
| `reranker` | string | null | "simple", "openai", or "openai_llm" |
| `persist_dir` | string | null | ChromaDB directory override |
| `base_dir` | string | null | Knowledge base directory override |

**Response:**

```json
{
  "results": [...],
  "metrics": {
    "init_ms": 2.1,
    "embed_ms": 120.3,
    "retrieve_ms": 15.4,
    "pdp_ms": 1.2,
    "rerank_ms": 0.0,
    "total_ms": 139.0
  },
  "correlation_id": "e5f6a7b8"
}
```

---

### POST /search/hybrid

Combined dense + sparse search with Reciprocal Rank Fusion.

**Request Body:** Same as POST /search/dense.

**Response:** Same format as POST /search/dense, with additional fusion metrics:

```json
{
  "results": [...],
  "metrics": {
    "init_ms": 2.1,
    "embed_ms": 120.3,
    "dense_ms": 15.4,
    "sparse_ms": 3.2,
    "fuse_ms": 0.5,
    "pdp_ms": 1.2,
    "rerank_ms": 0.0,
    "total_ms": 142.7,
    "dense_skipped": false
  },
  "correlation_id": "f6a7b8c9"
}
```

---

## HR Endpoints

### GET /hr/rows

Returns full HR employee records. Restricted to HR and C-level roles.

**Response:**

```json
{
  "rows": [
    {
      "employee_id": "E001",
      "full_name": "John Smith",
      "role": "Software Engineer",
      "department": "Engineering",
      "email": "john.smith@finsolve.com",
      "location": "San Francisco",
      "salary": 145000,
      "performance_rating": 4.2,
      "date_of_birth": "1990-03-15",
      "date_of_joining": "2021-06-01"
    }
  ],
  "count": 100,
  "correlation_id": "a7b8c9d0"
}
```

**Error (403):** Returned if user role is not HR or C-level.

---

### GET /hr/rows_masked

Returns HR records with PII fields redacted. Available to non-HR users when `HR_MASKED_ROWS_MODE=enabled`.

**Response:**

```json
{
  "rows": [
    {
      "employee_id": "[REDACTED]",
      "full_name": "[REDACTED]",
      "role": "Software Engineer",
      "department": "Engineering",
      "email": "[REDACTED]",
      "location": "San Francisco",
      "salary": "[REDACTED]",
      "performance_rating": "[REDACTED]",
      "date_of_birth": "[REDACTED]",
      "date_of_joining": "2021-06-01"
    }
  ],
  "count": 100,
  "correlation_id": "b8c9d0e1"
}
```

**Error (403):** Returned if `HR_MASKED_ROWS_MODE=disabled` and user is not HR/C-level.

---

### GET /hr/aggregate

Returns pre-computed aggregate HR statistics. Available to non-HR users when `HR_AGGREGATE_MODE=enabled`.

**Response:**

```json
{
  "aggregates": {
    "total_employees": 100,
    "count_by_department": {
      "Engineering": 35,
      "Marketing": 20,
      "Finance": 15,
      "HR": 10,
      "Operations": 20
    },
    "average_salary_by_department": {
      "Engineering": 145000,
      "Marketing": 125000,
      "Finance": 135000,
      "HR": 115000,
      "Operations": 105000
    },
    "average_performance_rating_by_department": {
      "Engineering": 4.1,
      "Marketing": 3.9,
      "Finance": 4.0,
      "HR": 4.2,
      "Operations": 3.8
    }
  },
  "correlation_id": "c9d0e1f2"
}
```

**Error (403):** Returned if `HR_AGGREGATE_MODE=disabled` and user is not HR/C-level.

---

## Admin Endpoints

All admin endpoints require C-level role.

### GET /admin/status

Returns system status including index counts, OpenAI health, and cache statistics.

**Response:**

```json
{
  "sparse_count": 203,
  "dense_count": 203,
  "persist_dir": ".chroma",
  "openai": {
    "ok": true,
    "model": "text-embedding-3-small",
    "status": 200
  },
  "cache": {
    "hits": 150,
    "misses": 30,
    "size": 180
  },
  "rag_cache": {
    "hits": 500,
    "misses": 100,
    "hit_rate": 0.83,
    "entries": 600,
    "singleflight_dedup": 45
  },
  "correlation_id": "d0e1f2a3"
}
```

---

### POST /admin/reindex

Rebuilds the in-memory BM25 sparse index from `resources/data/`.

**Response:**

```json
{
  "status": "ok",
  "count": 203
}
```

---

### POST /admin/reindex_dense

Rebuilds the ChromaDB dense vector index.

**Request Body (optional):**

```json
{
  "base_dir": "resources/data",
  "persist_dir": ".chroma"
}
```

**Response:**

```json
{
  "status": "ok",
  "count": 203,
  "persist_dir": ".chroma"
}
```

---

### POST /admin/cache/clear

Clears the legacy LLM and embedding cache.

**Response:**

```json
{
  "status": "ok",
  "cleared": {
    "hits": 150,
    "misses": 30,
    "size": 180
  },
  "correlation_id": "e1f2a3b4"
}
```

---

### POST /admin/rag_cache/clear

Clears the policy-safe RAG cache (all layers: RET0, ALLOWSET, DECISION, RR, LLM).

**Response:**

```json
{
  "status": "ok",
  "cleared": {
    "hits": 500,
    "misses": 100,
    "entries": 600
  },
  "correlation_id": "f2a3b4c5"
}
```

---

### GET /admin/rag_cache/stats

Returns detailed RAG cache statistics.

**Response:**

```json
{
  "stats": {
    "hits": 500,
    "misses": 100,
    "hit_rate": 0.83,
    "entries": 600,
    "singleflight_dedup": 45,
    "memory_estimate_mb": 12.5
  },
  "correlation_id": "a3b4c5d6"
}
```

---

## Health Endpoints

### GET /health

Basic health check. Does not require authentication.

**Response:**

```json
{
  "status": "ok"
}
```

---

### GET /health/openai

Validates LLM API connectivity by checking model metadata.

**Response (success):**

```json
{
  "ok": true,
  "model": "text-embedding-3-small",
  "status": 200
}
```

**Response (failure):**

```json
{
  "ok": false,
  "error": "Authentication failed"
}
```

---

### POST /authz/check

Test a policy decision without performing any retrieval. Useful for debugging access control rules.

**Request Body:**

```json
{
  "resource": {
    "owner_dept": "hr",
    "doc_type": "csv",
    "sensitivity": "restricted"
  },
  "action": "view_row"
}
```

**Response:**

```json
{
  "decision": "permit",
  "rule": "hr_rows_hr_and_c_level_only"
}
```

---

## UI Endpoints

### GET /

Serves the chat web interface. Renders `templates/chat.html` with the authenticated user's role and username. C-level users see an admin panel with reindex and status controls.

### GET /login

Login page for browser-based authentication.

---

## Common Response Headers

All responses include:

| Header | Description |
|--------|-------------|
| `X-Correlation-ID` | Unique request identifier for tracing in audit logs |
| `X-Frame-Options` | DENY (prevents clickjacking) |
| `X-Content-Type-Options` | nosniff (prevents MIME sniffing) |
| `X-XSS-Protection` | 1; mode=block |
| `Content-Security-Policy` | Restrictive CSP |
| `Referrer-Policy` | strict-origin-when-cross-origin |

---

## Error Responses

### 401 Unauthorized

Missing or invalid credentials.

```json
{
  "detail": "Invalid credentials"
}
```

### 403 Forbidden

Valid credentials but insufficient permissions.

```json
{
  "detail": "Only c_level can access this endpoint"
}
```

Or for policy-denied access:

```json
{
  "detail": "Access denied by policy: cross_department_deny"
}
```

### 422 Validation Error

Invalid request body (Pydantic validation failure).

```json
{
  "detail": [
    {
      "loc": ["body", "message"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

---

## Rate Limits

Requests are rate-limited per user:

| Endpoint Group | Limit |
|---------------|-------|
| Default | 100 requests/minute |
| Chat | 20 requests/minute |
| Search | 60 requests/minute |
| Admin | 10 requests/minute |

Rate limit headers are included in responses:

```
X-RateLimit-Limit: 20
X-RateLimit-Remaining: 19
X-RateLimit-Reset: 1706400060
```

When the limit is exceeded, the server returns `429 Too Many Requests`.
