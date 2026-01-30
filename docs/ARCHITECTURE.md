# System Architecture

## Overview

FinSolve is an internal chatbot that answers department-specific questions using a policy-driven Retrieval-Augmented Generation (RAG) pipeline. Access is governed by an Attribute-Based Access Control (ABAC) policy enforced by a Policy Decision Point (PDP). The system combines dense retrieval (ChromaDB with vector embeddings) and sparse retrieval (BM25), with optional reranking and LLM synthesis that cites sources.

**Current State:**
- Dense (ChromaDB) chunks: 203+ (persisted under `.chroma`)
- Sparse (BM25) chunks: 203+ (in-memory)
- Tests: 31 passing
- LLM backends: Anthropic Claude, OpenAI GPT, vLLM, Ollama, Vertex AI Gemini
- Embedding backends: OpenAI, Vertex AI, Sentence Transformers (local), TEI

---

## Architecture Diagram

```mermaid
flowchart TD
  subgraph Client
    U[Browser UI /]
  end

  subgraph FastAPI Backend
    A[HTTP Basic Auth<br/>users_db]
    P[PDP / ABAC<br/>policy.yaml]
    R[Retrieval Orchestrator<br/>/search/*, /chat]
    MH[Multi-Hop Decomposition<br/>complex query splitting]
    RR[Reranking<br/>simple | openai | openai_llm]
    F[Fusion RRF k=60]
    G[Generation<br/>Claude / GPT / vLLM / Ollama<br/>citations + grounded answers]
    H[HR CSV Service<br/>rows / aggregates / masked]
    C[Policy-Safe Cache<br/>PF-scoped keys<br/>singleflight + epochs]
    O[Observability<br/>metrics + X-Correlation-ID]
    L[Audit Logger<br/>JSONL daily rotation]
    RL[Rate Limiter<br/>slowapi per-user]
    CB[Circuit Breaker<br/>failure protection]
  end

  subgraph Data and Services
    D1[resources/data<br/>Markdown + HR CSV]
    D2[BM25 In-Memory Index]
    D3[ChromaDB .chroma<br/>Collection kb_main]
    S1[Embedding Backend<br/>OpenAI / Vertex / Local / TEI]
    S2[LLM Backend<br/>Claude / GPT / vLLM / Ollama]
    LG[logs/audit/*.jsonl]
  end

  U -->|HTTP Basic| A
  U -->|/chat, /search| R
  A --> R
  R -->|discover + chunk| D1
  D1 -->|build| D2
  D1 -->|index with context embeddings| S1
  S1 --> D3

  R -->|complex queries| MH
  MH -->|sub-queries| R
  R -->|dense| D3
  R -->|sparse| D2
  R --> F --> RR --> R
  R -->|PDP checks with caching| P
  P --> C
  R -->|snippets| G
  G -->|response with citations| U

  R --> O
  R --> L --> LG
  RL --> R
  CB --> S1
  CB --> S2

  U -->|/hr/*| H
  H --> P
  H --> L

  classDef store fill:#eef,stroke:#88a,stroke-width:1px
  class D1,D2,D3,LG store
```

---

## Component Breakdown

### Authentication and Authorization

| Component | File | Description |
|-----------|------|-------------|
| Auth | `app/routes/deps.py` | HTTP Basic auth, resolves username to role |
| PDP | `app/policy/pdp.py` | ABAC policy engine, evaluates YAML rules |
| Policy | `docs/policy.yaml` | 9 rules covering 6 roles, 5 actions, 3 doc types |

### Ingestion Pipeline

| Component | File | Description |
|-----------|------|-------------|
| Discovery | `app/ingest/discovery.py` | Scans resources/data/{dept}/ for .md and .csv files |
| Chunker | `app/ingest/chunker.py` | Heading-aware markdown chunking with section breadcrumbs |
| Metadata | `app/ingest/metadata_extractor.py` | Extracts year, quarter, topics, sensitivity indicators |
| Access Control | `app/ingest/allow_roles.py` | Computes 3-tier access per chunk: full, masked, aggregate |
| Dense Indexer | `app/ingest/dense_indexer.py` | Context-aware embeddings, ChromaDB upsert |
| HR CSV Indexer | `app/ingest/hr_csv_indexer.py` | Row-level indexing with PII separation |
| HR Aggregates | `app/ingest/hr_aggregates.py` | Pre-computed stats (headcount, salary, location) |
| Sparse Indexer | `app/ingest/runner.py` | BM25 in-memory index builder |

### Retrieval Pipeline

| Component | File | Description |
|-----------|------|-------------|
| BM25 | `app/retrieval/bm25.py` | Full BM25 implementation (k1=1.5, b=0.75) |
| Hybrid Fusion | `app/retrieval/hybrid.py` | Reciprocal Rank Fusion (RRF, k=60) |
| Pre-filtering | `app/retrieval/filtering.py` | Role-based allowed_roles filter |
| Reranking | `app/retrieval/rerank.py` | Pluggable: simple BM25, OpenAI embedding, LLM |
| Multi-hop | `app/services/multihop.py` | LLM-driven query decomposition with synonym expansion |

### LLM and Embedding Services

| Component | File | Description |
|-----------|------|-------------|
| Provider Factory | `app/services/providers.py` | Singleton factory for LLM and embedding backends |
| LLM Backends | `app/services/llm_backend.py` | OpenAI, Anthropic, vLLM, Ollama, Vertex AI |
| Embedding Backends | `app/services/embed_backend.py` | OpenAI, Local (Sentence Transformers), TEI, Vertex AI |
| Generation | `app/services/generation.py` | Builds context prompt, generates cited answers |
| Reranker | `app/services/reranker_openai.py` | Embedding cosine and LLM-based reranking |

### Caching Layer

| Component | File | Description |
|-----------|------|-------------|
| RAG Cache | `app/cache/rag_cache.py` | High-level facade for all cache operations |
| Policy Fingerprint | `app/cache/policy_fingerprint.py` | SHA256 hash of role + epochs + flags |
| Key Generation | `app/cache/keys.py` | Normalized, deterministic cache keys |
| Singleflight | `app/cache/singleflight.py` | Prevents duplicate computation for identical requests |
| Invalidation | `app/cache/invalidation.py` | Epoch-based, Redis pub/sub for distributed |
| Metrics | `app/cache/metrics.py` | Prometheus counters, histograms, gauges |
| Adapters | `app/cache/base.py` | InMemory (LRU, TTL), NoOp |
| Redis | `app/cache/redis_adapter.py` | Redis with circuit breaker and compression |

### Observability

| Component | File | Description |
|-----------|------|-------------|
| Audit Logger | `app/utils/audit.py` | JSONL logs with daily rotation, correlation IDs |
| Rate Limiter | `app/utils/rate_limit.py` | Per-user rate limiting with slowapi |
| Circuit Breaker | `app/utils/circuit_breaker.py` | Failure detection and recovery for external services |
| Retry | `app/utils/retry.py` | Exponential backoff with jitter |
| Config | `app/utils/config.py` | Environment variable management |

### HR Data Service

| Component | File | Description |
|-----------|------|-------------|
| CSV Service | `app/hr/csv_service.py` | Row access, PII masking, aggregate computation |

### API Routes

| Route Group | File | Endpoints |
|------------|------|-----------|
| Chat | `app/routes/chat.py` | POST /chat, POST /chat/stream |
| Search | `app/routes/search.py` | POST /search, /search/dense, /search/hybrid |
| Admin | `app/routes/admin.py` | GET /admin/status, POST /admin/reindex, /admin/reindex_dense |
| HR | `app/routes/hr.py` | GET /hr/rows, /hr/rows_masked, /hr/aggregate |
| Auth | `app/routes/auth.py` | GET /login |
| Metrics | `app/routes/metrics.py` | GET /metrics |
| Evaluation | `app/routes/evaluation.py` | RAGAS evaluation endpoints |

---

## Data Flow: Chat Request

```
1. User sends POST /chat with message and credentials
2. FastAPI authenticates user, resolves role
3. (Optional) Multi-hop: LLM decomposes complex query into sub-queries
4. For each (sub-)query:
   a. Embed query using configured embedding backend
   b. Dense search: ChromaDB with where={role_X: true} filter
   c. Sparse search: BM25 over pre-filtered allowed-roles subset
   d. RRF fusion merges dense and sparse rankings
   e. (Optional) Rerank using simple/openai/llm backend
5. PDP post-check per candidate (with decision caching via Policy Fingerprint)
6. Build context from permitted snippets
7. LLM generates answer with inline citations (with LLM answer caching via PF)
8. Return answer, citations, metrics, correlation ID
9. Audit log written with all decisions and timings
```

---

## Data Flow: Indexing

```
1. discovery.py scans resources/data/{dept}/ for files
2. For each markdown file:
   a. chunker.py splits by headings, builds section breadcrumbs
   b. metadata_extractor.py extracts year, quarter, topics
   c. allow_roles.py computes 3-tier access (full/masked/aggregate)
   d. dense_indexer.py prepends security context, generates embeddings
   e. Upsert to ChromaDB with role boolean flags and rich metadata
3. For each HR CSV file:
   a. hr_csv_indexer.py creates searchable text (non-PII) and masked text
   b. hr_aggregates.py generates aggregate statistics chunks
   c. Both indexed with appropriate access tiers
4. runner.py builds BM25 in-memory index from all markdown chunks
```

---

## Deployment Architecture

```
Production:
  Docker / Kubernetes / Cloud Run
  +-- FastAPI App (port 8000)
  +-- ChromaDB (persistent volume)
  +-- Redis (optional, for distributed caching)
  +-- Prometheus (optional, for metrics)

Self-Hosted:
  L4 GPU (24GB VRAM)
  +-- vLLM Server (Llama 3.1 8B AWQ, port 8000)
  +-- TEI Server (embedding, port 8080)
  +-- FastAPI App (port 8080)

Local Development:
  +-- FastAPI App (port 8000)
  +-- Ollama (local LLM, port 11434)
  +-- Sentence Transformers (local embedding, in-process)
```

---

## Security Model

1. **Authentication:** HTTP Basic (demo). Production upgrade path: OAuth2/OIDC.
2. **Authorization:** ABAC with YAML policy. 9 rules, 6 roles, fail-secure default deny.
3. **Defense in Depth:** 3-layer access control (index, retrieval, post-retrieval).
4. **Cache Safety:** Policy Fingerprint scoped keys prevent cross-role leakage.
5. **PII Protection:** HR data separated into full/masked/aggregate tiers.
6. **Security Headers:** X-Frame-Options, X-Content-Type-Options, CSP, Referrer-Policy.
7. **Rate Limiting:** Per-user limits on all endpoint groups.
8. **Audit Trail:** JSONL logs with correlation IDs, daily rotation.
