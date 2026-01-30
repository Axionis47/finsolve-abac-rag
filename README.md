# FinSolve ABAC RAG System

A production-grade Retrieval-Augmented Generation chatbot that answers questions from company documents while enforcing strict, policy-driven access control at every stage of the pipeline. Every piece of context that reaches the LLM has been verified against a declarative ABAC policy engine.

Built with FastAPI, ChromaDB, and pluggable LLM/embedding backends (Anthropic Claude, OpenAI, Vertex AI, vLLM, Ollama).

**437 tests passing** | **315 dense vectors indexed** | **5 roles, 9 ABAC rules** | **391x-1400x cache speedup** | **3-layer access enforcement** | **Zero cross-role data leakage**

---

## Table of Contents

- [See It In Action](#see-it-in-action)
- [The Problem This Solves](#the-problem-this-solves)
- [System Architecture](#system-architecture)
- [Key Technical Decisions](#key-technical-decisions)
- [Technical Deep Dives](#technical-deep-dives)
- [Codebase Navigation](#codebase-navigation)
- [Quick Start](#quick-start)
- [Deployment](#deployment)
- [Documentation Index](#documentation-index)
- [Tech Stack](#tech-stack)

---

## See It In Action

The best way to understand this system is to see it enforce access policies in real-time. Every example below uses actual responses from the running server.

### Example 1: Same Query, Two Users, Different Results

**Sam (Finance)** asks about Q4 revenue and gets the answer:

```bash
curl -X POST http://localhost:8000/chat \
  -u sam:financepass \
  -H "Content-Type: application/json" \
  -d '{"message": "What was the Q4 revenue?", "top_k": 5}'
```

```json
{
  "answer": "Based on the provided context, the Q4 revenue was $2.6 billion, representing a 35% year-over-year increase [1]. This strong performance was driven by end-of-year promotions and B2B marketing initiatives, as well as holiday campaigns and enterprise client acquisitions [1].",
  "citations": [
    {
      "source_path": "resources/data/finance/quarterly_financial_report.md",
      "section_path": "Quarterly Financial Report > Q4 - October to December 2024 > Quarterly Financial Overview"
    }
  ],
  "metrics": {
    "retrieve_ms": 2880.1,
    "llm_ms": 4412.2,
    "total_ms": 7292.3
  }
}
```

**Tony (Engineering)** asks the exact same question and gets nothing:

```bash
curl -X POST http://localhost:8000/chat \
  -u tony:password123 \
  -H "Content-Type: application/json" \
  -d '{"message": "What was the Q4 revenue?", "top_k": 5}'
```

```json
{
  "answer": "I don't know based on the available context. The provided context snippets contain information about employee benefits, engineering roadmaps, maintenance procedures, and performance indicators, but do not include any financial revenue data for Q4 or any other quarter.",
  "citations": [
    {
      "source_path": "resources/data/general/employee_handbook.md",
      "section_path": "Employee Handbook > Employee Onboarding & Benefits > Statutory Benefits"
    },
    {
      "source_path": "resources/data/engineering/engineering_master_doc.md",
      "section_path": "FinSolve Technologies Engineering Document > 9. Future Roadmap"
    }
  ],
  "metrics": {
    "retrieve_ms": 9.1,
    "llm_ms": 3375.0,
    "total_ms": 3384.2
  }
}
```

**Why:** Finance documents are tagged `owner_dept: finance`. The policy rule `cross_department_deny` blocks engineering users. The finance chunks never reach the LLM. Tony's retrieval only finds engineering and general docs, so the LLM correctly says "I don't know."

---

### Example 2: Search Isolation Across Departments

**Bruce (Marketing)** searches for "Q2 campaign performance" via hybrid search:

```bash
curl -X POST http://localhost:8000/search/hybrid \
  -u bruce:securepass \
  -H "Content-Type: application/json" \
  -d '{"query": "Q2 campaign performance", "top_k": 3}'
```

```json
{
  "count": 3,
  "results": [
    {
      "text": "Email Retargeting\n- Campaign Performance: Recorded a 28% open rate and 4% conversion rate on retargeting emails, with a 12% uplift in conversions.",
      "source_path": "resources/data/marketing/marketing_report_q2_2024.md",
      "section_path": "Comprehensive Marketing Report - Q2 2024 > Campaign Highlights > Email Retargeting",
      "owner_dept": "marketing"
    },
    {
      "text": "Preliminary data indicates solid progress toward our Q2 targets:\n- Customer Acquisition: Acquired 190,000 new customers...\n- Revenue: Generated $5.8 million in revenue...\n- Conversion Rate: Achieved a 12.1% conversion rate...",
      "source_path": "resources/data/marketing/marketing_report_q2_2024.md",
      "section_path": "Comprehensive Marketing Report - Q2 2024 > Performance Analysis",
      "owner_dept": "marketing"
    },
    {
      "text": "Recommendations for Q2 2024\nBased on Q1 performance, we propose the following strategies for Q2...",
      "source_path": "resources/data/marketing/marketing_report_q1_2024.md",
      "section_path": "Comprehensive Marketing Report - Q1 2024 > Recommendations for Q2 2024",
      "owner_dept": "marketing"
    }
  ],
  "metrics": {
    "dense_ms": 152.8,
    "sparse_ms": 2.6,
    "fuse_ms": 0.01,
    "pdp_ms": 0.11,
    "total_ms": 155.6
  }
}
```

**Note:** All three results are from marketing docs. Zero finance documents appear, even though finance also has Q2 data (`quarterly_financial_report.md` has a Q2 section). Dense + sparse + RRF fusion all respect role boundaries.

---

### Example 3: HR Data — Progressive Disclosure

Three tiers of access to the same employee data:

**Full Access** (Natasha, HR role):

```bash
curl -s http://localhost:8000/hr/rows?limit=1 -u natasha:hrpass123
```

```json
{
  "rows": [{
    "employee_id": "FINEMP1000",
    "full_name": "Aadhya Patel",
    "role": "Sales Manager",
    "department": "Sales",
    "email": "aadhya.patel@fintechco.com",
    "salary": "1332478.37",
    "date_of_birth": "1991-04-03",
    "performance_rating": "3"
  }]
}
```

**Masked Access** (PII redacted):

```bash
curl -s http://localhost:8000/hr/rows_masked?limit=1 -u natasha:hrpass123
```

```json
{
  "rows": [{
    "employee_id": "[REDACTED]",
    "full_name": "[REDACTED]",
    "role": "Sales Manager",
    "department": "Sales",
    "email": "[REDACTED]",
    "salary": "1332478.37",
    "date_of_birth": "[REDACTED]",
    "performance_rating": "3"
  }]
}
```

**Denied** (Tony, Engineering role):

```bash
curl -s http://localhost:8000/hr/rows -u tony:password123
```

```json
{
  "detail": "Access denied"
}
```

No rule names, no policy details, no indication of whether the endpoint exists. Generic denial only.

---

### Example 4: Admin Panel — Role-Gated System Status

**Clark (C-Level)** can see system internals:

```bash
curl -s http://localhost:8000/admin/status -u clark:chief
```

```json
{
  "sparse_count": 374,
  "dense_count": 315,
  "persist_dir": ".chroma",
  "rag_cache": {
    "enabled": true,
    "adapter": {
      "type": "memory",
      "entries": 6,
      "hits": 9,
      "misses": 6,
      "hit_rate_pct": 60.0,
      "evictions": 0
    },
    "singleflight": {
      "dedup_count": 0,
      "timeout_count": 0,
      "in_flight": 0
    }
  }
}
```

**Tony (Engineering)** gets nothing:

```bash
curl -s http://localhost:8000/admin/status -u tony:password123
```

```json
{
  "detail": "Access denied"
}
```

---

## The Problem This Solves

Most RAG systems treat all documents the same. You embed everything, retrieve the top-k matches, and send them to the LLM. This works for personal notes or public documentation.

But in a real company:
- The finance team should not see HR salary data
- A marketing intern should not read the engineering architecture docs
- HR records contain PII that must be redacted for non-HR users
- Compliance requires audit trails of who accessed what

**Traditional approaches are inadequate:**

| Approach | Problem |
|----------|---------|
| Post-retrieval filtering | Wastes compute retrieving forbidden docs, risks leakage if filter fails |
| Separate indexes per role | Explodes storage (N indexes x M roles), hard to maintain |
| LLM-based filtering | Unreliable, expensive, not auditable |
| Network segmentation | Coarse-grained, doesn't handle same-doc different-section access |

**This project's approach:** Encode access control directly into the indexing and retrieval pipeline:

1. **Index time:** Compute `allowed_roles` per chunk, store as boolean flags in ChromaDB
2. **Retrieval time:** Server-side `where` filters in ChromaDB + BM25 pre-filter by role
3. **Post-retrieval:** Full PDP evaluation per candidate with decision caching
4. **Audit:** Correlation IDs, JSONL logs, policy rule traces on every request

**Result:** Fast, provable, and auditable. Zero data leakage across 437 automated tests.

---

## System Architecture

```
                          +------------------+
                          |   Browser / API  |
                          |   (HTTP Basic)   |
                          +--------+---------+
                                   |
                          +--------v---------+
                          |    FastAPI App    |
                          | - Auth Layer     |
                          | - Rate Limiting  |
                          | - Correlation ID |
                          | - Security Hdrs  |
                          +--------+---------+
                                   |
              +--------------------+--------------------+
              |                    |                     |
     +--------v--------+  +-------v--------+  +--------v--------+
     |   /chat          |  |  /search/*     |  |    /hr/*        |
     |   /chat/stream   |  |  dense, hybrid |  |  rows, masked,  |
     |                  |  |  sparse        |  |  aggregates     |
     +--------+---------+  +-------+--------+  +--------+--------+
              |                    |                     |
              +--------------------+--------------------+
                                   |
                          +--------v---------+
                          |  Hybrid Retrieval |
                          |  Dense (Chroma)   |
                          |  Sparse (BM25)    |
                          |  RRF Fusion       |
                          +--------+---------+
                                   |
                          +--------v---------+
                          | PDP Policy Engine |
                          | (per-item check)  |
                          +--------+---------+
                                   |
                          +--------v---------+
                          | Policy-Safe Cache |
                          | (PF-scoped keys)  |
                          +--------+---------+
                                   |
                          +--------v---------+
                          |  LLM Generation   |
                          |  (Claude/GPT/etc) |
                          |  Citations + Audit|
                          +------------------+
```

### How a Query Flows Through the System

1. **Authenticate:** Verify credentials, resolve username to role
2. **Multi-hop check:** If query is complex ("compare Q1 vs Q3"), decompose into sub-queries with synonym expansion
3. **Dense search:** ChromaDB cosine similarity with `where={role_engineering: true}` server-side filter
4. **Sparse search:** BM25 (k1=1.5, b=0.75) over pre-filtered allowed-roles subset
5. **Fusion:** Reciprocal Rank Fusion (k=60) merges dense + sparse rankings without score normalization
6. **Policy check:** PDP evaluates each candidate against YAML rules (cached by Policy Fingerprint)
7. **Generate:** LLM builds answer from verified context only, with inline citations
8. **Audit:** Correlation ID, timings, cache hits, policy decisions logged to daily JSONL files

### Component Summary

| Component | Files | What It Does |
|-----------|-------|-------------|
| **Auth** | `app/routes/deps.py` | HTTP Basic auth, resolves role from user database |
| **Policy Engine** | `app/policy/pdp.py`, `docs/policy.yaml` | Evaluates ABAC rules: subject + resource + action = permit/deny |
| **Ingestion** | `app/ingest/` (9 modules) | Discovers docs, chunks by heading, computes access, builds indexes |
| **Retrieval** | `app/retrieval/` (6 modules) | BM25, ChromaDB vector search, RRF fusion, reranking |
| **Caching** | `app/cache/` (9 modules) | Policy-safe cache with fingerprints, singleflight, epoch invalidation |
| **Services** | `app/services/` (10 modules) | LLM generation, embeddings, multi-hop, reranking (pluggable backends) |
| **HR** | `app/hr/csv_service.py` | Row-level access, PII masking, aggregate computation |
| **Observability** | `app/utils/` (6 modules) | Audit logs, correlation IDs, rate limiting, circuit breaker, retry |

---

## Key Technical Decisions

| Problem | Decision | Why |
|---------|----------|-----|
| Documents have different owners and sensitivity levels | ABAC policy engine with YAML rules | Declarative, auditable, and separates policy from code |
| Dense search alone misses keyword matches | Hybrid retrieval (dense + sparse) with RRF fusion | Best of both worlds without normalization headaches |
| Same text in different departments should retrieve differently | Context-aware embeddings with security prefix | Embeddings encode who owns the data, not just what it says |
| HR data contains PII but aggregate stats are safe to share | Three-tier access: full, masked, aggregate | Progressive disclosure based on role |
| Caching must not leak data across roles | Policy Fingerprint scoped cache keys | SHA256 hash of role + policy version ensures isolation |
| Policy changes must instantly invalidate stale cache | Epoch-based invalidation | Bump a counter, all old keys become unreachable |
| Concurrent identical requests should not cause duplicate work | Singleflight pattern | One computation, all waiters share the result |
| Need to support air-gapped and cloud deployments | Pluggable LLM and embedding backends | Swap between OpenAI, Anthropic, vLLM, Ollama, Vertex AI without code changes |

---

## Technical Deep Dives

### ABAC Policy Engine

The system uses Attribute-Based Access Control, defined in `docs/policy.yaml`. Every access decision considers three things:

- **Subject:** Who is asking (role: finance, hr, engineering, marketing, c_level, employee)
- **Resource:** What they are asking for (owner_dept, doc_type, sensitivity)
- **Action:** What they want to do (retrieve_chunk, view_row, view_masked_row, aggregate_query)

**9 rules, evaluated top-down. First match wins. Default is deny.**

```yaml
rules:
  - name: c_level_full_access          # C-level sees everything
  - name: general_docs_any_role        # Employee handbook visible to all
  - name: same_department_internal_docs # Finance sees finance docs
  - name: employee_non_general_deny    # Employees only see general
  - name: cross_department_deny        # No cross-department access
  - name: hr_rows_hr_and_c_level_only  # Raw HR data is restricted
  - name: hr_aggregates_non_hr_if_enabled  # Flag-controlled aggregate access
  - name: high_sensitivity_md_deny     # Confidential = c_level only
```

**Access Matrix:**

| Resource | Finance | Marketing | HR | Engineering | C-Level |
|----------|---------|-----------|-----|-------------|---------|
| General docs | Yes | Yes | Yes | Yes | Yes |
| Own department docs | Yes | Yes | Yes | Yes | Yes |
| Other department docs | No | No | No | No | Yes |
| HR full rows (PII) | No | No | Yes | No | Yes |
| HR masked rows* | Yes* | Yes* | Yes | Yes* | Yes |
| HR aggregates* | Yes* | Yes* | Yes | Yes* | Yes |

*Requires runtime flag enabled (`hr_masked_rows_mode` or `hr_aggregate_mode`)

**Defense in Depth (3 Layers):**

1. **Index time:** `allowed_roles` computed per chunk, stored as boolean flags in ChromaDB
2. **Retrieval time:** ChromaDB `where` filter + BM25 pre-filter by role
3. **Post-retrieval:** Full PDP evaluation per candidate (catches edge cases, flag changes)

For the full policy specification, see [docs/POLICY_ENGINE.md](docs/POLICY_ENGINE.md).

---

### Hybrid Retrieval

**Problem:** Dense search captures semantics ("company revenue" matches "annual income") but misses exact keywords. Sparse search captures keywords ("Q4 2024") but misses meaning. Neither alone is sufficient.

**Solution:** Run both, fuse with Reciprocal Rank Fusion:

```
RRF_score(doc) = sum( 1 / (k + rank_in_source_i) )    where k = 60
```

No score normalization needed across different scales.

**Context-Aware Embeddings:** Two chunks with identical text but different owners should not have identical embeddings. The indexer prepends security context before embedding:

```
[DEPARTMENT: FINANCE] [SENSITIVITY: INTERNAL] [TYPE: MD] [QUARTER: Q2] [YEAR: 2024]
[DOCUMENT: quarterly financial report]

Revenue for Q2 2024 was $12.3M, up 8% year-over-year...
```

The original text is stored for display. The prefixed text is used only for embedding. This makes "Q2 revenue" match finance documents more strongly than marketing documents with the same words.

For more details, see [docs/INDEXING_STRATEGY.md](docs/INDEXING_STRATEGY.md).

---

### Policy-Safe Caching

**Core invariant:** Cached data must never leak across permission boundaries.

**Policy Fingerprint (PF):** Every cache key containing sensitive data includes a PF:

```
PF = sha256({role, tenant_id, policy_epoch, entitlement_epoch, flags})[:16]
```

If the user's role changes, or the policy changes, or a feature flag flips, the PF changes. Old cache entries become unreachable without needing to delete them.

**Cache Types:**

| Type | What It Caches | Scoped by PF? | Why |
|------|---------------|---------------|-----|
| **RET0** | Raw retrieval results | No | Content-neutral, re-filtered after cache hit |
| **DECISION** | Policy permit/deny per item | Yes | Contains authorization decisions |
| **LLM** | Generated answers | Yes | Answers contain role-filtered content |
| **ALLOWSET** | Set of allowed doc IDs | Yes | Role-specific permission set |
| **RR** | Reranking order | No | Just ordering, no content |

**Performance:**

| Cache Type | Miss Latency | Hit Latency | Speedup |
|------------|-------------|-------------|---------|
| Decision | 1.53ms | 0.004ms | 391x |
| LLM Answer | 56.5ms | 0.04ms | 1,400x |
| Retrieval | 26.5ms | 0.07ms | 380x |

**Epoch-Based Invalidation:** Bump `CACHE_POLICY_EPOCH=2` when policy.yaml changes. All PF hashes change, making old entries unreachable. No mass deletes needed.

**Singleflight Pattern:** 20 concurrent identical requests = 1 computation, 19 waiters. 95% deduplication rate under load.

For the full caching design, see [docs/CACHING.md](docs/CACHING.md).

---

### Indexing Strategy

**Heading-Aware Chunking:** Markdown headings create natural boundaries. Section breadcrumbs (`H1 > H2 > H3`) are preserved for citations. Fallback: recursive splitting into paragraphs, sentences, words when sections exceed the token limit.

**Temporal Metadata:** Regex extraction of quarter and year from filenames (`q([1-4])[-_]?(20\d{2})`) and content (`\b(Q[1-4])\b`).

**Pre-Computed Role Booleans:** Each chunk stored with `{role_finance: true, role_marketing: false, ...}` flags in ChromaDB, enabling server-side filtering during retrieval.

**HR Three-Tier Access:**
- **Full:** Complete row with salary, DOB, performance rating (HR + C-level only)
- **Masked:** `[REDACTED]` for PII fields (flag-controlled)
- **Aggregate:** Pre-computed stats like "Average salary in Engineering: $145k" (non-HR safe)

For the full indexing design, see [docs/INDEXING_STRATEGY.md](docs/INDEXING_STRATEGY.md).

---

## Codebase Navigation

**If you want to understand...**

- **How access control works:** Start with `app/policy/pdp.py` (150 lines, YAML rule evaluator) and `docs/policy.yaml` (9 rules)
- **How retrieval works:** `app/retrieval/hybrid.py` (RRF fusion), `app/retrieval/bm25.py` (full BM25 implementation)
- **How caching works:** `app/cache/rag_cache.py` (facade), `app/cache/policy_fingerprint.py` (PF computation)
- **How the API is structured:** `app/routes/chat.py` (main endpoint), `app/routes/deps.py` (auth + PDP)
- **How indexing works:** `app/ingest/runner.py` (orchestrator), `app/ingest/chunker.py` (heading-aware)

### Project Structure

```
ds-rpc-01/
|-- app/
|   |-- main.py                    # FastAPI app entry point, middleware, lifespan
|   |-- policy/
|   |   +-- pdp.py                 # ABAC Policy Decision Point engine
|   |-- ingest/
|   |   |-- discovery.py           # Scan resources/data/ for documents
|   |   |-- chunker.py             # Heading-aware markdown chunking
|   |   |-- metadata_extractor.py  # Temporal, topic, sensitivity extraction
|   |   |-- allow_roles.py         # Compute chunk-level access tiers
|   |   |-- dense_indexer.py       # ChromaDB indexing with context embeddings
|   |   |-- hr_csv_indexer.py      # HR CSV row-level indexing with PII separation
|   |   |-- hr_aggregates.py       # Pre-computed aggregate statistics
|   |   +-- runner.py              # BM25 sparse index builder
|   |-- retrieval/
|   |   |-- bm25.py                # BM25 implementation with TF-IDF
|   |   |-- hybrid.py              # Reciprocal Rank Fusion (RRF)
|   |   |-- filtering.py           # Role-based pre-filtering
|   |   |-- rerank.py              # Pluggable reranking (simple, openai, llm)
|   |   +-- search.py              # Search orchestration
|   |-- services/
|   |   |-- providers.py           # Backend factory (LLM + embedding)
|   |   |-- llm_backend.py         # OpenAI, Anthropic, vLLM, Ollama, Vertex
|   |   |-- embed_backend.py       # OpenAI, Local, TEI, Vertex embeddings
|   |   |-- generation.py          # LLM answer synthesis with citations
|   |   |-- multihop.py            # Multi-hop query decomposition
|   |   |-- reranker_openai.py     # Embedding and LLM-based reranking
|   |   +-- cache.py               # Persistent disk cache (SQLite/diskcache)
|   |-- cache/
|   |   |-- rag_cache.py           # High-level RAGCache facade
|   |   |-- policy_fingerprint.py  # PF computation and epoch tracking
|   |   |-- keys.py                # Cache key generation and normalization
|   |   |-- singleflight.py        # Concurrent request deduplication
|   |   |-- invalidation.py        # Epoch-based distributed invalidation
|   |   |-- metrics.py             # Prometheus cache metrics
|   |   |-- base.py                # InMemory, Redis, NoOp adapters
|   |   +-- redis_adapter.py       # Redis with circuit breaker and compression
|   |-- routes/
|   |   |-- chat.py                # /chat and /chat/stream endpoints
|   |   |-- search.py              # /search, /search/dense, /search/hybrid
|   |   |-- admin.py               # /admin/* (c_level only)
|   |   |-- hr.py                  # /hr/rows, /hr/rows_masked, /hr/aggregate
|   |   |-- auth.py                # /login
|   |   |-- metrics.py             # /metrics (Prometheus)
|   |   |-- evaluation.py          # /evaluation (RAGAS)
|   |   +-- deps.py                # FastAPI dependencies (auth, PDP, index)
|   |-- schemas/
|   |   +-- validators.py          # Pydantic request/response models
|   |-- hr/
|   |   +-- csv_service.py         # HR data access, masking, aggregates
|   |-- evaluation/
|   |   |-- ragas_evaluator.py     # RAGAS framework integration
|   |   +-- policy_aware_eval.py   # Policy-aware RAG evaluation
|   +-- utils/
|       |-- config.py              # Environment and settings management
|       |-- audit.py               # JSON audit logging with daily rotation
|       |-- retry.py               # Exponential backoff retry decorator
|       |-- circuit_breaker.py     # Circuit breaker pattern
|       +-- rate_limit.py          # slowapi rate limiting
|-- docs/
|   |-- policy.yaml                # ABAC policy definition (9 rules)
|   |-- metadata_schema.yaml       # Chunk metadata schema
|   |-- ARCHITECTURE.md            # System architecture documentation
|   |-- CACHING.md                 # Caching layer design document
|   |-- INDEXING_STRATEGY.md       # Indexing decisions and thought process
|   |-- POLICY_ENGINE.md           # Policy engine deep dive
|   |-- API_REFERENCE.md           # Complete API documentation
|   +-- SELF_HOSTED_SETUP.md       # vLLM/Ollama self-hosted guide
|-- resources/data/                # Knowledge base (10 documents)
|   |-- engineering/               # Engineering docs
|   |-- finance/                   # Financial reports
|   |-- marketing/                 # Marketing reports (Q1-Q4 2024)
|   |-- hr/                        # HR employee CSV (PII)
|   +-- general/                   # Employee handbook (public)
|-- tests/                         # 35 test files (unit + integration + e2e)
|-- scripts/                       # Benchmarks, load tests, evaluation
|-- k8s/                           # Kubernetes manifests (9 files)
|-- static/                        # Frontend JS + CSS
|-- templates/                     # Chat UI template
|-- Dockerfile                     # Python 3.12, Cloud Run compatible
|-- docker-compose.yml             # Profiles: default, local (Ollama), gpu (vLLM)
|-- pyproject.toml                 # Dependencies and project config
+-- .github/workflows/             # CI/CD (tests + Cloud Run deploy)
```

### Test Coverage (35 files, 437 tests)

| Category | Key Files | What They Test |
|----------|-----------|---------------|
| **Policy** | `test_pdp.py`, `test_pdp_edge_cases.py`, `test_app_authz.py` | All 9 rules, edge cases, cross-role isolation |
| **Ingestion** | `test_ingest.py`, `test_chunker.py`, `test_ingest_runner.py` | Discovery, heading-aware chunking, BM25 building |
| **Retrieval** | `test_search_api.py`, `test_dense_search_api.py`, `test_retrieval_edge_cases.py` | All 3 search modes, empty inputs, degenerate cases |
| **Cache** | `test_cache_layer.py`, `test_cache_edge_cases.py`, `test_cache_integration.py` | PF isolation, singleflight, epoch invalidation |
| **HR** | `test_hr_csv_service.py`, `test_hr_edge_cases.py`, `test_hr_masked_rows.py` | Row access, PII masking, aggregates, edge cases |
| **Chat** | `test_chat_api.py`, `test_multihop.py`, `test_generation_edge_cases.py` | Answer generation, multi-hop, LLM failure paths |
| **Security** | `test_response_sanitization.py`, `test_cross_role_security.py`, `test_auth_edge_cases.py` | No rule leaks in 403s, parametrized cross-role tests |
| **Resilience** | `test_circuit_breaker.py`, `test_retry.py`, `test_utils_rate_limit.py` | Failure handling, backoff, rate limits |
| **E2E** | `test_e2e_samples.py` | Full query-to-answer flow with policy |

All tests run offline. LLM and ChromaDB are monkeypatched. No network required.

---

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url>
cd ds-rpc-01
python -m venv .venv
source .venv/bin/activate
pip install -e .

# 2. Configure backend (pick one)
export LLM_BACKEND=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
# OR
export LLM_BACKEND=openai
export OPENAI_API_KEY=sk-...

# 3. Start the server
uvicorn app.main:app --reload

# 4. Open http://127.0.0.1:8000/
# Login as Clark/chief (c_level admin)
# Click "Rebuild Vector Index" in admin panel
# Then ask questions as any user
```

### Demo Users

| Username | Password | Role | What They Can Access |
|----------|----------|------|---------------------|
| clark | chief | c_level | Everything + admin panel |
| natasha | hrpass123 | hr | HR data + HR docs + general |
| sam | financepass | finance | Finance docs + general |
| bruce | securepass | marketing | Marketing docs + general |
| tony | password123 | engineering | Engineering docs + general |
| peter | pete123 | engineering | Engineering docs + general |
| sid | sidpass123 | marketing | Marketing docs + general |

### Try These Queries

1. Login as **sam** (finance), ask: *"What was Q4 revenue?"* — Gets $2.6B answer with citations
2. Login as **tony** (engineering), ask the same — Gets "I don't know" (policy denial)
3. Login as **bruce** (marketing), ask: *"What was the Q2 campaign ROI?"* — Marketing-only results
4. Login as **clark** (c_level), open admin panel — See cache stats, index counts, reindex controls

---

## API Reference

All endpoints require HTTP Basic authentication.

### Chat

| Method | Path | Description |
|--------|------|-------------|
| POST | `/chat` | Ask a question, get LLM answer with citations |
| POST | `/chat/stream` | Same as /chat but with Server-Sent Events streaming |

Request body:
```json
{
  "message": "What is the Q4 marketing budget?",
  "top_k": 5,
  "rerank": true,
  "reranker": "openai",
  "multihop": false
}
```

Response includes: `answer`, `citations[]`, `metrics{}`, `correlation_id`

### Search

| Method | Path | Description |
|--------|------|-------------|
| POST | `/search` | Sparse keyword search (BM25 only) |
| POST | `/search/dense` | Dense vector search (ChromaDB only) |
| POST | `/search/hybrid` | Dense + sparse with RRF fusion |

### HR Data

| Method | Path | Description |
|--------|------|-------------|
| GET | `/hr/rows` | Full HR records (HR and C-level only) |
| GET | `/hr/rows_masked` | PII-redacted records (flag-controlled) |
| GET | `/hr/aggregate` | Safe aggregate statistics (flag-controlled) |

### Admin (C-level only)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/admin/status` | Index counts, LLM health, cache stats |
| POST | `/admin/reindex` | Rebuild sparse BM25 index |
| POST | `/admin/reindex_dense` | Rebuild dense ChromaDB index |
| POST | `/admin/cache/clear` | Clear legacy cache |
| POST | `/admin/rag_cache/clear` | Clear policy-safe RAG cache |
| GET | `/admin/rag_cache/stats` | Detailed RAG cache statistics |

### Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Basic health check (no auth) |
| GET | `/health/ready` | Kubernetes readiness probe |
| GET | `/health/live` | Kubernetes liveness probe |
| POST | `/authz/check` | Test policy decision for subject/resource/action |

For full request/response schemas, see [docs/API_REFERENCE.md](docs/API_REFERENCE.md).

---

## Configuration

Set via environment variables or `.env` file.

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BACKEND` | openai | Backend: openai, anthropic, vllm, ollama, vertex |
| `EMBEDDING_BACKEND` | openai | Backend: openai, local, tei, vertex |
| `HR_AGGREGATE_MODE` | disabled | Enable aggregate HR stats for non-HR roles |
| `HR_MASKED_ROWS_MODE` | disabled | Enable PII-masked HR rows for non-HR roles |
| `CACHE_ENABLED` | true | Master switch for all caching |
| `CACHE_BACKEND` | memory | memory or redis |
| `CACHE_POLICY_EPOCH` | 1 | Bump when policy.yaml changes |
| `CACHE_ENTITLEMENT_EPOCH` | 1 | Bump when user roles change |
| `CHROMA_DB_DIR` | .chroma | ChromaDB persistence directory |

---

## Deployment

### Docker Compose

```bash
# Default (OpenAI/Anthropic backend)
docker compose up

# Self-hosted with Ollama (no API key needed)
docker compose --profile local up

# GPU with vLLM + Text Embeddings Inference
docker compose --profile gpu up
```

### Kubernetes

Full K8s manifests in `k8s/`: Deployment with HPA, Redis StatefulSet, ConfigMap/Secrets, Ingress, PVC.

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/
```

### Google Cloud Run

CI/CD pipeline in `.github/workflows/deploy-cloudrun.yml` handles automated deployment on push to main.

### Observability

- **Metrics:** Prometheus at `/metrics` (cache hit rate, latency histograms, policy decisions)
- **Audit Logs:** Daily JSONL rotation in `logs/audit/` with correlation IDs
- **Health Probes:** `/health/ready` and `/health/live` for Kubernetes
- **Security Headers:** X-Frame-Options: DENY, CSP, X-Content-Type-Options: nosniff
- **Rate Limiting:** 20 req/min chat, 60 req/min search, 10 req/min admin

For self-hosted GPU setup (vLLM on L4), see [docs/SELF_HOSTED_SETUP.md](docs/SELF_HOSTED_SETUP.md).

---

## Documentation Index

| Document | Description |
|----------|-------------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture with Mermaid diagram |
| [docs/INDEXING_STRATEGY.md](docs/INDEXING_STRATEGY.md) | Deep dive on indexing decisions and thought process |
| [docs/POLICY_ENGINE.md](docs/POLICY_ENGINE.md) | ABAC policy engine specification |
| [docs/CACHING.md](docs/CACHING.md) | Policy-safe caching layer design |
| [docs/API_REFERENCE.md](docs/API_REFERENCE.md) | Complete API endpoint documentation |
| [docs/SELF_HOSTED_SETUP.md](docs/SELF_HOSTED_SETUP.md) | Self-hosted LLM setup (vLLM, Ollama) |
| [docs/policy.yaml](docs/policy.yaml) | ABAC policy definition |
| [docs/metadata_schema.yaml](docs/metadata_schema.yaml) | Chunk metadata schema |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Framework** | FastAPI (Python 3.12) |
| **Vector Store** | ChromaDB (persistent, cosine similarity) |
| **LLM** | Anthropic Claude, OpenAI GPT, vLLM, Ollama, Vertex AI Gemini |
| **Embeddings** | OpenAI, Vertex AI, Sentence Transformers (local), TEI |
| **Cache** | In-memory LRU (default), Redis (production) |
| **Auth** | HTTP Basic (demo), designed for OAuth2/OIDC upgrade |
| **Observability** | Prometheus metrics, JSONL audit logs, correlation IDs |
| **Testing** | pytest (35 files, 437 tests) |
| **Deployment** | Docker, Kubernetes, Google Cloud Run |
| **CI/CD** | GitHub Actions (tests + deploy) |
