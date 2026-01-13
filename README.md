# FinSolve ABAC RAG Chatbot

An internal chatbot that answers questions from company documents with role-based access control. Built with FastAPI, uses ABAC (Attribute-Based Access Control) policies, and implements a hybrid RAG (Retrieval-Augmented Generation) pipeline.

The system ensures users only see documents they are permitted to access. Every piece of context passed to the LLM goes through policy checks first.

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Policy System](#policy-system)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Testing](#testing)
- [Demo Users](#demo-users)
- [Troubleshooting](#troubleshooting)

---

## Features

**Retrieval Pipeline**
- Hybrid search combining dense (vector) and sparse (BM25) retrieval
- Dense: ChromaDB with OpenAI embeddings (`text-embedding-3-small`)
- Sparse: In-memory BM25 index with TF-IDF scoring
- Fusion via Reciprocal Rank Fusion (RRF) with k=60
- Optional reranking: simple score-based, OpenAI embedding similarity, or LLM-based

**Access Control**
- ABAC policy engine with YAML-based rules
- Pre-filtering at retrieval time using computed `allowed_roles`
- Post-filtering via Policy Decision Point (PDP) for defense-in-depth
- Department-based document ownership (engineering, finance, marketing, hr, general)
- Sensitivity levels: general, internal, confidential, restricted

**Chat and Generation**
- Answers generated only from policy-approved context
- Inline citations with source paths and section references
- System prompt enforces grounded responses (no hallucination)

**HR Data Endpoints**
- Row-level access (HR and C-level only)
- Masked rows with PII redaction (configurable via feature flag)
- Safe aggregates (averages, counts) without individual data

**Observability**
- Correlation ID on every request (X-Correlation-ID header)
- Per-stage timing metrics (embed_ms, retrieve_ms, pdp_ms, llm_ms)
- JSON audit logs to stdout and daily rotating files

**Security**
- HTTP Basic auth (demo; use proper auth in production)
- SHA256 password hashing with salt
- Security headers: X-Frame-Options, X-Content-Type-Options, X-XSS-Protection, CSP, Referrer-Policy

---

## Quick Start

**Requirements**
- Python 3.10 or higher
- pip
- OpenAI API key (for embeddings and LLM)

**Installation**

```bash
# Clone the repository
git clone <repo-url>
cd ds-rpc-01

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .

# Set OpenAI API key
export OPENAI_API_KEY=sk-...
# Or create a .env file:
# OPENAI_API_KEY=sk-...

# Start the server
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/ in your browser. Login with demo credentials (see [Demo Users](#demo-users)).

**First-time setup**: Login as `Clark/chief` (c_level) and click "Reindex (dense)" to build the vector index.

---

## Architecture

```
Request Flow:

  Client (HTTP Basic Auth)
         |
         v
  +------------------+
  |  FastAPI App     |
  |  - Auth Layer    |
  |  - Security MW   |
  |  - Correlation   |
  +--------+---------+
           |
     +-----+------+
     |            |
     v            v
  /chat        /hr/*
  /search/*    endpoints
     |            |
     v            v
  Hybrid       CSV Service
  Retrieval    (rows, mask,
  (Dense+BM25)  aggregates)
     |            |
     v            v
  +------------------+
  |  PDP Policy      |
  |  Evaluation      |
  |  (per-item)      |
  +--------+---------+
           |
           v
  LLM Generation
  (allowed context only)
```

**Data Flow**

1. **Ingestion** (`app/ingest/`)
   - `discovery.py`: Scans `resources/data/{dept}/` for .md and .csv files
   - `chunker.py`: Splits markdown by headings, tracks section path
   - `allow_roles.py`: Computes which roles can access each chunk
   - `dense_indexer.py`: Embeds chunks and stores in ChromaDB
   - `runner.py`: Builds in-memory BM25 index

2. **Retrieval** (`app/retrieval/`)
   - `bm25.py`: Sparse keyword search with BM25 scoring
   - `hybrid.py`: RRF fusion of dense and sparse results
   - `filtering.py`: Pre-filters by `allowed_roles` before search
   - `rerank.py`: Optional second-pass ranking

3. **Policy** (`app/policy/pdp.py`)
   - Loads YAML policy from `docs/policy.yaml`
   - Evaluates rules top-to-bottom, first match wins
   - Supports: `eq`, `ne`, `in`, `not_in` operators
   - Runtime flags for feature toggles

4. **Generation** (`app/services/generation.py`)
   - Builds prompt with context snippets and source references
   - Calls OpenAI chat completions API
   - Falls back to extractive response if no API key

---

## Project Structure

```
ds-rpc-01/
├── app/
│   ├── main.py              # FastAPI app, all endpoints
│   ├── policy/
│   │   └── pdp.py           # Policy Decision Point engine
│   ├── ingest/
│   │   ├── discovery.py     # File discovery and metadata
│   │   ├── chunker.py       # Markdown chunking
│   │   ├── allow_roles.py   # Role permission computation
│   │   ├── dense_indexer.py # ChromaDB indexing
│   │   └── runner.py        # BM25 index builder
│   ├── retrieval/
│   │   ├── bm25.py          # BM25 implementation
│   │   ├── hybrid.py        # RRF fusion
│   │   ├── filtering.py     # Role-based pre-filtering
│   │   └── rerank.py        # Reranking strategies
│   ├── services/
│   │   ├── embeddings.py    # OpenAI embedding calls
│   │   ├── generation.py    # LLM answer generation
│   │   └── reranker_openai.py # OpenAI-based reranking
│   ├── vectorstore/
│   │   └── chroma_store.py  # ChromaDB operations
│   ├── hr/
│   │   └── csv_service.py   # HR CSV data handling
│   └── utils/
│       ├── config.py        # Environment/settings
│       └── audit.py         # Logging and correlation IDs
├── docs/
│   ├── policy.yaml          # ABAC policy definition
│   └── ARCHITECTURE.md      # Detailed architecture doc
├── resources/data/          # Knowledge base documents
│   ├── engineering/
│   ├── finance/
│   ├── general/
│   ├── hr/
│   └── marketing/
├── templates/
│   └── chat.html            # Web UI template
├── static/
│   ├── app.js               # Frontend JavaScript
│   └── style.css            # Styling
├── tests/                   # Test suite
└── pyproject.toml           # Dependencies
```

---

## Policy System

The policy is defined in `docs/policy.yaml`. Key rules:

| Rule | Who | Can Access |
|------|-----|------------|
| c_level_full_access | C-level executives | Everything |
| general_docs_any_role | All roles | Documents in `general/` folder |
| same_department_internal_docs | Department members | Their own department's internal docs |
| hr_rows_hr_and_c_level_only | HR, C-level | Raw HR CSV data |
| hr_masked_rows_non_hr_if_enabled | Non-HR (when flag enabled) | Masked HR data (PII redacted) |
| hr_aggregates_non_hr_if_enabled | Non-HR (when flag enabled) | Safe aggregates only |

**Resource Attributes**
- `owner_dept`: engineering, finance, marketing, hr, general
- `doc_type`: md, csv, aggregate
- `sensitivity`: general, internal, confidential, restricted

**Actions**
- `retrieve_chunk`: Fetch text for RAG context
- `view_source`: Show citation to user
- `view_row`: Access raw HR CSV row
- `view_masked_row`: Access PII-redacted row
- `aggregate_query`: Request aggregated statistics

---

## API Reference

All endpoints require HTTP Basic authentication.

### Chat

**POST /chat**

Ask a question. Returns LLM-generated answer with citations.

```json
{
  "message": "What is the Q4 marketing budget?",
  "top_k": 5,
  "rerank": true,
  "reranker": "openai"
}
```

Response includes: `answer`, `citations[]`, `metrics{}`, `correlation_id`

### Search

**POST /search/dense** - Vector search only
**POST /search/hybrid** - Dense + sparse with RRF fusion
**POST /search** - Sparse/keyword search only

Request body same as chat. Returns `results[]` with text, source_path, section_path, and policy decision.

### HR Data

**GET /hr/rows** - Raw HR records (HR/C-level only)
**GET /hr/rows_masked** - PII-redacted records (requires HR_MASKED_ROWS_MODE=enabled)
**GET /hr/aggregate** - Safe aggregates (requires HR_AGGREGATE_MODE=enabled for non-HR)

### Admin (C-level only)

**GET /admin/status** - Index counts and OpenAI health
**POST /admin/reindex** - Rebuild sparse (BM25) index
**POST /admin/reindex_dense** - Rebuild dense (ChromaDB) index

### Health

**GET /health/openai** - Verify OpenAI API connectivity

---

## Configuration

Set via environment variables or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | OpenAI API key for embeddings and chat |
| `OPENAI_EMBEDDING_MODEL` | text-embedding-3-small | Embedding model |
| `OPENAI_CHAT_MODEL` | gpt-4o-mini | Chat completion model |
| `CHROMA_DB_DIR` | .chroma | ChromaDB persistence directory |
| `HR_AGGREGATE_MODE` | disabled | Enable aggregate endpoint for non-HR |
| `HR_MASKED_ROWS_MODE` | disabled | Enable masked rows for non-HR |
| `AUDIT_LOG_DIR` | logs/audit | Directory for audit log files |
| `ENABLE_DOCS` | true | Show /docs (Swagger UI) |
| `LLM_CACHE_DIR` | .llm_cache | Persistent cache directory |
| `LLM_CACHE_ENABLED` | true | Enable/disable caching |

### Persistent Cache

The system includes a disk-based cache (using SQLite via `diskcache`) that persists across restarts:

- **Embedding cache**: Stores OpenAI embeddings indefinitely (deterministic, safe to cache)
- **LLM response cache**: Stores chat responses for 1 hour (query + context hash)

This significantly speeds up repeated queries. Cache stats are shown in `/admin/status`.

To clear the cache: `POST /admin/cache/clear` (c_level only)

---

## Testing

```bash
# Run all tests
pytest -v

# Run specific test file
pytest tests/test_pdp.py -v

# Run with coverage
pytest --cov=app --cov-report=term-missing
```

Test categories:
- `test_pdp.py`: Policy engine unit tests
- `test_app_authz.py`: Authorization integration tests
- `test_ingest.py`, `test_chunker.py`: Ingestion pipeline tests
- `test_retrieval_filtering.py`: Pre-filter tests
- `test_*_search_api.py`: Search endpoint tests
- `test_chat_api.py`: Chat endpoint tests
- `test_e2e_samples.py`: End-to-end sample queries

Tests use monkeypatching for OpenAI calls and ChromaDB, no network required.

---

## Demo Users

| Username | Password | Role | Access |
|----------|----------|------|--------|
| Tony | password123 | engineering | Engineering docs + general |
| Bruce | securepass | marketing | Marketing docs + general |
| Sam | financepass | finance | Finance docs + general |
| Peter | pete123 | engineering | Engineering docs + general |
| Sid | sidpass123 | marketing | Marketing docs + general |
| Natasha | hrpass123 | hr | HR docs/data + general |
| Clark | chief | c_level | Everything + admin panel |

---

## Troubleshooting

**Chat returns empty or low-quality answers**
- Dense index may be empty. Login as Clark, click "Reindex (dense)" in admin panel.
- Verify `OPENAI_API_KEY` is set correctly.

**"chromadb package not installed" error**
- Run `pip install chromadb`

**403 Forbidden on search**
- User's role does not have access to the requested resource.
- Check policy rules in `docs/policy.yaml`.

**Slow embedding calls**
- First call may take longer due to connection setup.
- Typical embed time: 100-500ms per query.

**HR endpoints return 403**
- `/hr/rows`: Only HR and C-level can access.
- `/hr/rows_masked`: Set `HR_MASKED_ROWS_MODE=enabled` in environment.
- `/hr/aggregate`: Set `HR_AGGREGATE_MODE=enabled` in environment.

---

## Sample Results

Tested with live OpenAI embeddings:

**Query**: "marketing ROI 2024" (as Bruce/marketing)
- Retrieved: marketing_report_q1_2024.md, market_report_q4_2024.md
- Answer discussed Q1 target ~3x ROI, Q4 spend of ~$2.5M

**Query**: "employee salary details" (as Bruce/marketing)
- Retrieved: employee_handbook.md (Salary Structure section)
- HR CSV data was blocked by policy

**Query**: "Q4 2024 revenue" (as Sam/finance)
- Retrieved: quarterly_financial_report.md
- Policy correctly allowed finance department access

---

## License

Internal use only.
