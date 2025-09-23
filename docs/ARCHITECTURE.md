# FinSolve ABAC RAG — System Documentation

## 1) Executive Summary
FinSolve’s internal chatbot answers department-specific questions using a policy-driven Retrieval-Augmented Generation (RAG) pipeline. Access is governed by an Attribute-Based Access Control (ABAC) policy enforced by a Policy Decision Point (PDP). The system combines dense retrieval (Chroma + OpenAI embeddings) and sparse retrieval (BM25), with optional reranking and LLM synthesis that cites sources. Observability includes per-stage timings, correlation IDs, and JSONL audit logs.

Status: complete and operational.

- Dense (Chroma) chunks: 203 (persisted under `.chroma`)
- Sparse (BM25) chunks: 203 (in-memory)
- Tests: 28 passing

---

## 2) System Architecture
```mermaid
flowchart TD
  subgraph Client
    U[Browser UI (/)]
  end

  subgraph FastAPI Backend
    A[HTTP Basic Auth\nusers_db (dev)]
    P[PDP / ABAC\npolicy.yaml -> PDP.evaluate]
    R[Retrieval Orchestrator\n/search/*, /chat]
    RR[Reranking\n simple | openai | openai_llm]
    F[Fusion (RRF)]
    G[Generation\nOpenAI Chat (urllib)\n"answer only from context" + citations]
    H[HR CSV Service\nrows / aggregates / masked]
    O[Observability\nmetrics + X-Correlation-ID]
    L[Audit Logger\nJSONL daily rotation]
  end

  subgraph Data & Services
    D1[resources/data\nMarkdown & HR CSV]
    D2[BM25 In-Memory Index]
    D3[ChromaDB (.chroma)\nCollection kb_main]
    S1[OpenAI Embeddings\n(text-embedding-3-small)]
    S2[OpenAI Chat\n(gpt-4o-mini)]
    LG[logs/audit/*.jsonl]
  end

  U -->|HTTP Basic| A
  U -->|/chat & /search| R
  A --> R
  R -->|discover + chunk| D1
  D1 -->|build| D2
  D1 -->|index_corpus_to_chroma| S1
  S1 --> D3

  R -->|dense| D3
  R -->|sparse| D2
  R --> F --> RR --> R
  R -->|PDP checks| P
  R -->|snippets| G
  G -->|response with citations| U

  R --> O
  R --> L --> LG

  U -->|/hr/*| H
  H --> P
  H --> L

  classDef store fill:#eef,stroke:#88a,stroke-width:1px
  class D1,D2,D3,LG store;
```

---

## 3) Key Components
- Authentication (dev): HTTP Basic; role resolved server-side from `users_db`.
- ABAC & PDP: Policy rules in `docs/policy.yaml`; checks for `retrieve_chunk`, `view_source`, `view_row`, `view_masked_row`, `aggregate_query`. Feature flags: `hr_aggregate_mode`, `hr_masked_rows_mode`.
- Ingestion:
  - Discovery: scans `resources/data/<dept>/` for Markdown/CSV.
  - Chunking: heading-aware Markdown chunking; builds `section_path`.
  - Allowed roles: computed and denormalized as boolean flags (`role_<role>`) for Chroma.
  - Dense index: embeddings + upsert to Chroma collection `kb_main`.
  - Sparse index: in-memory BM25 over chunk texts.
- Retrieval:
  - Dense: Chroma query with where filter by role flag.
  - Sparse: BM25 over allowed-roles subset.
  - Fusion: Reciprocal Rank Fusion (RRF).
  - Optional reranking: `simple` (local BM25), `openai` (embedding cosine), `openai_llm` (chat prompt returning JSON order).
- Generation:
  - OpenAI Chat completions via urllib with strict prompt: answer only from context, inline citations [#], final Citations section.
  - Safe fallback if no API key.
- HR Data:
  - `/hr/rows`: HR/C-level only (row-level PII).
  - `/hr/aggregate`: non-HR allowed if `HR_AGGREGATE_MODE=enabled`.
  - `/hr/rows_masked`: PII-redacted rows for non-HR if `HR_MASKED_ROWS_MODE=enabled`; HR/C-level always allowed.
- Observability & Audit:
  - Stage timings (ms): embed, retrieve, rerank, pdp, llm, total.
  - `X-Correlation-ID` header + in body.
  - JSONL audit logs with daily rotation under `logs/audit/`.

---

## 4) Data Flow (Chat)
1. Ensure indices (sparse memory; dense persisted).
2. Embed query (OpenAI embeddings).
3. Retrieve candidates: dense (Chroma) + sparse (BM25).
4. Fuse with RRF; optionally rerank (simple/openai/openai_llm).
5. PDP post-check per candidate; build allowed context.
6. Generate answer via OpenAI Chat with strict prompt and citations.
7. Return answer, citations, metrics, and correlation ID; log audit events.

---

## 5) Endpoints
- UI
  - `GET /` → Chat UI (Admin panel visible for `c_level`).
- Auth/Health
  - `GET /login`, `GET /health/openai`.
- Admin (c_level)
  - `POST /admin/reindex` → build sparse (BM25) index in memory.
  - `POST /admin/reindex_dense` → index Markdown to Chroma (`.chroma`).
  - `GET /admin/status` → `{sparse_count, dense_count, persist_dir, openai}`.
- Search
  - `POST /search/dense`
  - `POST /search/hybrid`
  - Body: `{ query, top_k, persist_dir?, base_dir?, rerank?:bool, reranker?:"simple"|"openai"|"openai_llm" }`.
- Chat
  - `POST /chat` → `{ answer, citations[], metrics{}, correlation_id }`.
- HR
  - `GET /hr/rows` → HR/C-level only.
  - `GET /hr/aggregate` → non-HR only if `HR_AGGREGATE_MODE=enabled`.
  - `GET /hr/rows_masked` → non-HR only if `HR_MASKED_ROWS_MODE=enabled` (PII redacted), HR/C-level always.

---

## 6) Configuration
- `OPENAI_API_KEY` (required for dense indexing and live LLMs)
- `OPENAI_EMBEDDING_MODEL` (default: `text-embedding-3-small`)
- `OPENAI_CHAT_MODEL` (default: `gpt-4o-mini`)
- `CHROMA_DB_DIR` (default: `.chroma`)
- `HR_AGGREGATE_MODE`, `HR_MASKED_ROWS_MODE` (default: `disabled`)
- `AUDIT_LOG_DIR` (default: `logs/audit`)
- `HOST`, `PORT`

---

## 7) Running & Population
- Run the app (example): `uvicorn app.main:app --reload`
- Populate dense store (requires `OPENAI_API_KEY`):
  - API: `POST /admin/reindex_dense` (c_level)
  - Programmatic: `index_corpus_to_chroma(base_dir="resources/data", persist_dir=".chroma")`
- Populate sparse in-memory index:
  - API: `POST /admin/reindex` (c_level)
- Verify status:
  - API: `GET /admin/status` (c_level) → counts & OpenAI health

---

## 8) UI Usage
- Visit `GET /` and authenticate via HTTP Basic.
- Ask a question; select Top-K and reranker if desired.
- Observe answer, citations, per-stage timings, and correlation ID.
- If logged in as `c_level`, use the Admin panel to reindex and view status.

---

## 9) Audit & Observability
- Each request emits `X-Correlation-ID` header and includes the same in the JSON body.
- JSONL audit logs written to `logs/audit/YYYY-MM-DD.jsonl` (rotated daily).
- Search/Chat responses include per-stage timings: `embed_ms`, `retrieve_ms`, `rerank_ms`, `pdp_ms`, `llm_ms`, `total_ms`.

---

## 10) Testing
- Test suite: 28 tests (unit + integration).
- External services (OpenAI) are monkeypatched; tests are deterministic and offline.
- Run: `pytest -q`

---

## 11) Prompt Usage
- Generation prompt mandates: answer only from provided snippets, include inline citations [#], and a final Citations section.
- LLM reranker prompt: instruction to return a JSON array of snippet indices in relevance order.

---

## 12) Security Notes & Future Work
- Dev-only auth: move to a real IdP/SSO; add password hashing and session management.
- Add rate limiting and input validation.
- Consider streaming responses (SSE/WebSocket) for chat.
- Dockerize and provide a deployment runbook.
- Expand policy for finer sensitivity and field-level controls as real data evolves.

