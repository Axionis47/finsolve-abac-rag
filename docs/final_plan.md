# FinSolve RAG Chatbot – Final Architecture and Build Plan

Version: 1.0  
Last Updated: 2025-09-22

## 1) Purpose and scope
Build an internal, access-controlled RAG chatbot for FinSolve that answers questions with high precision, grounded in company documents, and enforces ABAC over RBAC at chunk/row level. All answers include citations.

## 2) Core requirements
- Authentication: BasicAuth (per challenge), user→role mapping
- Access control: ABAC layered over RBAC; strict HR handling
- Data: Markdown knowledge (engineering, finance, marketing, general), HR CSV (restricted)
- RAG: Hybrid retrieval (dense + sparse) with policy filtering and citations
- LLM/Embeddings: OpenAI (gpt-4o-mini, text-embedding-3-small)
- Vector DB: ChromaDB (persistent ./.chroma)
- Observability: Audit of PDP decisions and citations

## 3) Key decisions (final)
- Vector store: ChromaDB
- Retrieval: Hybrid (Chroma dense) + in-memory BM25, fused by RRF; optional reranker later
- Policy model: ABAC over RBAC (docs/policy.yaml); compute allowed_roles at ingest time
- Query normalization: alias “FinNova”→“FinSolve” at query time; preserve originals for citations
- HR aggregates for non‑HR: disabled by default; can be enabled via env flag

## 4) Access control strategy (summary)
- Roles: finance, marketing, engineering, hr, c_level, employee
- Ownership (enforcement anchor): owner_dept inferred from directory (engineering | finance | marketing | hr | general)
- General docs: visible to all roles
- Departmental Markdown: internal; visible only to same department and c_level
- HR CSV rows: restricted; visible only to hr and c_level at row/field level
- Mentions do NOT grant access; only ownership + sensitivity matter
- Default: deny

## 5) Data audit (summary)
- Engineering: long, structured Markdown; internal, no PII
- Finance: quarterly + annual summaries; internal
- Marketing: Q1–Q4 + annual; internal; “FinNova” alias issue in Q1
- General: employee handbook; available to all
- HR: csv with PII & sensitive metrics; restricted

## 6) Metadata schema (docs/metadata_schema.yaml)
Common fields: id, owner_dept, doc_type, sensitivity, source_path, title, section_path, year, quarter, topics, allowed_roles, created_at, updated_at, checksum  
Markdown chunk: text, section_level, anchor  
HR row: row_id, columns, pii_fields_present (true)  
HR aggregate (optional): agg_key, metrics, provenance

## 7) Policy (docs/policy.yaml)
- Actions: retrieve_chunk, view_source, view_row, aggregate_query
- C-Level: permit all
- General: permit all roles for retrieve_chunk/view_source
- Departmental Markdown: permit same dept and c_level; deny cross-dept by default
- HR rows: view_row permitted only to hr and c_level
- HR aggregates: permitted to hr/c_level; non‑HR only if HR_AGGREGATE_MODE=enabled
- allowed_roles computation included to speed filtering

## 8) Ingestion & indexing (docs/ingestion_plan.md)
Pipeline:
1) Discover files under resources/data/**; infer owner_dept from path
2) Parse & normalize (no text mutation)
3) Chunk Markdown heading-aware; keep tables/code blocks intact; window long sections (120–600 tokens)
4) HR CSV → one record per row (restricted); serialize columns for embedding
5) Attach metadata; compute allowed_roles via policy; compute checksum
6) Index
   - Dense: ChromaDB collections
     - kb (Markdown + general)
     - hr_sensitive (HR rows)
     - hr_aggregates (optional)
   - Sparse: in-memory BM25 over the same records for current process
7) Verification: spot-check metadata; ensure HR rows appear only in hr_sensitive
8) Updates: idempotent upserts via deterministic ids + checksums

## 9) Retrieval & generation
Query flow:
- Normalize query (aliases)
- Build PDP filter: role ∈ allowed_roles
- Dense top_k from Chroma with metadata filter
- Sparse top_k from BM25 with same filter
- Fuse via RRF; optional rerank later
- Redact/deny per policy; pack approved context with section_path and source_path
- Generate with OpenAI chat model; include citations

## 10) API surface (initial)
- GET /login (auth check)
- GET /test (protected ping)
- GET /health/openai (checks model availability)
- POST /chat {message: str}
  - Authenticated; role from user session
  - Returns: {answer: str, citations: [str]}

## 11) Configuration
- .env: OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL, OPENAI_CHAT_MODEL, CHROMA_DB_DIR, HR_AGGREGATE_MODE, HOST, PORT
- .gitignore: .env (already), recommend .chroma and .DS_Store

## 12) Observability & audit
- Per-request audit: user, role, query, PDP filter, collections used, ids of retrieved chunks, redactions/denials, timings
- Metrics: retrieval hit ratios, RRF contributions, policy denial counts

## 13) Security & privacy
- HR CSV: PII restricted to hr/c_level
- Non‑HR HR requests: denied by default; aggregate-only is explicit opt-in
- Least privilege by default; default deny

## 14) Testing strategy
- Unit tests
  - PDP rule table tests per role/owner_dept/sensitivity/action
  - Metadata extraction tests (section_path, year/quarter)
  - allowed_roles computation tests
- Integration tests
  - Ingestion → index: ensure records placed in correct collections
  - Retrieval: policy-filtered results; Q-specific queries hit right sections
  - HR access: hr/c_level allowed; others denied (or aggregates if enabled)
- E2E tests
  - /login, /test, /health/openai, /chat (role-aware responses with citations)

## 15) Build milestones (iterative)
M1. Wiring & policy foundation
- [x] Env loader & OpenAI health
- [ ] PDP (policy engine) + unit tests

M2. Ingestion metadata skeleton
- [ ] File discovery, chunking (Markdown), HR CSV row serialization
- [ ] Metadata extraction (section_path, time); allowed_roles computation; checksums

M3. Indexing
- [ ] Chroma collections + embedding calls
- [ ] BM25 sparse index in-memory

M4. Retrieval & generation
- [ ] RRF fusion with ABAC filtering
- [ ] Answer generation with citations

M5. UI & observability
- [ ] Streamlit chat UI (role badge, sources, redaction notices)
- [ ] Audit logging and basic metrics

Optional M6. Enhancements
- [ ] Enable HR aggregates for non‑HR; add hr_aggregates ingestion and retrieval path
- [ ] Cross-encoder reranker if needed

## 16) Risks & mitigations
- Rate limits/costs → batch embeddings, backoff; index only diffs (checksums)
- Policy drift → single source of truth in docs/policy.yaml; test first
- Data quality (aliases) → query normalization; preserve originals for citations
- Over-splitting → heading-aware chunking with windowing

## 17) Change management
- This document is the living plan. Any change proposal must include: rationale, impacted modules, and test updates.

