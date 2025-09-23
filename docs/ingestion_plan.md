# Ingestion & Indexing Plan (FinSolve RAG)

Version: 0.1  
Last Updated: 2025-09-22

## Goals
- Create high-quality retrievable units (chunks/rows) with rich metadata to support ABAC filtering and precise citations.
- Build dense (ChromaDB) and sparse (BM25) indices over the same, policy-filtered corpus.
- Keep all choices swappable with minimal coupling.

## Sources
- Markdown (engineering/finance/marketing/general)
- HR CSV (restricted) with optional safe aggregates (if hr_aggregate_mode enabled)

## Pipeline
1) Discovery
   - Enumerate files under resources/data/**
   - Determine owner_dept from path segment: engineering | finance | marketing | hr | general
2) Parse & Normalize
   - Markdown: parse headings (H1..Hn), identify tables and fenced code blocks
   - CSV (HR): load rows; set row_id from employee_id; attach sensitivity=restricted
   - Do NOT modify original text. For aliasing (FinNova→FinSolve), apply only at query time.
3) Chunking
   - Markdown:
     - Split by headings; keep tables/code blocks intact with their nearest heading
     - Max chunk tokens ~600, min ~120; allow overlap up to 50 tokens when splitting long sections
     - Capture section_path as "H1/H2/H3"
   - HR CSV:
     - Each row becomes a record with columns serialized to a human-readable text string for embeddings (for HR & C-Level only)
     - If hr_aggregate_mode enabled: compute aggregates with pandas and serialize as compact fact strings
4) Metadata Attachment
   - owner_dept, doc_type, sensitivity, source_path, section_path, title, year, quarter, topics
   - Compute allowed_roles using docs/policy.yaml rules and attach
   - Compute checksum for drift detection
5) Indexing
   - Dense: ChromaDB
     - Collections:
       - kb (all Markdown chunks + general)
       - hr_sensitive (HR rows; restricted)
       - hr_aggregates (optional)
     - Embeddings: OpenAI text-embedding-3-small (1536-d)
     - Metadata stored with each embedding record
   - Sparse: In-memory BM25 (Whoosh/Tantivy-like) over the same records:
     - Tokenize text and headings
     - Maintain a mapping id → metadata for filtering
6) Verification
   - Spot-check 10 samples per collection for metadata correctness (owner_dept, section_path, allowed_roles)
   - Ensure HR rows only list allowed_roles: [hr, c_level]
7) Updates & Drift
   - Recompute checksum per resource segment; if changed, upsert in Chroma and BM25
   - Maintain idempotent upserts (id = hash(source_path + offsets|row_id))
8) Policy Filters in Retrieval
   - At query time, pre-compute PDP filter: role ∈ allowed_roles
   - Pass the filter to Chroma; apply same filter pre/post BM25 retrieval

## Retrieval Fusion
- Execute top_k_dense (Chroma) and top_k_sparse (BM25) under the same ABAC filter
- Fuse via Reciprocal Rank Fusion (RRF) with k=60
- Optional reranker: Cross-Encoder (can be enabled later if needed)

## Citations
- Return source_path with section_path (Markdown) or source_path + row_id (HR)
- For tables, include nearest section heading in citation label

## HR Handling (strict)
- Default (hr_aggregate_mode = disabled):
  - Only hr and c_level can access hr_sensitive collection (row level)
  - Others are denied for HR queries
- Optional future mode: enable hr_aggregate_mode to expose hr_aggregates to non‑HR

## Operational Notes
- Batch sizes: 64 records per embedding batch (rate-limit friendly)
- Backoff on OpenAI rate errors with jitter
- Environment
  - OpenAI API key via env var
  - ChromaDB persistent directory: ./.chroma
- Logging
  - Ingestion log per file: counts, errors, computed metadata summary

## Minimal QA Checklist
- [ ] Owner_dept correct for 20 sampled chunks
- [ ] Section_path reasonable and human-readable
- [ ] Year/quarter extracted for Finance/Marketing sections
- [ ] allowed_roles aligns with policy.yaml
- [ ] HR rows present only in hr_sensitive; not in kb
- [ ] Q4 queries return Q4 sections in top-5 (manual check)

