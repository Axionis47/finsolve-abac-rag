# Indexing Strategy: Design Decisions and Thought Process

This document explains how the document structure informed every indexing decision. It covers why the pipeline was built this way, what alternatives were considered, and what trade-offs were made.

---

## The Core Question

When you have documents owned by different departments, with different sensitivity levels, and different data formats (markdown vs. CSV), how do you build an index that:

1. Returns relevant results fast
2. Never shows unauthorized content
3. Supports different levels of disclosure (full, masked, aggregate)
4. Handles temporal queries ("Q2 2024 marketing spend")

---

## Document Landscape Analysis

Before writing any indexing code, the first step was understanding what the data looks like.

### What We Have

```
resources/data/
  general/employee_handbook.md              # 1 file, public, all roles
  engineering/engineering_master_doc.md      # 1 file, internal, engineering-owned
  finance/financial_summary.md              # 2 files, internal, finance-owned
  finance/quarterly_financial_report.md
  marketing/marketing_report_2024.md        # 5 files, internal, marketing-owned
  marketing/marketing_report_q1_2024.md     # Temporal: Q1-Q4 naming pattern
  marketing/marketing_report_q2_2024.md
  marketing/marketing_report_q3_2024.md
  marketing/market_report_q4_2024.md
  hr/hr_data.csv                            # 1 file, restricted, PII, HR-owned
```

### What This Tells Us

**Observation 1: Documents have natural owners.**
The directory structure maps to departments. `finance/` files belong to finance. `hr/` files belong to HR. This directory-based ownership is the simplest and most reliable way to assign access control at discovery time.

**Observation 2: Markdown files have semantic structure.**
Headings (H1, H2, H3) create natural chunk boundaries. A section about "Q2 Campaign Performance" is a self-contained unit that makes sense on its own. Fixed-size windowing would break these boundaries.

**Observation 3: Some filenames encode time.**
`marketing_report_q1_2024.md` tells us this is Q1 2024 data. This metadata is not in the file content header; it is in the filename. The indexer must extract it.

**Observation 4: HR data is structurally different.**
CSV rows are tabular, not prose. Each row is an employee record with PII fields (salary, DOB, email). You cannot chunk a CSV the same way you chunk markdown. Each row needs its own indexing logic.

**Observation 5: Access is not binary.**
Some data is fully restricted (HR salary). Some is safe as aggregates (average salary by department). Some is safe with redaction (employee name and department without salary). This means access control needs more than "allow" and "deny."

---

## Decision 1: Directory-Based Ownership Discovery

### The Decision

The `discovery.py` module scans `resources/data/` and assigns `owner_dept` based on the subdirectory name. A file at `finance/quarterly_report.md` gets `owner_dept="finance"`.

### Why

- The directory structure is the single source of truth for document ownership.
- It requires zero configuration. Adding a new department means adding a new folder.
- It is impossible to accidentally index a file without an owner, because every file lives in a department folder.

### What Was Considered

**Alternative: YAML frontmatter in each file.** Each markdown file could declare its owner in a YAML header. Rejected because it requires every document author to add metadata, and missing frontmatter would create unowned documents.

**Alternative: Filename pattern matching.** Parse the filename to infer the department. Rejected because filenames are inconsistent (some say `financial_`, some say `finance_`).

### Implementation

```python
# discovery.py
DEPTS = {"engineering", "finance", "marketing", "hr", "general"}

def discover_resources(base_dir):
    items = []
    for dept_dir in Path(base_dir).iterdir():
        if dept_dir.name in DEPTS:
            for f in dept_dir.iterdir():
                items.append({
                    "path": f,
                    "owner_dept": dept_dir.name,
                    "doc_type": "csv" if f.suffix == ".csv" else "md",
                    "sensitivity": "restricted" if dept_dir.name == "hr" else "internal",
                })
    return items
```

---

## Decision 2: Heading-Aware Markdown Chunking

### The Decision

The `chunker.py` module splits markdown by headings (H1 through H6). Each chunk is a self-contained section with a breadcrumb path like `Financial Report > Q2 Results > Revenue Breakdown`.

### Why

Headings mark semantic boundaries. A chunk that starts at "## Revenue" and ends before "## Expenses" contains one coherent topic. This is better for:

- **Retrieval accuracy.** The embedding captures one topic, not a blend of two.
- **Citation quality.** The section path tells the user exactly where the answer came from.
- **Access control.** A document might have a public introduction and a confidential appendix. Heading-based chunking lets you assign different access levels to different sections.

### What Was Considered

**Alternative: Fixed-size windows (512 tokens).** Simple to implement, but splits mid-sentence and mid-section. A chunk might contain half of "Revenue" and half of "Expenses," making it semantically ambiguous.

**Alternative: Recursive character splitting.** Used as a fallback when a heading section exceeds the token limit. The chunker first tries paragraphs (`\n\n`), then sentences (`. `), then words (` `), with overlap between consecutive chunks.

### Implementation Details

- Token estimation: 4 characters = 1 token (avoids tiktoken dependency).
- Maximum chunk size: configurable, default 512 tokens.
- Overlap: 50 tokens between split chunks for context continuity.
- Code blocks: Fenced code blocks (```) are not split. Headings inside code blocks are ignored.

### Trade-off

Heading-aware chunking produces variable-size chunks. Some sections are 50 tokens, some are 500. This means BM25 length normalization matters more. The BM25 `b` parameter (default 0.75) compensates for this.

---

## Decision 3: Temporal Metadata Extraction

### The Decision

The `metadata_extractor.py` module extracts year and quarter from both filenames and content. This metadata is stored in ChromaDB alongside the embeddings.

### Why

The marketing team has quarterly reports. A user asking "What was Q2 2024 marketing spend?" expects results from the Q2 file, not Q1. Without temporal metadata, the retriever relies entirely on embedding similarity, which might rank Q1 and Q2 equally if they use similar language.

### How It Works

1. **Filename extraction:** Regex `q([1-4])[-_]?(20\d{2})` matches `marketing_report_q1_2024.md` and extracts `Q1, 2024`.
2. **Content extraction:** Regex `\b(Q[1-4])\b` finds all quarter mentions in the text. The most frequent one becomes the primary quarter.
3. **Year extraction:** Regex `\b(202[0-9])\b` finds year references. Content years override filename years if they disagree.

### What Gets Stored

```python
{
    "year": "2024",
    "quarter": "Q1",
    "temporal_refs": "Q1,Q2,Q3",  # All mentioned quarters
}
```

### Trade-off

Temporal extraction is heuristic. A document that compares Q1 and Q3 will have both quarters mentioned. The "primary" quarter is the most frequent one, which is usually correct but not always. For precise temporal filtering, the user would need to add explicit metadata.

---

## Decision 4: Context-Aware Embeddings

### The Decision

Before embedding a chunk, the `dense_indexer.py` prepends metadata tags:

```
[DEPARTMENT: FINANCE] [SENSITIVITY: INTERNAL] [TYPE: MD] [QUARTER: Q2] [YEAR: 2024]
[DOCUMENT: quarterly financial report]

Revenue for Q2 2024 was $12.3M...
```

The original text is stored separately for display. Only the prefixed text goes through the embedding model.

### Why

Two chunks with identical text but different owners should not have identical embeddings. Consider: both the finance report and the marketing report mention "Q2 revenue was $12.3M." Without context prefixing, both chunks would have nearly identical embeddings, and the retriever would not distinguish between them.

With the prefix, the finance chunk's embedding is shifted toward the "FINANCE" semantic space, and the marketing chunk's embedding is shifted toward "MARKETING." A query about "finance Q2 revenue" now has a stronger cosine similarity to the finance chunk.

### What Was Considered

**Alternative: No prefix, rely on post-filtering.** The retriever returns both chunks, and the PDP filters out the unauthorized one. This works but wastes retrieval slots. If `top_k=5` and 3 of the 5 results are unauthorized, the user only gets 2 useful results.

**Alternative: Separate collections per department.** Each department gets its own ChromaDB collection. Rejected because cross-department queries (C-level asking "compare engineering and marketing budgets") would require searching multiple collections and merging results.

### Trade-off

The prefix adds tokens to the embedding input. This means the embedding model processes more text per chunk, which increases cost slightly. For `text-embedding-3-small` at 1536 dimensions, the added prefix (about 30 tokens) is negligible compared to the typical chunk size (200-500 tokens).

---

## Decision 5: Three-Tier HR Data Access

### The Decision

HR CSV data is indexed at three levels:

1. **Full row:** All fields including salary, DOB, performance rating. Stored as a chunk for HR and C-level.
2. **Masked row:** Name, department, role, location visible. PII fields replaced with "[REDACTED]". Available when `hr_masked_rows_mode=enabled`.
3. **Aggregates:** Pre-computed statistics (headcount by department, average salary by department, location distribution). Available when `hr_aggregate_mode=enabled`.

### Why

The HR CSV has 15 columns. Some are public (name, department), some are sensitive (salary, DOB), and some are operational (leave balance, attendance). A simple allow/deny at the row level means either you see everything or nothing. The business needs something in between.

Example scenario: A marketing manager asks "How many people are in engineering?" This is a reasonable question that does not require seeing individual salaries. The aggregate tier provides this answer.

### How the Tiers Map to the Pipeline

| Tier | What Gets Indexed | Who Sees It | When |
|------|------------------|-------------|------|
| Full | Searchable text: name, dept, role, location, join date. Full row stored in metadata. | HR, C-level | Always |
| Masked | Searchable text: dept, role, location. PII fields show "[REDACTED]". | All departments | `hr_masked_rows_mode=enabled` |
| Aggregate | Pre-computed stats as prose chunks. | All departments | `hr_aggregate_mode=enabled` |

### PII Field Classification

```python
PII_FIELDS = [
    "email", "salary", "date_of_birth", "manager_id",
    "leave_balance", "leaves_taken", "attendance_pct",
    "performance_rating", "last_review_date"
]

SEARCHABLE_FIELDS = [
    "full_name", "role", "department", "location", "date_of_joining"
]
```

The searchable text for embedding uses only `SEARCHABLE_FIELDS`. This means the embedding model never processes salary or DOB data. Even if the embedding were somehow extracted, it would not contain PII.

### Pre-Computed Aggregates

The `hr_aggregates.py` module generates five types of aggregates:

1. **Headcount by department:** Count of employees per department.
2. **Salary by department:** Average, min, max, count per department.
3. **Location distribution:** Employees per office location.
4. **Tenure statistics:** Average tenure by department.
5. **Role distribution:** Count of each job title.

Each aggregate is indexed as a separate chunk with its own access control. Headcount and location stats are broadly shareable. Salary and performance stats are restricted to HR and C-level even in aggregate form.

### Trade-off

Pre-computing aggregates at index time means they become stale if the CSV changes. The reindex endpoint rebuilds them. For a production system with frequent HR updates, you would want either scheduled reindexing or an event-driven pipeline.

---

## Decision 6: Pre-Computed Role Booleans in ChromaDB

### The Decision

Each chunk stored in ChromaDB has boolean metadata flags for each role:

```json
{
    "role_finance": true,
    "role_c_level": true,
    "role_marketing": false,
    "role_hr": false,
    "role_engineering": false,
    "role_employee": false
}
```

The query includes a `where` filter: `{"role_finance": true}`.

### Why

ChromaDB evaluates `where` filters server-side before computing cosine similarity. This means unauthorized chunks never leave the database. It is the fastest possible access control because the database does the work.

Without this, the retriever would fetch `top_k * N` results (over-fetching) and then filter client-side. Over-fetching wastes bandwidth and computation. With server-side filtering, the retriever asks for exactly `top_k` and gets `top_k` authorized results.

### What Was Considered

**Alternative: Role bitmap as integer.** Instead of 6 boolean fields, store a single integer where each bit represents a role (finance=8, c_level=32, etc.). ChromaDB does not support bitwise queries, so this was stored alongside the booleans for future migration to a database that supports bitwise operations (PostgreSQL, for example).

**Alternative: String list of allowed roles.** Store `allowed_roles: "finance,c_level"` and filter with `$contains`. ChromaDB's `$contains` works on strings, not arrays. A string search for "finance" would also match "finance_intern" if such a role existed. Boolean flags are exact.

### Trade-off

Six boolean fields per chunk use more storage than a single bitmap integer. For 203 chunks, this is negligible (about 1.2KB total overhead). For millions of chunks, bitmap storage would be better.

---

## Decision 7: Dual Index Strategy (Dense + Sparse)

### The Decision

Every chunk is indexed in both ChromaDB (dense, vector) and an in-memory BM25 index (sparse, keyword). At query time, both are searched and results are merged using Reciprocal Rank Fusion (RRF).

### Why

Dense and sparse search have complementary strengths:

| Query Type | Dense (Embedding) | Sparse (BM25) |
|-----------|-------------------|----------------|
| "What is the company revenue?" | Strong. Matches "annual income," "total sales." | Weak. Misses if "revenue" is not in the text. |
| "Q4 2024 financial report" | Weak. "Q4 2024" is a precise reference, not a semantic concept. | Strong. Exact keyword match. |
| "marketing ROI analysis" | Strong. Matches related concepts. | Medium. Matches if exact words appear. |

By combining both, the system handles both semantic and keyword queries well.

### RRF Fusion

Reciprocal Rank Fusion combines rankings without normalizing scores:

```python
score(doc) = sum(1 / (k + rank_in_list)) for each list
```

With `k=60` (standard), a document ranked #1 in dense and #3 in sparse gets:

```
score = 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323
```

A document ranked #1 in dense but absent from sparse gets:

```
score = 1/(60+1) = 0.0164
```

RRF naturally favors documents that appear in both rankings without needing to normalize dense cosine scores (0-1) against BM25 scores (0-infinity).

### Trade-off

Running two searches doubles the retrieval latency. Dense search: 20-50ms. Sparse search: 1-5ms. Total: 21-55ms. The sparse search is fast because it runs in-memory. The dense search is the bottleneck. For latency-sensitive applications, you could skip sparse search and rely on dense + PDP post-check.

---

## Decision 8: Chunk-Level Access Control (Not Document-Level)

### The Decision

Access control is computed per chunk, not per document. A document with a public introduction and a confidential appendix can have different access levels for different sections.

### Why

Consider `quarterly_financial_report.md`. It has:

- A summary section (safe for all finance users)
- A detailed revenue breakdown (safe for finance)
- A section about executive compensation (confidential, C-level only)

With document-level access control, the entire document is either allowed or denied. With chunk-level access control, the summary and revenue sections can be served to finance users, while the compensation section is restricted to C-level.

### How It Works

The `compute_chunk_allowed_roles()` function analyzes each chunk's text for sensitive keywords:

```python
SENSITIVE_KEYWORDS = [
    "salary", "compensation", "bonus", "payroll",
    "confidential", "restricted", "private",
    "performance review", "termination", "disciplinary",
]
```

If a chunk contains sensitive keywords, its access is tightened:
- `full_access` is restricted to the owner department and C-level.
- `masked_access` may be granted to finance (for salary/budget context).
- Other departments are denied.

### Trade-off

Keyword-based sensitivity detection is a heuristic. The word "salary" in "the salary for this position is competitive" triggers the restriction even though the sentence is generic. A production system might use an LLM classifier for more precise sensitivity detection. The current approach errs on the side of caution, which is the correct default for access control.

---

## Pipeline Summary

```
Document Discovery (directory-based ownership)
       |
       v
Markdown Chunking (heading-aware, section breadcrumbs)
       |
       v
Metadata Extraction (temporal: year/quarter, topics, sensitivity indicators)
       |
       v
Access Control (chunk-level, three tiers: full/masked/aggregate)
       |
       v
Context-Aware Embedding (security prefix prepended)
       |
       v
Dual Store (ChromaDB with role booleans + BM25 in-memory)
       |
       v
Query: Dense + Sparse -> RRF Fusion -> PDP Post-Check -> LLM Generation
```

### Key Properties

1. **Unauthorized data never reaches the embedding model** (PII fields are excluded from searchable text).
2. **Unauthorized data never leaves the vector database** (server-side role filtering in ChromaDB).
3. **Even if filtering fails, the PDP post-check blocks unauthorized results** (defense in depth).
4. **Cache keys include the user's permission fingerprint** (no cross-role cache leakage).
5. **Temporal metadata improves quarterly query accuracy** without requiring the user to specify filters manually.
6. **Context-aware embeddings improve cross-department disambiguation** without requiring separate collections per department.
