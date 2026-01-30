# Indexing Strategy Improvements for ABAC RAG

## Current State Assessment

| Area | Score | Issue |
|------|-------|-------|
| HR Data Coverage | ❌ | CSV not indexed |
| Access Granularity | ⚠️ | Document-level only |
| Metadata Richness | ⚠️ | Missing temporal/topics |
| Policy Compliance | ⚠️ | Static roles, flags ignored |
| Embedding Context | ⚠️ | No security awareness |

---

## Priority 1: Index HR CSV Data

### Current Problem
```python
# dense_indexer.py line ~50
if it.get("doc_type") == "md":  # ← Skips CSV!
    chunks = chunk_markdown(...)
```

### Recommended Fix
```python
def index_hr_csv(csv_path: str, policy_flags: dict) -> List[dict]:
    """Index HR CSV with PII awareness."""
    import csv

    chunks = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Determine access based on policy flags
            allowed_roles = ["hr", "c_level"]

            # If aggregate mode enabled, create summary chunks
            if policy_flags.get("hr_aggregate_mode") == "enabled":
                allowed_roles.extend(["finance", "marketing", "engineering"])

            # Create searchable text (non-PII fields)
            searchable_text = f"""
            Employee: {row['full_name']}
            Department: {row['department']}
            Role: {row['role']}
            Hire Date: {row['date_of_hire']}
            """

            # Store with metadata
            chunk = {
                "text": searchable_text,
                "row_id": row['employee_id'],
                "source_path": csv_path,
                "section_path": f"Employee Record > {row['employee_id']}",
                "owner_dept": "hr",
                "doc_type": "csv_row",
                "sensitivity": "restricted",
                "allowed_roles": allowed_roles,
                "pii_fields": ["email", "salary", "date_of_birth", "manager_id"],
                # Masked version for limited access
                "masked_text": f"Employee in {row['department']} as {row['role']}",
            }
            chunks.append(chunk)

    return chunks
```

---

## Priority 2: Extract Temporal Metadata

### Current Problem
Files named `marketing_report_q1_2024.md` have no extracted quarter/year.

### Recommended Fix
```python
import re

def extract_temporal_metadata(text: str, filename: str) -> dict:
    """Extract year and quarter from content and filename."""

    # From filename
    quarter_match = re.search(r'q([1-4])[-_]?(\d{4})', filename, re.I)
    year_match = re.search(r'(20\d{2})', filename)

    # From content
    content_quarters = re.findall(r'\b(Q[1-4])\b', text, re.I)
    content_years = re.findall(r'\b(202[0-9])\b', text)

    return {
        "primary_quarter": quarter_match.group(1) if quarter_match else None,
        "primary_year": quarter_match.group(2) if quarter_match else (year_match.group(1) if year_match else None),
        "mentioned_quarters": list(set(content_quarters)),
        "mentioned_years": list(set(content_years)),
    }
```

### Update Chunker
```python
# In chunker.py
def chunk_markdown(...):
    temporal = extract_temporal_metadata(text, source_path)

    for chunk in chunks:
        chunk["year"] = temporal["primary_year"]
        chunk["quarter"] = temporal["primary_quarter"]
        chunk["temporal_refs"] = temporal["mentioned_quarters"]
```

---

## Priority 3: Context-Aware Embeddings

### Current Problem
Two chunks with same text but different access levels get identical embeddings.

### Recommended Fix
```python
def create_contextual_embedding_text(chunk: dict) -> str:
    """Create embedding input that includes security context."""

    # Prepend metadata for semantic differentiation
    context_prefix = f"""
[DEPARTMENT: {chunk['owner_dept']}]
[SENSITIVITY: {chunk['sensitivity']}]
[DOCUMENT: {chunk['source_path'].split('/')[-1]}]
"""

    return context_prefix + chunk['text']

# In dense_indexer.py
def index_chunks(chunks: List[dict]):
    # Use contextual text for embedding
    texts_for_embedding = [
        create_contextual_embedding_text(c) for c in chunks
    ]

    embeddings = embed_texts(texts_for_embedding)
    # Store original text, but embed with context
```

---

## Priority 4: Fine-Grained Access Control

### Current Problem
```python
# allow_roles.py - all chunks in doc get same roles
def compute_allowed_roles(owner, doc_type, sensitivity):
    return [owner, "c_level"]  # Too coarse!
```

### Recommended Fix
```python
def compute_chunk_allowed_roles(
    chunk: dict,
    policy_rules: list,
    policy_flags: dict
) -> dict:
    """Compute access per chunk, not per document."""

    # Check for sensitive content markers
    sensitive_keywords = ["salary", "compensation", "confidential", "restricted"]
    contains_sensitive = any(kw in chunk['text'].lower() for kw in sensitive_keywords)

    # Start with base roles from owner
    base_roles = [chunk['owner_dept'], "c_level"]

    # Apply policy rules
    access_tiers = {
        "full_access": base_roles.copy(),
        "masked_access": [],
        "aggregate_access": [],
        "denied": [],
    }

    # If contains sensitive content, restrict further
    if contains_sensitive:
        access_tiers["full_access"] = ["hr", "c_level"]
        access_tiers["masked_access"] = ["finance"]

    # Apply flag-based expansions
    if policy_flags.get("hr_aggregate_mode") == "enabled":
        access_tiers["aggregate_access"].extend(["finance", "marketing"])

    return access_tiers
```

---

## Priority 5: Efficient Role Filtering

### Current Problem
```python
# 6 boolean metadata fields per chunk
{"role_finance": True, "role_marketing": False, ...}  # Wasteful
```

### Recommended Fix
```python
# Use bitmap instead
ROLE_BITS = {
    "employee": 0b000001,
    "engineering": 0b000010,
    "marketing": 0b000100,
    "finance": 0b001000,
    "hr": 0b010000,
    "c_level": 0b100000,
}

def roles_to_bitmap(roles: list) -> int:
    return sum(ROLE_BITS.get(r, 0) for r in roles)

def bitmap_to_roles(bitmap: int) -> list:
    return [role for role, bit in ROLE_BITS.items() if bitmap & bit]

# Store single integer instead of 6 booleans
chunk_metadata = {
    "allowed_roles_bitmap": roles_to_bitmap(["finance", "c_level"]),  # = 0b101000 = 40
}

# Query with bitwise AND
# ChromaDB: where={"allowed_roles_bitmap": {"$bitor": ROLE_BITS["finance"]}}
```

---

## Priority 6: HR Aggregates Pre-Indexing

### When `hr_aggregate_mode` is enabled, pre-compute and index aggregates

```python
def generate_hr_aggregates(hr_csv_path: str) -> List[dict]:
    """Generate aggregate statistics for indexing."""
    import pandas as pd

    df = pd.read_csv(hr_csv_path)
    aggregates = []

    # Average salary by department
    dept_salary = df.groupby('department')['salary'].mean()
    aggregates.append({
        "text": f"Average salary by department: {dept_salary.to_dict()}",
        "agg_type": "avg_salary_by_dept",
        "source_path": "hr_aggregates",
        "section_path": "Salary Statistics > By Department",
        "owner_dept": "hr",
        "doc_type": "aggregate",
        "sensitivity": "internal",  # Lower than raw data
        "allowed_roles": ["finance", "marketing", "engineering", "c_level"],
    })

    # Headcount by department
    headcount = df.groupby('department').size()
    aggregates.append({
        "text": f"Employee headcount by department: {headcount.to_dict()}",
        "agg_type": "headcount_by_dept",
        "source_path": "hr_aggregates",
        "section_path": "Headcount Statistics > By Department",
        "owner_dept": "hr",
        "doc_type": "aggregate",
        "sensitivity": "internal",
        "allowed_roles": ["finance", "marketing", "engineering", "hr", "c_level"],
    })

    return aggregates
```

---

## Implementation Checklist

- [ ] **Phase 1**: Add HR CSV indexing with PII awareness
- [ ] **Phase 2**: Extract temporal metadata (year, quarter)
- [ ] **Phase 3**: Implement context-aware embeddings
- [ ] **Phase 4**: Chunk-level access control with tiers
- [ ] **Phase 5**: Bitmap-based role filtering
- [ ] **Phase 6**: Pre-compute HR aggregates when flag enabled
- [ ] **Phase 7**: Add policy version tracking in metadata
- [ ] **Phase 8**: Implement change detection via checksums

---

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| HR data in RAG | 0% | 100% |
| Temporal filtering | Not possible | Q1-Q4, 2024 supported |
| Access granularity | Document-level | Chunk/field-level |
| Policy compliance | Partial | Full flag support |
| Embedding security | No context | Security-aware |
