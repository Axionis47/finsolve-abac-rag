"""
HR CSV Indexer with PII Awareness.

Indexes HR employee data while:
- Protecting PII fields (salary, DOB, email, etc.)
- Creating searchable text from non-PII fields
- Generating masked versions for limited access
- Supporting policy-based access tiers
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.ingest.allow_roles import compute_chunk_allowed_roles, flatten_access_tiers


# PII fields that should not be in searchable text or exposed to non-HR roles
PII_FIELDS = [
    "email",
    "salary",
    "date_of_birth",
    "manager_id",
    "leave_balance",
    "leaves_taken",
    "attendance_pct",
    "performance_rating",
    "last_review_date",
]

# Non-PII fields safe for search text generation
SEARCHABLE_FIELDS = [
    "full_name",
    "role",
    "department",
    "location",
    "date_of_joining",
]


def index_hr_csv(
    csv_path: str,
    policy_flags: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Index HR CSV rows as searchable chunks with PII awareness.

    Each row becomes a chunk with:
    - Searchable text (non-PII fields only)
    - Masked text (for limited access roles)
    - Full row data (stored for HR/C-level retrieval)
    - Tiered access control

    Args:
        csv_path: Path to HR CSV file
        policy_flags: Policy flags (e.g., {"hr_masked_rows_mode": "enabled"})

    Returns:
        List of chunk dictionaries ready for indexing
    """
    policy_flags = policy_flags or {}
    chunks: List[Dict[str, Any]] = []

    path = Path(csv_path)
    if not path.exists():
        return chunks

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Create searchable text from non-PII fields only
            searchable_text = create_searchable_text(row)

            # Create masked version for limited access
            masked_text = create_masked_text(row)

            # Build chunk with all metadata
            chunk: Dict[str, Any] = {
                "text": searchable_text,
                "masked_text": masked_text,
                "row_id": row.get("employee_id", ""),
                "source_path": csv_path,
                "section_path": f"Employee Record > {row.get('employee_id', 'Unknown')}",
                "owner_dept": "hr",
                "doc_type": "csv_row",
                "sensitivity": "restricted",
                "pii_fields": PII_FIELDS,
                # Store full row for HR/C-level access (as JSON-safe dict)
                "full_row": dict(row),
            }

            # Compute tiered access
            access_tiers = compute_chunk_allowed_roles(chunk, policy_flags)

            # Store access tiers as comma-separated strings
            chunk["full_access_roles"] = ",".join(sorted(access_tiers.get("full_access", [])))
            chunk["masked_access_roles"] = ",".join(sorted(access_tiers.get("masked_access", [])))
            chunk["aggregate_access_roles"] = ",".join(sorted(access_tiers.get("aggregate_access", [])))

            # Flatten for backward-compatible allowed_roles
            chunk["allowed_roles"] = flatten_access_tiers(access_tiers)

            chunks.append(chunk)

    return chunks


def create_searchable_text(row: Dict[str, str]) -> str:
    """
    Create searchable text from non-PII fields.

    This text is used for embedding generation and search matching.
    It excludes PII to prevent unauthorized information disclosure.

    Args:
        row: CSV row dictionary

    Returns:
        Searchable text string
    """
    parts = []

    if row.get("full_name"):
        parts.append(f"Employee: {row['full_name']}")

    if row.get("role"):
        parts.append(f"Role: {row['role']}")

    if row.get("department"):
        parts.append(f"Department: {row['department']}")

    if row.get("location"):
        parts.append(f"Location: {row['location']}")

    if row.get("date_of_joining"):
        parts.append(f"Joined: {row['date_of_joining']}")

    return "\n".join(parts)


def create_masked_text(row: Dict[str, str]) -> str:
    """
    Create masked text for limited access roles.

    This version reveals only department and role,
    suitable for users with masked_access tier.

    Args:
        row: CSV row dictionary

    Returns:
        Masked text string
    """
    dept = row.get("department", "Unknown")
    role = row.get("role", "Unknown")
    location = row.get("location", "")

    if location:
        return f"Employee in {dept} department as {role} based in {location}"
    return f"Employee in {dept} department as {role}"


def get_row_for_role(chunk: Dict[str, Any], role: str) -> Dict[str, Any]:
    """
    Get appropriate row data based on role's access level.

    Args:
        chunk: Indexed chunk dictionary
        role: User's role

    Returns:
        Dictionary with appropriate data for the role
    """
    # Check access tiers
    full_access = chunk.get("full_access_roles", "").split(",")
    masked_access = chunk.get("masked_access_roles", "").split(",")

    if role in full_access:
        # Full access - return complete row
        return {
            "access_level": "full",
            "text": chunk.get("text", ""),
            "full_row": chunk.get("full_row", {}),
        }
    elif role in masked_access:
        # Masked access - return masked text only
        return {
            "access_level": "masked",
            "text": chunk.get("masked_text", ""),
            "full_row": None,
        }
    else:
        # No access
        return {
            "access_level": "denied",
            "text": None,
            "full_row": None,
        }
