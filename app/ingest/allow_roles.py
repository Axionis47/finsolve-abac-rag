"""
Access Control Role Computation for RAG Indexing.

Computes allowed roles for chunks at index time, supporting:
- Document-level access (original behavior)
- Chunk-level tiered access (full/masked/aggregate)
- Policy flag integration (hr_aggregate_mode, hr_masked_rows_mode)
- Bitmap encoding for efficient storage
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# All possible roles in the system
ALL_ROLES = ["finance", "marketing", "hr", "engineering", "c_level", "employee"]

# Keywords indicating sensitive content that needs extra protection
SENSITIVE_KEYWORDS = [
    "salary", "salaries", "compensation", "bonus", "payroll",
    "remuneration", "pay grade", "pay scale", "total comp",
    "annual earnings", "base pay", "stock option", "equity grant",
    "confidential", "restricted", "private", "classified",
    "performance review", "termination", "disciplinary",
    "severance", "probation", "written warning",
    "ssn", "social security", "bank account", "tax id",
    "date of birth", "passport", "national id",
]

# Role bitmap encoding for efficient storage
ROLE_BITS = {
    "employee": 0b000001,     # 1
    "engineering": 0b000010,  # 2
    "marketing": 0b000100,    # 4
    "finance": 0b001000,      # 8
    "hr": 0b010000,           # 16
    "c_level": 0b100000,      # 32
}


def roles_to_bitmap(roles: List[str]) -> int:
    """Convert role list to bitmap integer."""
    return sum(ROLE_BITS.get(r, 0) for r in roles)


def bitmap_to_roles(bitmap: int) -> List[str]:
    """Convert bitmap back to role list."""
    return [role for role, bit in ROLE_BITS.items() if bitmap & bit]


def check_role_access(bitmap: int, role: str) -> bool:
    """Check if role has access via bitmap."""
    return bool(bitmap & ROLE_BITS.get(role, 0))


def compute_allowed_roles(md: Dict) -> List[str]:
    """
    Original document-level access control computation.

    Kept for backward compatibility. For new chunk-level access,
    use compute_chunk_allowed_roles() instead.

    Args:
        md: Metadata dict with owner_dept, doc_type, sensitivity

    Returns:
        List of roles allowed to access the document
    """
    owner = md.get("owner_dept")
    doc_type = md.get("doc_type")
    sensitivity = md.get("sensitivity")

    if owner == "general":
        return ALL_ROLES.copy()

    if doc_type == "md" and sensitivity in ("general", "internal"):
        return [owner, "c_level"]

    if doc_type == "csv" and owner == "hr":
        return ["hr", "c_level"]

    if doc_type == "aggregate" and owner == "hr":
        return ["hr", "c_level"]  # runtime flag may include others

    # Default least-privilege: no roles
    return []


def contains_sensitive_content(text: str) -> bool:
    """Check if text contains sensitive keywords."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in SENSITIVE_KEYWORDS)


def compute_chunk_allowed_roles(
    chunk: Dict[str, Any],
    policy_flags: Optional[Dict[str, str]] = None,
) -> Dict[str, List[str]]:
    """
    Compute tiered access control per chunk.

    This provides fine-grained access control where different roles
    can see different versions of the same chunk:
    - full_access: Can see complete content
    - masked_access: Can see content with PII redacted
    - aggregate_access: Can only see aggregated statistics

    Args:
        chunk: Chunk dictionary with text and metadata
        policy_flags: Policy flags (e.g., {"hr_aggregate_mode": "enabled"})

    Returns:
        Dict with access tiers:
        {
            "full_access": [...],      # Can see everything
            "masked_access": [...],    # Can see with PII redacted
            "aggregate_access": [...], # Can see aggregated stats only
        }
    """
    policy_flags = policy_flags or {}
    owner = chunk.get("owner_dept", "")
    doc_type = chunk.get("doc_type", "")
    sensitivity = chunk.get("sensitivity", "internal")
    text = chunk.get("text", "")

    # Check for sensitive content in the chunk text
    has_sensitive = contains_sensitive_content(text)

    access_tiers: Dict[str, List[str]] = {
        "full_access": [],
        "masked_access": [],
        "aggregate_access": [],
    }

    # C-level always has full access to everything
    access_tiers["full_access"].append("c_level")

    # General documents - everyone has full access
    if owner == "general":
        access_tiers["full_access"].extend(
            ["finance", "marketing", "hr", "engineering", "employee"]
        )
        return access_tiers

    # Owner department always has full access to their documents
    if owner and owner not in access_tiers["full_access"]:
        access_tiers["full_access"].append(owner)

    # HR CSV row handling - most restricted
    if doc_type == "csv_row" and owner == "hr":
        # Only HR and C-level get full row access
        access_tiers["full_access"] = ["hr", "c_level"]

        # If masked mode enabled, others get masked access (no PII)
        if policy_flags.get("hr_masked_rows_mode") == "enabled":
            access_tiers["masked_access"].extend(
                ["finance", "marketing", "engineering"]
            )

        return access_tiers

    # HR aggregates handling
    if doc_type == "aggregate" and owner == "hr":
        # HR and C-level always get full access to aggregates
        access_tiers["full_access"] = ["hr", "c_level"]

        # If aggregate mode enabled, other departments can see aggregates
        if policy_flags.get("hr_aggregate_mode") == "enabled":
            # Headcount and location stats are broadly accessible
            agg_type = chunk.get("agg_type", "")
            if agg_type in ("headcount_by_department", "location_distribution"):
                access_tiers["full_access"].extend(
                    ["finance", "marketing", "engineering"]
                )
            else:
                # Salary and performance aggregates need aggregate_access tier
                access_tiers["aggregate_access"].extend(
                    ["finance", "marketing", "engineering"]
                )

        return access_tiers

    # Sensitive content in markdown - restrict further
    if has_sensitive and doc_type == "md":
        # Only owner and C-level for sensitive chunks
        access_tiers["full_access"] = [owner, "c_level"] if owner else ["c_level"]

        # Finance might get masked access to salary/budget discussions
        if any(kw in text.lower() for kw in ["salary", "compensation", "budget"]):
            if "finance" not in access_tiers["full_access"]:
                access_tiers["masked_access"].append("finance")

        return access_tiers

    # Standard markdown with no sensitive content
    # Owner + C-level have full access (already set above)
    return access_tiers


def flatten_access_tiers(tiers: Dict[str, List[str]]) -> List[str]:
    """
    Flatten access tiers to list of all roles with any access level.

    Args:
        tiers: Access tier dictionary

    Returns:
        List of all roles that have at least some access
    """
    all_roles = set()
    for tier_roles in tiers.values():
        all_roles.update(tier_roles)
    all_roles.discard("")  # Remove empty strings
    return sorted(all_roles)


def get_access_level(tiers: Dict[str, List[str]], role: str) -> Optional[str]:
    """
    Get the access level for a specific role.

    Args:
        tiers: Access tier dictionary
        role: Role to check

    Returns:
        Access level ("full_access", "masked_access", "aggregate_access")
        or None if no access
    """
    for level in ["full_access", "masked_access", "aggregate_access"]:
        if role in tiers.get(level, []):
            return level
    return None
