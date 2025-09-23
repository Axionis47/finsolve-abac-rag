from __future__ import annotations
from typing import Dict, List

ALL_ROLES = ["finance", "marketing", "hr", "engineering", "c_level", "employee"]


def compute_allowed_roles(md: Dict) -> List[str]:
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

