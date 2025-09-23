from __future__ import annotations
from typing import Dict, List


def prefilter_by_allowed_roles(items: List[Dict], role: str) -> List[Dict]:
    """Return only items where role is in allowed_roles."""
    out: List[Dict] = []
    for it in items:
        roles = it.get("allowed_roles") or []
        if role in roles:
            out.append(it)
    return out

