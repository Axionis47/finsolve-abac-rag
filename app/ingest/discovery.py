from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, List

DEPTS = {"engineering", "finance", "marketing", "hr", "general"}


def infer_doc_type(path: Path) -> str:
    suf = path.suffix.lower()
    if suf == ".md":
        return "md"
    if suf == ".csv":
        return "csv"
    return "unknown"


def default_sensitivity(owner_dept: str, doc_type: str) -> str:
    if owner_dept == "general":
        return "general"
    if doc_type == "csv" and owner_dept == "hr":
        return "restricted"
    return "internal"


def discover_resources(base_dir: str = "resources/data") -> List[Dict]:
    base = Path(base_dir)
    items: List[Dict] = []
    if not base.exists():
        return items

    for dept_dir in base.iterdir():
        if not dept_dir.is_dir():
            continue
        owner_dept = dept_dir.name
        if owner_dept not in DEPTS:
            continue
        for p in dept_dir.rglob("*"):
            if not p.is_file():
                continue
            doc_type = infer_doc_type(p)
            if doc_type == "unknown":
                continue
            md = {
                "source_path": str(p),
                "owner_dept": owner_dept,
                "doc_type": doc_type,
                "sensitivity": default_sensitivity(owner_dept, doc_type),
            }
            items.append(md)
    return items

