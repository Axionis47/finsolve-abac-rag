from __future__ import annotations
from typing import Dict, List
from pathlib import Path

from app.ingest.discovery import discover_resources
from app.ingest.chunker import chunk_markdown
from app.ingest.allow_roles import compute_allowed_roles


def build_index(base_dir: str = "resources/data") -> List[Dict]:
    """
    Build an in-memory index of chunks with metadata and allowed_roles.
    Currently indexes Markdown files; CSV/aggregate are skipped for retrieval.
    Returns a list of dicts with fields: text, owner_dept, doc_type, sensitivity,
    source_path, section_path, allowed_roles
    """
    resources = discover_resources(base_dir=base_dir)
    index: List[Dict] = []
    for res in resources:
        doc_type = res.get("doc_type")
        if doc_type == "md":
            p = Path(res["source_path"])  # type: ignore
            if not p.exists():
                continue
            text = p.read_text(encoding="utf-8", errors="ignore")
            chunks = chunk_markdown(text, source_path=str(p))
            for ch in chunks:
                md = {
                    "owner_dept": res.get("owner_dept"),
                    "doc_type": "md",
                    "sensitivity": res.get("sensitivity"),
                    "source_path": ch["source_path"],
                    "section_path": ch["section_path"],
                }
                allowed = compute_allowed_roles(md)
                index.append({
                    **md,
                    "text": ch["text"],
                    "allowed_roles": allowed,
                })
        else:
            # Skip csv/aggregate for retrieval index in this minimal version
            continue
    return index

