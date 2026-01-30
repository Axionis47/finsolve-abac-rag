from __future__ import annotations
from typing import Dict, List
from pathlib import Path

from app.ingest.discovery import discover_resources
from app.ingest.chunker import chunk_markdown
from app.ingest.allow_roles import compute_allowed_roles
from app.ingest.hr_csv_indexer import index_hr_csv
from app.ingest.hr_aggregates import generate_hr_aggregates


def build_index(base_dir: str = "resources/data") -> List[Dict]:
    """
    Build an in-memory index of chunks with metadata and allowed_roles.

    Indexes Markdown files, HR CSV employee records, and HR aggregates.
    HR chunks use tiered access control with allowed_roles flattened
    from full/masked/aggregate tiers so BM25 pre-filtering works.

    Returns a list of dicts with fields: text, owner_dept, doc_type, sensitivity,
    source_path, section_path, allowed_roles
    """
    resources = discover_resources(base_dir=base_dir)
    index: List[Dict] = []

    for res in resources:
        doc_type = res.get("doc_type")
        p = Path(res["source_path"])
        if not p.exists():
            continue

        if doc_type == "md":
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

        elif doc_type == "csv" and res.get("owner_dept") == "hr":
            # Index HR CSV rows for keyword search (BM25)
            hr_chunks = index_hr_csv(str(p))
            for ch in hr_chunks:
                index.append({
                    "text": ch["text"],
                    "owner_dept": ch.get("owner_dept", "hr"),
                    "doc_type": ch.get("doc_type", "csv_row"),
                    "sensitivity": ch.get("sensitivity", "restricted"),
                    "source_path": ch.get("source_path", str(p)),
                    "section_path": ch.get("section_path", ""),
                    "allowed_roles": ch.get("allowed_roles", ["hr", "c_level"]),
                })

            # Index HR aggregates for keyword search (BM25)
            agg_chunks = generate_hr_aggregates(str(p))
            for ch in agg_chunks:
                allowed = ch.get("allowed_roles", ["hr", "c_level"])
                index.append({
                    "text": ch["text"],
                    "owner_dept": ch.get("owner_dept", "hr"),
                    "doc_type": ch.get("doc_type", "aggregate"),
                    "sensitivity": ch.get("sensitivity", "internal"),
                    "source_path": ch.get("source_path", "hr_aggregates"),
                    "section_path": ch.get("section_path", ""),
                    "allowed_roles": allowed,
                })

    return index
