from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional

from app.ingest.discovery import discover_resources
from app.ingest.chunker import chunk_markdown
from app.ingest.allow_roles import compute_allowed_roles
from app.services.embeddings import embed_texts
from app.vectorstore.chroma_store import get_client, get_or_create_collection, upsert_chunks


def index_corpus_to_chroma(base_dir: str, persist_dir: Optional[str] = None,
                           embedding_model: Optional[str] = None, api_key: Optional[str] = None) -> int:
    base = Path(base_dir)
    if not base.exists():
        return 0

    items = discover_resources(base_dir)
    chunks: List[Dict[str, Any]] = []

    for it in items:
        # Resolve path robustly whether discover_resources returned absolute or base-prefixed relative paths
        p_candidate = Path(it["source_path"])  # may be relative (already base-prefixed) or absolute
        p = p_candidate if p_candidate.exists() else (base / it["source_path"])
        if it.get("doc_type") == "md" and p.exists():
            text = p.read_text(encoding="utf-8", errors="ignore")
            md_common = {
                "source_path": str(p),
                "owner_dept": it.get("owner_dept"),
                "doc_type": it.get("doc_type"),
                "sensitivity": it.get("sensitivity"),
            }
            allowed = compute_allowed_roles(md_common)
            for ch in chunk_markdown(text, source_path=str(p)):
                chunks.append({**md_common, **ch, "allowed_roles": allowed})
        # CSV retrieval not indexed in dense path

    if not chunks:
        return 0

    texts = [c.get("text", "") for c in chunks]
    embs = embed_texts(texts, model=embedding_model, api_key=api_key)

    client = get_client(persist_dir)
    col = get_or_create_collection(client, name="kb_main")
    count = upsert_chunks(col, chunks, embs)
    return count

