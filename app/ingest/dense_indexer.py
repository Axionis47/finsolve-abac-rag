from __future__ import annotations
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from app.ingest.discovery import discover_resources
from app.ingest.chunker import chunk_markdown
from app.ingest.allow_roles import compute_allowed_roles
from app.ingest.hr_csv_indexer import index_hr_csv
from app.ingest.hr_aggregates import generate_hr_aggregates
from app.services.providers import embed_texts
from app.vectorstore.chroma_store import get_client, get_or_create_collection, upsert_chunks


def create_contextual_embedding_text(chunk: Dict[str, Any]) -> str:
    """
    Create embedding input that includes security and document context.

    This prepends metadata tags to the chunk text so that embeddings
    capture security context, not just semantic content. Two chunks with
    the same text but different departments will have different embeddings.

    Args:
        chunk: Chunk dictionary with text and metadata

    Returns:
        Contextual text for embedding generation
    """
    context_parts = []

    # Department context
    if chunk.get("owner_dept"):
        context_parts.append(f"[DEPARTMENT: {chunk['owner_dept'].upper()}]")

    # Sensitivity context
    if chunk.get("sensitivity"):
        context_parts.append(f"[SENSITIVITY: {chunk['sensitivity'].upper()}]")

    # Document type context
    if chunk.get("doc_type"):
        context_parts.append(f"[TYPE: {chunk['doc_type'].upper()}]")

    # Temporal context
    if chunk.get("quarter"):
        context_parts.append(f"[QUARTER: {chunk['quarter']}]")
    if chunk.get("year"):
        context_parts.append(f"[YEAR: {chunk['year']}]")

    # Document name for context
    if chunk.get("source_path"):
        doc_name = os.path.basename(chunk["source_path"]).replace("_", " ").replace(".md", "")
        context_parts.append(f"[DOCUMENT: {doc_name}]")

    context_prefix = " ".join(context_parts) + "\n\n" if context_parts else ""
    return context_prefix + chunk.get("text", "")


def index_corpus_to_chroma(
    base_dir: str,
    persist_dir: Optional[str] = None,
    use_contextual_embeddings: bool = True,
    policy_flags: Optional[Dict[str, str]] = None,
) -> int:
    """
    Index all markdown documents to ChromaDB using the configured embedding backend.

    Args:
        base_dir: Base directory containing documents to index
        persist_dir: Directory for ChromaDB persistence (None for ephemeral)
        use_contextual_embeddings: If True, prepend security context to embedding text
        policy_flags: Policy flags dict (e.g., {"hr_aggregate_mode": "enabled"})

    Uses the EMBEDDING_BACKEND env var to determine provider (vertex, openai, etc.)
    """
    base = Path(base_dir)
    if not base.exists():
        return 0

    policy_flags = policy_flags or {}
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

        # HR CSV indexing - index employee records with PII protection
        elif it.get("doc_type") == "csv" and it.get("owner_dept") == "hr" and p.exists():
            hr_chunks = index_hr_csv(str(p), policy_flags)
            chunks.extend(hr_chunks)

            # Also generate HR aggregates for broader access
            agg_chunks = generate_hr_aggregates(str(p), policy_flags)
            chunks.extend(agg_chunks)

    if not chunks:
        return 0

    # Generate embeddings - optionally with security context
    if use_contextual_embeddings:
        texts = [create_contextual_embedding_text(c) for c in chunks]
    else:
        texts = [c.get("text", "") for c in chunks]

    # Use provider abstraction - respects EMBEDDING_BACKEND env var
    embs = embed_texts(texts, use_cache=True)

    client = get_client(persist_dir)
    col = get_or_create_collection(client, name="kb_main")
    count = upsert_chunks(col, chunks, embs)
    return count

