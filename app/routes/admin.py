"""
Admin routes (status, reindex, cache management).
"""
from __future__ import annotations
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel

from app.routes.deps import authenticate
from app.utils.config import get_settings
from app.utils.audit import gen_correlation_id
# Rate limiting removed - use middleware instead if needed

from app.ingest.runner import build_index
from app.ingest.dense_indexer import index_corpus_to_chroma
from app.vectorstore.chroma_store import get_client, get_or_create_collection
from app.services.openai_client import get_model_metadata
from app.services.cache import get_cache

router = APIRouter(prefix="/admin", tags=["admin"])
SETTINGS = get_settings()


class ReindexDenseRequest(BaseModel):
    base_dir: Optional[str] = None
    persist_dir: Optional[str] = None


def _require_c_level(user: dict):
    """Ensure user is c_level."""
    if user["role"] != "c_level":
        raise HTTPException(status_code=403, detail="Only c_level can access this endpoint")


@router.get("/status")
def admin_status(request: Request, response: Response, user=Depends(authenticate)):
    """Admin status - counts, health, cache stats. C-level only."""
    _require_c_level(user)
    cid = gen_correlation_id()
    response.headers["X-Correlation-ID"] = cid

    # Sparse count
    sparse_count = len(request.app.state.index) if isinstance(request.app.state.index, list) else 0

    # Dense count
    persist_dir = SETTINGS.get("CHROMA_DB_DIR", ".chroma")
    dense_count = 0
    try:
        client = get_client(persist_dir)
        col = get_or_create_collection(client, name="kb_main")
        if hasattr(col, "count"):
            dense_count = col.count()
    except Exception:
        pass

    # OpenAI health
    try:
        status, payload = get_model_metadata(
            SETTINGS.get("OPENAI_API_KEY", ""),
            SETTINGS.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        )
        model_id = payload.get("id") if isinstance(payload, dict) else None
        openai_status = {"ok": status == 200, "model": model_id, "status": status}
    except Exception as e:
        openai_status = {"ok": False, "error": str(e)[:120]}

    # Cache stats
    cache_stats = get_cache().stats()

    return {
        "sparse_count": sparse_count,
        "dense_count": dense_count,
        "persist_dir": persist_dir,
        "openai": openai_status,
        "cache": cache_stats,
        "correlation_id": cid,
    }


@router.post("/reindex")
def admin_reindex(request: Request, user=Depends(authenticate)):
    """Rebuild in-memory sparse index. C-level only."""
    _require_c_level(user)
    request.app.state.index = build_index(base_dir="resources/data")
    return {"status": "ok", "count": len(request.app.state.index)}


@router.post("/reindex_dense")
def admin_reindex_dense(req: ReindexDenseRequest = None, user=Depends(authenticate)):
    """Rebuild Chroma dense index. C-level only."""
    _require_c_level(user)
    base_dir = (req.base_dir if req else None) or "resources/data"
    persist_dir = (req.persist_dir if req else None) or SETTINGS.get("CHROMA_DB_DIR", ".chroma")
    count = index_corpus_to_chroma(base_dir=base_dir, persist_dir=persist_dir)
    return {"status": "ok", "count": count, "persist_dir": persist_dir}


@router.post("/cache/clear")
def admin_cache_clear(response: Response, user=Depends(authenticate)):
    """Clear LLM and embedding cache. C-level only."""
    _require_c_level(user)
    cid = gen_correlation_id()
    response.headers["X-Correlation-ID"] = cid
    cache = get_cache()
    old_stats = cache.stats()
    cache.clear()
    return {"status": "ok", "cleared": old_stats, "correlation_id": cid}

