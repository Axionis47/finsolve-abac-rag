"""
Search routes (sparse, dense, hybrid).
"""
from __future__ import annotations
import time
from typing import Dict, Optional

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel

from app.routes.deps import authenticate, get_pdp, get_index, set_index
from app.utils.config import get_settings
from app.utils.audit import gen_correlation_id, log_event
# Rate limiting removed - use middleware instead if needed

from app.policy.pdp import PDP
from app.ingest.runner import build_index
from app.retrieval.search import search_index
from app.retrieval.filtering import prefilter_by_allowed_roles
from app.retrieval.bm25 import bm25_search
from app.retrieval.hybrid import rrf_fuse
from app.retrieval.rerank import rerank
from app.services.reranker_openai import openai_embedding_rerank, openai_llm_rerank
from app.vectorstore.chroma_store import get_client, get_or_create_collection, query as chroma_query
from app.services.embeddings import embed_texts

router = APIRouter(prefix="/search", tags=["search"])
SETTINGS = get_settings()


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    persist_dir: Optional[str] = None
    base_dir: Optional[str] = None
    rerank: bool = False
    reranker: Optional[str] = None  # "simple" | "openai" | "openai_llm"


@router.post("")
def search(req: SearchRequest, request: Request, user=Depends(authenticate)):
    """Simple keyword search."""
    index = get_index(request)
    if not index:
        index = build_index(base_dir="resources/data")
        set_index(request, index)
    
    results = search_index(index, req.query, role=user["role"], top_k=req.top_k, pdp=get_pdp(request))
    return {
        "count": len(results),
        "results": [
            {
                "text": r.get("text"),
                "source_path": r.get("source_path"),
                "section_path": r.get("section_path"),
                "owner_dept": r.get("owner_dept"),
                "doc_type": r.get("doc_type"),
                "sensitivity": r.get("sensitivity"),
            }
            for r in results
        ],
    }


@router.post("/dense")
def search_dense(req: SearchRequest, request: Request, response: Response, user=Depends(authenticate)):
    """Dense vector search via Chroma."""
    cid = gen_correlation_id()
    response.headers["X-Correlation-ID"] = cid
    log_event(cid, "search_dense.start", {"query": req.query, "role": user["role"]})
    t0 = time.perf_counter()
    
    persist_dir = req.persist_dir or SETTINGS.get("CHROMA_DB_DIR", ".chroma")
    client = get_client(persist_dir)
    col = get_or_create_collection(client, name="kb_main")

    t1 = time.perf_counter()
    q_emb = embed_texts([req.query])[0]
    t2 = time.perf_counter()
    candidates = chroma_query(col, q_emb, top_k=req.top_k, role=user["role"])

    # Optional rerank
    rerank_ms = 0.0
    if req.rerank:
        t_rr0 = time.perf_counter()
        backend = req.reranker or "simple"
        kwargs = _get_rerank_kwargs(backend)
        candidates = rerank(req.query, candidates, backend=backend, top_k=req.top_k, **kwargs)
        rerank_ms = (time.perf_counter() - t_rr0) * 1000

    # PDP post-check
    pdp: PDP = get_pdp(request)
    t3 = time.perf_counter()
    results = _filter_by_pdp(candidates, pdp, user["role"])
    t4 = time.perf_counter()

    resp = {
        "count": len(results),
        "results": results,
        "metrics": {
            "init_ms": (t1 - t0) * 1000,
            "embed_ms": (t2 - t1) * 1000,
            "retrieve_ms": (t3 - t2) * 1000,
            "pdp_ms": (t4 - t3) * 1000,
            "rerank_ms": rerank_ms,
            "total_ms": (t4 - t0) * 1000,
        },
        "correlation_id": cid,
    }
    log_event(cid, "search_dense.end", {"count": resp["count"], "metrics": resp["metrics"]})
    return resp


@router.post("/hybrid")
def search_hybrid(req: SearchRequest, request: Request, response: Response, user=Depends(authenticate)):
    """Hybrid search (dense + sparse + policy)."""
    cid = gen_correlation_id()
    response.headers["X-Correlation-ID"] = cid
    log_event(cid, "search_hybrid.start", {"query": req.query, "role": user["role"]})
    t0 = time.perf_counter()
    
    # Ensure sparse index
    index = get_index(request)
    if not index:
        base_dir = req.base_dir or "resources/data"
        index = build_index(base_dir=base_dir)
        set_index(request, index)

    # Dense candidates
    persist_dir = req.persist_dir or SETTINGS.get("CHROMA_DB_DIR", ".chroma")
    t1 = time.perf_counter()
    dense_cands = []
    try:
        client = get_client(persist_dir)
        col = get_or_create_collection(client, name="kb_main")
        q_emb = embed_texts([req.query])[0]
        dense_cands = chroma_query(col, q_emb, top_k=req.top_k, role=user["role"])
    except Exception:
        pass
    t2 = time.perf_counter()

    # Sparse candidates
    allowed_subset = prefilter_by_allowed_roles(index, user["role"])
    sparse_cands = bm25_search(allowed_subset, req.query, top_k=req.top_k)
    t3 = time.perf_counter()

    fused = rrf_fuse(dense_cands, sparse_cands, k=60, top_k=req.top_k)

    # Optional rerank
    rerank_ms = 0.0
    if req.rerank:
        t_rr0 = time.perf_counter()
        backend = req.reranker or "simple"
        kwargs = _get_rerank_kwargs(backend)
        fused = rerank(req.query, fused, backend=backend, top_k=req.top_k, **kwargs)
        rerank_ms = (time.perf_counter() - t_rr0) * 1000

    # PDP post-check
    pdp: PDP = get_pdp(request)
    t4 = time.perf_counter()
    results = _filter_by_pdp(fused, pdp, user["role"])
    t5 = time.perf_counter()

    resp = {
        "count": len(results),
        "results": results,
        "metrics": {
            "dense_ms": (t2 - t1) * 1000,
            "sparse_ms": (t3 - t2) * 1000,
            "fuse_ms": (t4 - t3) * 1000,
            "pdp_ms": (t5 - t4) * 1000,
            "rerank_ms": rerank_ms,
            "total_ms": (t5 - t0) * 1000,
            "dense_skipped": dense_cands == [],
        },
        "correlation_id": cid,
    }
    log_event(cid, "search_hybrid.end", {"count": resp["count"], "metrics": resp["metrics"]})
    return resp


def _get_rerank_kwargs(backend: str) -> Dict:
    """Get reranker kwargs based on backend type."""
    kwargs = {}
    if backend == "openai":
        kwargs["openai_rerank_fn"] = lambda q, items, top_k=None: openai_embedding_rerank(
            q, items, top_k=top_k,
            model=SETTINGS.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            api_key=SETTINGS.get("OPENAI_API_KEY", ""),
        )
    elif backend == "openai_llm":
        kwargs["openai_rerank_fn"] = lambda q, items, top_k=None: openai_llm_rerank(
            q, items, top_k=top_k,
            model=SETTINGS.get("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            api_key=SETTINGS.get("OPENAI_API_KEY", ""),
        )
    return kwargs


def _filter_by_pdp(candidates, pdp: PDP, role: str):
    """Filter candidates by PDP authorization."""
    results = []
    for c in candidates:
        res_md = {
            "owner_dept": c.get("owner_dept"),
            "doc_type": c.get("doc_type"),
            "sensitivity": c.get("sensitivity"),
        }
        d = pdp.evaluate(subject={"role": role}, resource=res_md, action="retrieve_chunk")
        if d.effect == "permit":
            results.append({
                "text": c.get("text"),
                "source_path": c.get("source_path"),
                "section_path": c.get("section_path"),
                **res_md,
                "decision": {"effect": d.effect, "rule": d.rule},
            })
    return results

