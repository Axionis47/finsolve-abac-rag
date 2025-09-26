"""
Simple overview (by me)
- I built a FastAPI app that answers questions from company docs using a policy (ABAC) and RAG.
- Every request goes through policy checks. I only pass allowed snippets to the LLM and I always cite sources.
- I also expose small HR CSV endpoints with strict rules.
This docstring is written in simple Indian English.
"""

from typing import Dict, Optional
import os
import time

from fastapi import FastAPI, HTTPException, Depends, Response, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse


from app.utils.config import load_env, get_settings
from app.services.openai_client import get_model_metadata
from app.utils.audit import gen_correlation_id, log_event

from app.policy.pdp import PDP
from app.ingest.runner import build_index
from app.retrieval.search import search_index
from app.retrieval.filtering import prefilter_by_allowed_roles
from app.retrieval.bm25 import bm25_search
from app.retrieval.hybrid import rrf_fuse
from app.retrieval.rerank import rerank
from app.services.reranker_openai import openai_embedding_rerank, openai_llm_rerank
from app.hr.csv_service import load_hr_rows, compute_safe_aggregates, mask_row
from app.ingest.dense_indexer import index_corpus_to_chroma
from app.vectorstore.chroma_store import get_client, get_or_create_collection, query as chroma_query
from app.services.embeddings import embed_texts
from app.services.generation import generate_answer



class AuthzRequest(BaseModel):
    resource: Dict[str, str]
    action: str

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    persist_dir: Optional[str] = None
    base_dir: Optional[str] = None
    rerank: bool = False
    reranker: Optional[str] = None  # "simple" | "openai" | "openai_llm"

class ChatRequest(BaseModel):
    message: str
    top_k: int = 5
    persist_dir: Optional[str] = None
    base_dir: Optional[str] = None
    rerank: bool = False
    reranker: Optional[str] = None  # "simple" | "openai" | "openai_llm"

class ReindexDenseRequest(BaseModel):
    base_dir: Optional[str] = None
    persist_dir: Optional[str] = None



# Load environment from .env without overriding existing env vars
load_env(override=False)
SETTINGS = get_settings()


app = FastAPI()
# Load PDP once at startup
POLICY_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "policy.yaml"))
# Templates and static assets for simple UI
templates = Jinja2Templates(directory="templates")
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")



app.state.pdp = PDP.load(POLICY_PATH)

app.state.index = []

security = HTTPBasic()

# Dummy user database
users_db: Dict[str, Dict[str, str]] = {
    "Tony": {"password": "password123", "role": "engineering"},
    "Bruce": {"password": "securepass", "role": "marketing"},
    "Sam": {"password": "financepass", "role": "finance"},
    "Peter": {"password": "pete123", "role": "engineering"},
    "Sid": {"password": "sidpass123", "role": "marketing"},
    "Natasha": {"password": "hrpass123", "role": "hr"},
    "Clark": {"password": "chief", "role": "c_level"}
}


def authorize(action: str, resource: Dict[str, str], user: Dict[str, str]):
    pdp: PDP = app.state.pdp
    decision = pdp.evaluate(subject={"role": user["role"]}, resource=resource, action=action)
    if decision.effect != "permit":
        raise HTTPException(status_code=403, detail={"decision": decision.effect, "rule": decision.rule})
    return {"decision": decision.effect, "rule": decision.rule}


# Authentication dependency
def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username
    password = credentials.password
    user = users_db.get(username)
    if not user or user["password"] != password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"username": username, "role": user["role"]}


# Login endpoint
@app.get("/login")
def login(user=Depends(authenticate)):
    return {"message": f"Welcome {user['username']}!", "role": user["role"]}


# Protected test endpoint
@app.get("/test")
def test(user=Depends(authenticate)):
    return {"message": f"Hello {user['username']}! You can now chat.", "role": user["role"]}


# Health check: validates OpenAI access without incurring token usage

# Simple chat UI (requires HTTP Basic auth)
@app.get("/", response_class=HTMLResponse)
def home(request: Request, response: Response, user=Depends(authenticate)):
    cid = gen_correlation_id()
    response.headers["X-Correlation-ID"] = cid
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "user": user,
        "correlation_id": cid,
    })

@app.get("/health/openai")
def health_openai():
    model_id = SETTINGS.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    api_key = SETTINGS.get("OPENAI_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    status, payload = get_model_metadata(api_key, model_id)
    if status == 200 and isinstance(payload, dict) and payload.get("id") == model_id:
        return {"status": "ok", "model": model_id}
    raise HTTPException(status_code=502, detail={"status": status, "error": payload})

# PDP-backed authorization check endpoint (dev/testing)

# Admin status (counts + health) — c_level only
@app.get("/admin/status")
def admin_status(response: Response, user=Depends(authenticate)):
    if user["role"] != "c_level":
        raise HTTPException(status_code=403, detail="Only c_level can view status")
    cid = gen_correlation_id()
    response.headers["X-Correlation-ID"] = cid
    # Sparse
    sparse_count = len(app.state.index) if isinstance(app.state.index, list) else 0
    # Dense
    persist_dir = SETTINGS.get("CHROMA_DB_DIR", ".chroma")
    dense_count = 0
    try:
        client = get_client(persist_dir)
        col = get_or_create_collection(client, name="kb_main")
        if hasattr(col, "count"):
            dense_count = col.count()
    except Exception:
        dense_count = 0
    # OpenAI health
    try:
        meta = get_model_metadata(SETTINGS.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
        openai = {"ok": True, "model": meta.get("id")}
    except Exception as e:
        openai = {"ok": False, "error": str(e)[:120]}
    return {"sparse_count": sparse_count, "dense_count": dense_count, "persist_dir": persist_dir, "openai": openai, "correlation_id": cid}

@app.post("/authz/check")
def authz_check(req: AuthzRequest, user=Depends(authenticate)):
    return authorize(action=req.action, resource=req.resource, user=user)


# Admin reindex (build in-memory index)
@app.post("/admin/reindex")
def admin_reindex(user=Depends(authenticate)):
    if user["role"] != "c_level":
        raise HTTPException(status_code=403, detail="Only c_level can reindex")
    app.state.index = build_index(base_dir="resources/data")
    return {"status": "ok", "count": len(app.state.index)}

# Admin reindex (dense, Chroma)
@app.post("/admin/reindex_dense")
def admin_reindex_dense(req: ReindexDenseRequest | None = None, user=Depends(authenticate)):
    if user["role"] != "c_level":
        raise HTTPException(status_code=403, detail="Only c_level can reindex dense store")
    base_dir = (req.base_dir if req else None) or "resources/data"
    persist_dir = (req.persist_dir if req else None) or SETTINGS.get("CHROMA_DB_DIR", ".chroma")
    count = index_corpus_to_chroma(base_dir=base_dir, persist_dir=persist_dir)
    return {"status": "ok", "count": count, "persist_dir": persist_dir}


# Dense search endpoint (Chroma + OpenAI embeddings)
@app.post("/search/dense")
def search_dense(req: SearchRequest, response: Response, user=Depends(authenticate)):
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
    candidates = chroma_query(col, q_emb, top_k=req.top_k, role=user["role"])  # server-side + client-side role filter

    # Optional rerank (client-side)
    rerank_ms = 0.0
    if req.rerank:
        t_rr0 = time.perf_counter()
        backend = req.reranker or "simple"
        kwargs = {}
        if backend == "openai":
            kwargs["openai_rerank_fn"] = lambda q, items, top_k=None: openai_embedding_rerank(
                q,
                items,
                top_k=top_k,
                model=SETTINGS.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
                api_key=SETTINGS.get("OPENAI_API_KEY", ""),
            )
        elif backend == "openai_llm":
            kwargs["openai_rerank_fn"] = lambda q, items, top_k=None: openai_llm_rerank(
                q,
                items,
                top_k=top_k,
                model=SETTINGS.get("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
                api_key=SETTINGS.get("OPENAI_API_KEY", ""),
            )
        candidates = rerank(req.query, candidates, backend=backend, top_k=req.top_k, **kwargs)
        t_rr1 = time.perf_counter()
        rerank_ms = (t_rr1 - t_rr0) * 1000

    # PDP post-check for defense in depth
    pdp: PDP = app.state.pdp
    t3 = time.perf_counter()
    results = []
    for c in candidates:
        res_md = {
            "owner_dept": c.get("owner_dept"),
            "doc_type": c.get("doc_type"),
            "sensitivity": c.get("sensitivity"),
        }
        d = pdp.evaluate(subject={"role": user["role"]}, resource=res_md, action="retrieve_chunk")
        if d.effect == "permit":
            results.append({
                "text": c.get("text"),
                "source_path": c.get("source_path"),
                "section_path": c.get("section_path"),
                **res_md,
                "decision": {"effect": d.effect, "rule": d.rule},
            })
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

# Hybrid search endpoint (RRF over dense + sparse)
@app.post("/search/hybrid")
def search_hybrid(req: SearchRequest, response: Response, user=Depends(authenticate)):
    """Hybrid search (dense + sparse + policy).
    I return PDP-permitted snippets, metrics, and a correlation id.
    Simple Indian English so it is easy to understand.
    """
    cid = gen_correlation_id()
    response.headers["X-Correlation-ID"] = cid
    log_event(cid, "search_hybrid.start", {"query": req.query, "role": user["role"]})
    t0 = time.perf_counter()
    # Ensure we have an in-memory sparse index
    if not app.state.index:
        base_dir = req.base_dir or "resources/data"
        app.state.index = build_index(base_dir=base_dir)

    # Dense candidates from Chroma
    persist_dir = req.persist_dir or SETTINGS.get("CHROMA_DB_DIR", ".chroma")
    t1 = time.perf_counter()
    client = get_client(persist_dir)
    col = get_or_create_collection(client, name="kb_main")
    q_emb = embed_texts([req.query])[0]
    dense_cands = chroma_query(col, q_emb, top_k=req.top_k, role=user["role"])  # server + client role filter
    t2 = time.perf_counter()

    # Sparse candidates from in-memory BM25 on allowed subset
    allowed_subset = prefilter_by_allowed_roles(app.state.index, user["role"])
    sparse_cands = bm25_search(allowed_subset, req.query, top_k=req.top_k)
    t3 = time.perf_counter()

    fused = rrf_fuse(dense_cands, sparse_cands, k=60, top_k=req.top_k)

    # Optional rerank on fused list
    rerank_ms = 0.0
    if req.rerank:
        t_rr0 = time.perf_counter()
        backend = req.reranker or "simple"
        kwargs = {}
        if backend == "openai":
            kwargs["openai_rerank_fn"] = lambda q, items, top_k=None: openai_embedding_rerank(
                q,
                items,
                top_k=top_k,
                model=SETTINGS.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
                api_key=SETTINGS.get("OPENAI_API_KEY", ""),
            )
        elif backend == "openai_llm":
            kwargs["openai_rerank_fn"] = lambda q, items, top_k=None: openai_llm_rerank(
                q,
                items,
                top_k=top_k,
                model=SETTINGS.get("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
                api_key=SETTINGS.get("OPENAI_API_KEY", ""),
            )
        fused = rerank(req.query, fused, backend=backend, top_k=req.top_k, **kwargs)
        t_rr1 = time.perf_counter()
        rerank_ms = (t_rr1 - t_rr0) * 1000

    # PDP post-check
    pdp: PDP = app.state.pdp
    results = []
    t4 = time.perf_counter()
    for c in fused:
        res_md = {
            "owner_dept": c.get("owner_dept"),
            "doc_type": c.get("doc_type"),
            "sensitivity": c.get("sensitivity"),
        }
        d = pdp.evaluate(subject={"role": user["role"]}, resource=res_md, action="retrieve_chunk")
        if d.effect == "permit":
            results.append({
                "text": c.get("text"),
                "source_path": c.get("source_path"),
                "section_path": c.get("section_path"),
                **res_md,
                "decision": {"effect": d.effect, "rule": d.rule},
            })
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
        },
        "correlation_id": cid,
    }
    log_event(cid, "search_hybrid.end", {"count": resp["count"], "metrics": resp["metrics"]})
    return resp


# HR CSV endpoints
@app.get("/hr/rows")
def hr_rows(limit: int = 10, offset: int = 0, response: Response = None, user=Depends(authenticate)):
    cid = gen_correlation_id()
    if response is not None:
        response.headers["X-Correlation-ID"] = cid
    log_event(cid, "hr_rows.start", {"limit": limit, "offset": offset, "role": user["role"]})
    # Authorize row-level access (HR/C-level only per policy)
    resource = {"owner_dept": "hr", "doc_type": "csv"}
    authorize(action="view_row", resource=resource, user=user)
    rows = load_hr_rows(limit=limit, offset=offset)
    resp = {"count": len(rows), "rows": rows, "correlation_id": cid}
    log_event(cid, "hr_rows.end", {"count": resp["count"]})
    return resp


@app.get("/hr/aggregate")
def hr_aggregate(response: Response = None, user=Depends(authenticate)):
    cid = gen_correlation_id()
    if response is not None:
        response.headers["X-Correlation-ID"] = cid
    log_event(cid, "hr_aggregate.start", {"role": user["role"]})
    # PDP with flags controlling non-HR access to aggregates
    flags = {"hr_aggregate_mode": SETTINGS.get("HR_AGGREGATE_MODE", "disabled")}
    resource = {"owner_dept": "hr", "doc_type": "aggregate"}
    pdp: PDP = app.state.pdp
    d = pdp.evaluate(subject={"role": user["role"]}, resource=resource, action="aggregate_query", flags=flags)
    if d.effect != "permit":
        raise HTTPException(status_code=403, detail={"decision": d.effect, "rule": d.rule}, headers={"X-Correlation-ID": cid})
    agg = compute_safe_aggregates()
    resp = {"decision": {"effect": d.effect, "rule": d.rule}, "aggregates": agg, "correlation_id": cid}
    log_event(cid, "hr_aggregate.end", {"decision": resp["decision"]})
    return resp


@app.get("/hr/rows_masked")
def hr_rows_masked(limit: int = 10, offset: int = 0, response: Response = None, user=Depends(authenticate)):
    cid = gen_correlation_id()
    if response is not None:
        response.headers["X-Correlation-ID"] = cid
    log_event(cid, "hr_rows_masked.start", {"limit": limit, "offset": offset, "role": user["role"]})
    # Policy-driven: allow HR/C-level always; allow others if HR_MASKED_ROWS_MODE=enabled
    flags = {"hr_masked_rows_mode": SETTINGS.get("HR_MASKED_ROWS_MODE", "disabled")}
    resource = {"owner_dept": "hr", "doc_type": "csv"}
    pdp: PDP = app.state.pdp
    d = pdp.evaluate(subject={"role": user["role"]}, resource=resource, action="view_masked_row", flags=flags)
    if d.effect != "permit":
        raise HTTPException(status_code=403, detail={"decision": d.effect, "rule": d.rule}, headers={"X-Correlation-ID": cid})
    rows = load_hr_rows(limit=limit, offset=offset)
    masked = [mask_row(r, level="strict") for r in rows]
    resp = {"decision": {"effect": d.effect, "rule": d.rule}, "count": len(masked), "rows": masked, "correlation_id": cid}
    log_event(cid, "hr_rows_masked.end", {"count": resp["count"]})
    return resp






# Simple search endpoint (keyword-based)
@app.post("/search")
def search(req: SearchRequest, user=Depends(authenticate)):
    if not app.state.index:
        app.state.index = build_index(base_dir="resources/data")
    results = search_index(app.state.index, req.query, role=user["role"], top_k=req.top_k, pdp=app.state.pdp)
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



# Chat endpoint: retrieval + PDP + LLM synthesis with citations
@app.post("/chat")
def chat(req: ChatRequest, response: Response, user=Depends(authenticate)):
    """Chat endpoint.
    I retrieve (dense+sparse), apply policy, and then ask the LLM to answer only from allowed context.
    I always send citations and timings, with a correlation id.
    """
    cid = gen_correlation_id()
    response.headers["X-Correlation-ID"] = cid
    log_event(cid, "chat.start", {"role": user["role"]})

    t0 = time.perf_counter()
    # Ensure sparse index
    if not app.state.index:
        base_dir = req.base_dir or "resources/data"
        app.state.index = build_index(base_dir=base_dir)

    # Dense via Chroma
    persist_dir = req.persist_dir or SETTINGS.get("CHROMA_DB_DIR", ".chroma")
    client = get_client(persist_dir)
    col = get_or_create_collection(client, name="kb_main")

    # Embed query
    t1 = time.perf_counter()
    q_emb = embed_texts([req.message])[0]
    t2 = time.perf_counter()

    # Retrieve
    dense_cands = chroma_query(col, q_emb, top_k=req.top_k, role=user["role"])  # role filter
    allowed_subset = prefilter_by_allowed_roles(app.state.index, user["role"])  # for BM25
    sparse_cands = bm25_search(allowed_subset, req.message, top_k=req.top_k)
    t3 = time.perf_counter()

    fused = rrf_fuse(dense_cands, sparse_cands, k=60, top_k=req.top_k)

    # Optional rerank
    rerank_ms = 0.0
    if req.rerank:
        t_rr0 = time.perf_counter()
        backend = req.reranker or "simple"
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
        fused = rerank(req.message, fused, backend=backend, top_k=req.top_k, **kwargs)
        t_rr1 = time.perf_counter()
        rerank_ms = (t_rr1 - t_rr0) * 1000

    # PDP filtering and context build
    pdp: PDP = app.state.pdp
    t4 = time.perf_counter()
    context_snippets = []
    citations = []
    for c in fused:
        res_md = {
            "owner_dept": c.get("owner_dept"),
            "doc_type": c.get("doc_type"),
            "sensitivity": c.get("sensitivity"),
        }
        d = pdp.evaluate(subject={"role": user["role"]}, resource=res_md, action="retrieve_chunk")
        if d.effect == "permit":
            context_snippets.append({
                "text": c.get("text"),
                "source_path": c.get("source_path"),
                "section_path": c.get("section_path"),
            })
            citations.append({
                "source_path": c.get("source_path"),
                "section_path": c.get("section_path"),
                "rule": d.rule,
            })
        if len(context_snippets) >= req.top_k:
            break
    t5 = time.perf_counter()

    # Generate answer
    t_llm0 = time.perf_counter()
    answer = generate_answer(
        req.message,
        context_snippets,
        model=SETTINGS.get("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        api_key=SETTINGS.get("OPENAI_API_KEY", ""),
    )
    t_llm1 = time.perf_counter()

    resp = {
        "answer": answer,
        "citations": citations,
        "metrics": {
            "embed_ms": (t2 - t1) * 1000,
            "retrieve_ms": (t3 - t2) * 1000,
            "rerank_ms": rerank_ms,
            "pdp_ms": (t5 - t4) * 1000,
            "llm_ms": (t_llm1 - t_llm0) * 1000,
            "total_ms": (t_llm1 - t0) * 1000,
        },
        "correlation_id": cid,
    }
    log_event(cid, "chat.end", {"citations": len(citations), "metrics": resp["metrics"]})
    return resp