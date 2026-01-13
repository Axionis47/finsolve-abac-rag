"""
Chat routes with streaming support.
"""
from __future__ import annotations
import json
import os
import ssl
import time
import urllib.request
from typing import Optional

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.routes.deps import authenticate, get_pdp, get_index, set_index
from app.utils.config import get_settings
from app.utils.audit import gen_correlation_id, log_event
# Rate limiting removed - use middleware instead if needed

from app.policy.pdp import PDP
from app.ingest.runner import build_index
from app.retrieval.filtering import prefilter_by_allowed_roles
from app.retrieval.bm25 import bm25_search
from app.retrieval.hybrid import rrf_fuse
from app.retrieval.rerank import rerank
from app.services.reranker_openai import openai_embedding_rerank, openai_llm_rerank
from app.vectorstore.chroma_store import get_client, get_or_create_collection, query as chroma_query
from app.services.embeddings import embed_texts
from app.services.generation import generate_answer

router = APIRouter(tags=["chat"])
SETTINGS = get_settings()


class ChatRequest(BaseModel):
    message: str
    top_k: int = 5
    persist_dir: Optional[str] = None
    base_dir: Optional[str] = None
    rerank: bool = False
    reranker: Optional[str] = None


def _retrieve_context(request: Request, message: str, user: dict, top_k: int, 
                      persist_dir: str, base_dir: str, do_rerank: bool, reranker: str):
    """Retrieve and filter context snippets."""
    # Ensure sparse index
    index = get_index(request)
    if not index:
        index = build_index(base_dir=base_dir)
        set_index(request, index)

    # Dense via Chroma
    client = get_client(persist_dir)
    col = get_or_create_collection(client, name="kb_main")

    q_emb = None
    try:
        q_emb = embed_texts([message])[0]
    except Exception:
        pass

    dense_cands = []
    if q_emb is not None:
        try:
            dense_cands = chroma_query(col, q_emb, top_k=top_k, role=user["role"])
        except Exception:
            pass

    # Sparse
    allowed_subset = prefilter_by_allowed_roles(index, user["role"])
    sparse_cands = bm25_search(allowed_subset, message, top_k=top_k)

    fused = rrf_fuse(dense_cands, sparse_cands, k=60, top_k=top_k)

    # Optional rerank
    if do_rerank:
        backend = reranker or "simple"
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
        fused = rerank(message, fused, backend=backend, top_k=top_k, **kwargs)

    # PDP filtering
    pdp: PDP = get_pdp(request)
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
        if len(context_snippets) >= top_k:
            break

    return context_snippets, citations


@router.post("/chat")
def chat(req: ChatRequest, request: Request, response: Response, user=Depends(authenticate)):
    """Chat endpoint with RAG retrieval and LLM synthesis."""
    cid = gen_correlation_id()
    response.headers["X-Correlation-ID"] = cid
    log_event(cid, "chat.start", {"role": user["role"]})

    t0 = time.perf_counter()
    persist_dir = req.persist_dir or SETTINGS.get("CHROMA_DB_DIR", ".chroma")
    base_dir = req.base_dir or "resources/data"

    t1 = time.perf_counter()
    context_snippets, citations = _retrieve_context(
        request, req.message, user, req.top_k, persist_dir, base_dir, req.rerank, req.reranker
    )
    t2 = time.perf_counter()

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
            "retrieve_ms": (t2 - t1) * 1000,
            "llm_ms": (t_llm1 - t_llm0) * 1000,
            "total_ms": (t_llm1 - t0) * 1000,
        },
        "correlation_id": cid,
    }
    log_event(cid, "chat.end", {"citations": len(citations), "metrics": resp["metrics"]})
    return resp


def _build_chat_messages(query: str, snippets: list) -> list:
    """Build chat messages for the LLM."""
    system = (
        "You are a careful assistant. Answer ONLY using the provided context snippets. "
        "Cite sources in-line as [#] and include a final 'Citations' section listing source and section. "
        "If the answer is not in the context, reply: 'I don't know based on the available context.'"
    )
    lines = ["Query:", query, "\nContext snippets:"]
    for i, s in enumerate(snippets, start=1):
        txt = (s.get("text") or "").strip().replace("\n", " ")
        src = s.get("source_path") or ""
        sec = s.get("section_path") or ""
        lines.append(f"[{i}] {txt}\n(Source: {src}#{sec})")
    user_msg = "\n".join(lines)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]


def _get_api_base() -> str:
    """Get the appropriate API base URL."""
    backend = os.getenv("LLM_BACKEND", "openai")
    if backend == "vllm":
        return os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    elif backend == "ollama":
        return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/v1"
    return "https://api.openai.com/v1"


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request, response: Response, user=Depends(authenticate)):
    """
    Streaming chat endpoint using Server-Sent Events (SSE).
    Returns tokens as they are generated for better UX.
    """
    cid = gen_correlation_id()
    response.headers["X-Correlation-ID"] = cid
    log_event(cid, "chat_stream.start", {"role": user["role"]})

    persist_dir = req.persist_dir or SETTINGS.get("CHROMA_DB_DIR", ".chroma")
    base_dir = req.base_dir or "resources/data"

    # Retrieve context
    context_snippets, citations = _retrieve_context(
        request, req.message, user, req.top_k, persist_dir, base_dir, req.rerank, req.reranker
    )

    api_key = SETTINGS.get("OPENAI_API_KEY", "")
    model = SETTINGS.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    async def generate():
        # Send metadata first
        yield f"data: {json.dumps({'type': 'metadata', 'citations': citations, 'correlation_id': cid})}\n\n"

        if not api_key:
            # Fallback for no API key
            if not context_snippets:
                no_context_msg = "I don't know based on the available context."
                yield f"data: {json.dumps({'type': 'token', 'content': no_context_msg})}\n\n"
            else:
                top = context_snippets[0]
                src_path = top.get('source_path', '')
                sec_path = top.get('section_path', '')
                src = f"{src_path}#{sec_path}".strip('#')
                text_preview = top.get('text', '')[:200]
                fallback = f"Based on [1], {text_preview}...\n\nCitations: [1] {src}"
                yield f"data: {json.dumps({'type': 'token', 'content': fallback})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        # Stream from LLM
        url = f"{_get_api_base()}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {
            "model": model,
            "messages": _build_chat_messages(req.message, context_snippets),
            "temperature": 0.2,
            "stream": True,
        }

        req_obj = urllib.request.Request(url, method="POST", data=json.dumps(payload).encode("utf-8"))
        for k, v in headers.items():
            req_obj.add_header(k, v)
        context = ssl.create_default_context()

        try:
            with urllib.request.urlopen(req_obj, context=context, timeout=120) as resp:
                for line in resp:
                    line = line.decode("utf-8").strip()
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)[:100]})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        log_event(cid, "chat_stream.end", {"citations": len(citations)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Correlation-ID": cid,
        }
    )

