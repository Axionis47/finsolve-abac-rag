"""
RAG Chatbot with ABAC Policy Engine

A FastAPI application that provides:
- Role-based access control via YAML-driven ABAC policy
- Hybrid search (dense + sparse) with RRF fusion
- LLM-powered answer generation with citations
- HR data endpoints with PII protection

Routes are organized into modules for maintainability.
"""
from __future__ import annotations
import os
from typing import Dict

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.utils.config import load_env, get_settings
from app.utils.rate_limit import limiter
from app.policy.pdp import PDP
from app.services.openai_client import get_model_metadata

# Import route modules
from app.routes.auth import router as auth_router
from app.routes.search import router as search_router
from app.routes.chat import router as chat_router
from app.routes.admin import router as admin_router
from app.routes.hr import router as hr_router
from app.routes.deps import authenticate

# Load environment
load_env(override=False)
SETTINGS = get_settings()

# Create FastAPI app
app = FastAPI(
    title="ABAC RAG Chatbot",
    description="Internal chatbot with attribute-based access control",
    version="2.0.0",
    docs_url="/docs" if os.getenv("ENABLE_DOCS", "true").lower() == "true" else None,
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
    )
    return response


# Load PDP at startup
POLICY_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "policy.yaml"))
app.state.pdp = PDP.load(POLICY_PATH)
app.state.index = []

# Mount static files
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Include route modules
app.include_router(auth_router)
app.include_router(search_router)
app.include_router(chat_router)
app.include_router(admin_router)
app.include_router(hr_router)


# Health check endpoints
@app.get("/health")
def health():
    """Basic health check."""
    return {"status": "ok"}


@app.get("/health/openai")
def health_openai():
    """Validate OpenAI access."""
    model_id = SETTINGS.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    api_key = SETTINGS.get("OPENAI_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    status, payload = get_model_metadata(api_key, model_id)
    if status == 200 and isinstance(payload, dict) and payload.get("id") == model_id:
        return {"status": "ok", "model": model_id}
    raise HTTPException(status_code=502, detail={"status": status, "error": payload})


# Authorization check endpoint (for testing)
class AuthzRequest(BaseModel):
    resource: Dict[str, str]
    action: str


@app.post("/authz/check")
def authz_check(req: AuthzRequest, user=Depends(authenticate)):
    """Check authorization via PDP."""
    pdp: PDP = app.state.pdp
    decision = pdp.evaluate(subject={"role": user["role"]}, resource=req.resource, action=req.action)
    if decision.effect != "permit":
        raise HTTPException(status_code=403, detail={"decision": decision.effect, "rule": decision.rule})
    return {"decision": decision.effect, "rule": decision.rule}
