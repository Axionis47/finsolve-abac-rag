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
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
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
from app.routes.metrics import router as metrics_router
from app.routes.evaluation import router as evaluation_router
from app.routes.deps import authenticate

# Load environment and reset any cached providers/settings
load_env(override=False)
from app.utils.config import reset_settings
from app.services.providers import reset_providers
reset_settings()  # Clear settings singleton to pick up .env values
reset_providers()  # Clear provider cache to use fresh settings
SETTINGS = get_settings()

# Configure logging
logger = logging.getLogger(__name__)

# Active request tracking for graceful shutdown
_active_requests = 0
_shutdown_event = asyncio.Event()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for graceful startup/shutdown."""
    # Startup
    logger.info("Application starting up...")
    yield
    # Shutdown
    logger.info("Initiating graceful shutdown...")
    _shutdown_event.set()

    # Wait for active requests to complete (max 30s)
    shutdown_timeout = 30
    for i in range(shutdown_timeout * 2):
        if _active_requests == 0:
            logger.info("All requests completed, shutting down")
            break
        if i % 10 == 0:
            logger.info(f"Waiting for {_active_requests} active requests...")
        await asyncio.sleep(0.5)
    else:
        logger.warning(f"Shutdown timeout: {_active_requests} requests still active")

    logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="ABAC RAG Chatbot",
    description="Internal chatbot with attribute-based access control",
    version="2.0.0",
    docs_url="/docs" if os.getenv("ENABLE_DOCS", "true").lower() == "true" else None,
    lifespan=lifespan,
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS configuration
# Set CORS_ORIGINS env var to comma-separated list of allowed origins
# Example: CORS_ORIGINS=https://app.example.com,https://admin.example.com
cors_origins = os.getenv("CORS_ORIGINS", "")
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in cors_origins.split(",") if origin.strip()],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# Request tracking middleware for graceful shutdown
@app.middleware("http")
async def track_requests(request: Request, call_next):
    global _active_requests
    _active_requests += 1
    try:
        response = await call_next(request)
        return response
    finally:
        _active_requests -= 1


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
app.include_router(metrics_router)
app.include_router(evaluation_router)


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
