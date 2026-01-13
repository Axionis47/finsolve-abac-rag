"""
Rate Limiting Configuration

Uses slowapi for request rate limiting.
Limits are configurable via environment variables.
"""
from __future__ import annotations
import os
from typing import Optional

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address


def get_user_identifier(request: Request) -> str:
    """
    Get rate limit key from authenticated user or IP.
    Uses username if authenticated, otherwise falls back to IP.
    """
    # Try to get username from request state (set by auth middleware)
    if hasattr(request.state, "user") and request.state.user:
        return f"user:{request.state.user.get('username', get_remote_address(request))}"
    
    # Try Authorization header for basic auth
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Basic "):
        import base64
        try:
            decoded = base64.b64decode(auth[6:]).decode("utf-8")
            username = decoded.split(":")[0]
            return f"user:{username}"
        except Exception:
            pass
    
    # Fallback to IP
    return get_remote_address(request)


# Rate limit defaults (can be overridden via env vars)
DEFAULT_RATE_LIMIT = os.getenv("RATE_LIMIT_DEFAULT", "100/minute")
CHAT_RATE_LIMIT = os.getenv("RATE_LIMIT_CHAT", "20/minute")
SEARCH_RATE_LIMIT = os.getenv("RATE_LIMIT_SEARCH", "60/minute")
ADMIN_RATE_LIMIT = os.getenv("RATE_LIMIT_ADMIN", "10/minute")

# Create limiter instance
limiter = Limiter(
    key_func=get_user_identifier,
    default_limits=[DEFAULT_RATE_LIMIT],
    storage_uri=os.getenv("RATE_LIMIT_STORAGE", "memory://"),
)


def get_limiter() -> Limiter:
    """Get the rate limiter instance."""
    return limiter

