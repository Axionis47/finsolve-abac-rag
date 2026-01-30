"""
Shared dependencies for route modules.
"""
from __future__ import annotations
import hashlib
import secrets
from typing import Dict

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic()


def _hash_password(password: str, salt: str = "") -> str:
    """Hash password with SHA256 + salt. For production, use bcrypt/argon2."""
    if not salt:
        salt = secrets.token_hex(16)
    hashed = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}${hashed}"


def _verify_password(password: str, stored: str) -> bool:
    """Verify password against stored hash."""
    if "$" not in stored:
        return password == stored
    salt, _ = stored.split("$", 1)
    return _hash_password(password, salt) == stored


# User database - plaintext for demo, use hashed passwords in production
# Format: username -> {password, role}
users_db: Dict[str, Dict[str, str]] = {
    "tony": {"password": "password123", "role": "engineering"},
    "bruce": {"password": "securepass", "role": "marketing"},
    "sam": {"password": "financepass", "role": "finance"},
    "peter": {"password": "pete123", "role": "engineering"},
    "sid": {"password": "sidpass123", "role": "marketing"},
    "natasha": {"password": "hrpass123", "role": "hr"},
    "clark": {"password": "chief", "role": "c_level"},
}


def authenticate(credentials: HTTPBasicCredentials = Depends(security)) -> Dict[str, str]:
    """Authentication dependency that returns user info (case-insensitive username)."""
    username = credentials.username.lower()
    password = credentials.password
    user = users_db.get(username)
    if not user or not _verify_password(password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"username": username, "role": user["role"]}


def get_pdp(request: Request):
    """Get PDP from app state."""
    return request.app.state.pdp


def get_index(request: Request):
    """Get search index from app state."""
    return request.app.state.index


def set_index(request: Request, index):
    """Set search index in app state."""
    request.app.state.index = index


def authorize(request: Request, action: str, resource: Dict[str, str], user: Dict[str, str]):
    """Check authorization via PDP."""
    import logging
    from app.policy.pdp import PDP
    pdp: PDP = get_pdp(request)
    decision = pdp.evaluate(subject={"role": user["role"]}, resource=resource, action=action)
    if decision.effect != "permit":
        logging.info(
            "PDP deny: user=%s role=%s action=%s resource=%s rule=%s",
            user.get("username"), user["role"], action, resource, decision.rule,
        )
        raise HTTPException(status_code=403, detail="Access denied")
    return decision

