"""One small httpx wrapper the three pages share.

The UI holds no logic of its own: it never masks, ranks or decides anything, it
only calls the API and draws what came back. That is deliberate. If a page could
compute an answer, there would be two implementations of the rules and one of
them would be wrong.

Errors are returned, not raised, so a page can show a 403 as the policy speaking
rather than as a crash.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

API_BASE_ENV = "FINRAG_API"
DEFAULT_BASE = "http://127.0.0.1:8000"

# Generation can take a few seconds; retrieval never does.
TIMEOUT_SECONDS = 60.0


def base_url() -> str:
    """Where the API is. Override with FINRAG_API when it is not local."""
    return os.environ.get(API_BASE_ENV, DEFAULT_BASE).rstrip("/")


def get_json(path: str) -> dict[str, Any]:
    """GET one endpoint. Returns {"ok": bool, "status": int, "body": ...}."""
    try:
        response = httpx.get(f"{base_url()}{path}", timeout=TIMEOUT_SECONDS)
    except httpx.HTTPError as error:
        return _unreachable(error)
    return {"ok": response.is_success, "status": response.status_code, "body": _body(response)}


def ask(question: str, role: str, mode: str, k: int) -> dict[str, Any]:
    """POST /ask. A 403 or 422 comes back as ok=False with the reason in body."""
    payload = {"question": question, "role": role, "mode": mode, "k": k}
    try:
        response = httpx.post(f"{base_url()}/ask", json=payload, timeout=TIMEOUT_SECONDS)
    except httpx.HTTPError as error:
        return _unreachable(error)
    return {"ok": response.is_success, "status": response.status_code, "body": _body(response)}


def modes(fallback: list[str]) -> list[str]:
    """The retrieval modes the API offers, or `fallback` when it is down."""
    result = get_json("/modes")
    if result["ok"]:
        return list(result["body"]["modes"])
    return list(fallback)


def roles(fallback: list[str]) -> list[str]:
    """The roles the API accepts, or `fallback` when it is down."""
    result = get_json("/roles")
    if result["ok"]:
        return list(result["body"]["roles"])
    return list(fallback)


def _body(response: httpx.Response) -> Any:
    """The JSON body, or the raw text when the reply was not JSON."""
    try:
        return response.json()
    except ValueError:
        return {"detail": response.text}


def _unreachable(error: Exception) -> dict[str, Any]:
    """The API is not running. Say so with the fix, not with a stack trace."""
    return {
        "ok": False,
        "status": 0,
        "body": {
            "detail": (
                f"Cannot reach the API at {base_url()}: {error}. "
                "Start it with `finrag serve`."
            )
        },
    }
