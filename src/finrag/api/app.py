"""The HTTP front door for registry discovery and composed RAG questions.

Thin on purpose. Every decision that matters happens below this file: the role
mask in `retrieval/base.py`, the grounding rules in `generate.py`, the rules in
`policy.yaml`. This module only turns a request into those calls and a result
into a status code.

Four status codes carry meaning:

  200  an answer, or an honest `reason: no_answer` with no answer text
  403  the policy gives this role nothing in this corpus, and says which rule did it
  422  the request itself is wrong: unknown role or strategy, empty question
  503  the request was fine but a selected model, embedding, or graph service is unavailable

THE ROLE COMES FROM THE REQUEST BODY. That is a demo shortcut and nothing else:
a real deployment would take it from a verified token, because a client that
picks its own role has no access control at all. It is written this way so the
ABAC behaviour can be exercised from a UI role picker without an identity
provider, and the whole point of the exercise is that the layers underneath do
not care where the role came from.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from finrag.config import Settings, load_env
from finrag.generate import (
    DEFAULT_K,
    Answerer,
    get_answerer,
)
from finrag.models import ROLES
from finrag.pipeline import StrategyUnavailableError
from finrag.querying.registry import DEFAULT_QUERY_STRATEGY
from finrag.reranking.registry import DEFAULT_RERANKER
from finrag.retrieval.registry import DEFAULT_RETRIEVER

# Long enough for a real question, short enough that nobody posts a document.
MAX_QUESTION_CHARS = 1000

# More than this many hits stops being a citation list and starts being a dump.
MAX_K = 25


class AskRequest(BaseModel):
    """One question through one independently selected strategy combination."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    role: str
    retrieval_strategy: str = DEFAULT_RETRIEVER
    query_strategy: str = DEFAULT_QUERY_STRATEGY
    reranking_strategy: str = DEFAULT_RERANKER
    k: int = Field(default=DEFAULT_K, ge=1, le=MAX_K)


def known_roles(answerer: Answerer) -> list[str]:
    """The roles this deployment accepts, from the policy when it has one.

    The policy file is the authority on which roles exist; `ROLES` in models.py
    is the fallback so an answerer built without a policy still validates input.
    """
    if answerer.policy is not None:
        return list(answerer.policy.roles)
    return list(ROLES)


def create_app(answerer: Answerer | None = None) -> FastAPI:
    """Build the app. Pass an `Answerer` to serve a test corpus offline.

    A factory rather than a module-level `app` so importing this module reads no
    artifacts and opens no client. `uvicorn finrag.api.app:create_app --factory`.
    """
    if answerer is None:
        answerer = _default_answerer()

    app = FastAPI(
        title="finrag",
        description="FinSolve ABAC RAG: role-masked retrieval with cited answers.",
        version="0.1.0",
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        """Is the process up, and what is it serving with."""
        return {
            "status": "ok",
            "chunks": len(answerer.chunks),
            "strategies": answerer.strategies(),
            "llm": answerer.llm_name(),
        }

    @app.get("/strategies")
    def strategies() -> dict[str, Any]:
        """The three registry catalogs, including defaults and requirements."""
        return answerer.strategies()

    @app.get("/roles")
    def roles() -> dict[str, Any]:
        """Roles the policy defines, for the UI role picker."""
        return {"roles": known_roles(answerer)}

    @app.post("/ask")
    def ask(request: AskRequest) -> dict[str, Any]:
        """Retrieve as this role and answer from the hits, or explain why not."""
        _check_role(answerer, request.role)
        _check_strategies(answerer, request)

        refusal = answerer.policy_refusal(request.role)
        if refusal is not None:
            # Not an error in the request: a correct answer from the policy, and
            # 403 is the honest way to say it.
            raise HTTPException(status_code=403, detail=refusal)

        try:
            return answerer.answer(
                request.question,
                request.role,
                retrieval_strategy=request.retrieval_strategy,
                query_strategy=request.query_strategy,
                reranking_strategy=request.reranking_strategy,
                k=request.k,
            )
        except StrategyUnavailableError as err:
            # 503, not 500: the request was fine, a dependency is down, and the
            # message says exactly how to bring it back.
            raise HTTPException(
                status_code=503,
                detail={"reason": "strategy_unavailable", "message": str(err)},
            ) from err

    return app


def _check_role(answerer: Answerer, role: str) -> None:
    """Refuse an unknown role loudly. Fail closed, never treat it as a guest."""
    allowed = known_roles(answerer)
    if role not in allowed:
        raise HTTPException(
            status_code=422,
            detail={
                "reason": "unknown_role",
                "role": role,
                "known_roles": allowed,
            },
        )


def _check_strategies(answerer: Answerer, request: AskRequest) -> None:
    """Validate each independent selector against its own registry."""
    fields = (
        ("retrieval_strategy", request.retrieval_strategy, answerer.retrieval_strategies()),
        ("query_strategy", request.query_strategy, answerer.query_strategies()),
        ("reranking_strategy", request.reranking_strategy, answerer.reranking_strategies()),
    )
    for field, selected, allowed in fields:
        if selected not in allowed:
            raise HTTPException(
                status_code=422,
                detail={
                    "reason": f"unknown_{field}",
                    field: selected,
                    "known_strategies": allowed,
                },
            )


def _default_answerer() -> Answerer:
    """The answerer over the real artifacts, with a plain message when they are missing."""
    # `finrag serve` reads .env at its own front door, but uvicorn started by hand
    # does not go through the CLI, so read it here too. Anything already in the
    # environment wins, so this cannot clobber a deliberate override.
    load_env()

    settings = Settings.from_env()
    if not settings.chunks_path.exists():
        raise RuntimeError(
            f"No chunks at {settings.chunks_path}. Run `finrag ingest` before serving."
        )
    return get_answerer()
