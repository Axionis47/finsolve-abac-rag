"""Contracts shared by direct, HyDE and ReAct query strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from finrag.models import Hit
from finrag.retrieval.base import Retriever


@dataclass(frozen=True)
class QueryRequest:
    """One question and the candidate depth the downstream stages need."""

    question: str
    role: str
    k: int


@dataclass(frozen=True)
class QueryTrace:
    """One query issued by a query strategy."""

    kind: str
    query: str
    n_hits: int

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "query": self.query, "n_hits": self.n_hits}


@dataclass
class QueryResult:
    """Evidence and diagnostics produced before reranking and generation."""

    hits: list[Hit]
    generated_queries: list[str] = field(default_factory=list)
    trace: list[QueryTrace] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class QueryStrategy(Protocol):
    """A strategy that may issue one or more searches through one retriever."""

    name: str

    def execute(self, request: QueryRequest, retriever: Retriever) -> QueryResult:
        """Return candidate evidence for `request`."""
        ...
