"""Contract for the optional reranking stage."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from finrag.models import Hit


@runtime_checkable
class Reranker(Protocol):
    name: str
    depth: int

    def rerank(self, question: str, hits: Sequence[Hit]) -> list[Hit]:
        """Return the same hits in a new order."""
        ...
