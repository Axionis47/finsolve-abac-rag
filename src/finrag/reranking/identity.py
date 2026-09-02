"""The no-op reranker used when no second-stage ranking is requested."""

from __future__ import annotations

from collections.abc import Sequence

from finrag.models import Hit


class IdentityReranker:
    name = "none"
    depth = 0

    def rerank(self, question: str, hits: Sequence[Hit]) -> list[Hit]:
        return list(hits)
