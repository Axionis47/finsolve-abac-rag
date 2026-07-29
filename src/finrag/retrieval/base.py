"""Retriever contract and the role-mask helper every retriever must use.

INVARIANT (the whole point of this file): a retriever masks its candidate set
through `visible_ids` BEFORE any ranking cut. Not after scoring, not as a
post-filter on the top k, and not as a re-rank. Denied chunks must never
occupy a rank slot, contribute to a fusion score, or serve as an intermediate
hop in graph expansion, because any of those leaks information about content
the role cannot see. Fusion and rerank modes mask each input list, not just
the merged output.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from finrag.models import Chunk, Hit


@runtime_checkable
class Retriever(Protocol):
    """A named search strategy over the chunk set."""

    name: str

    def search(self, query: str, role: str, k: int = 10) -> list[Hit]:
        """Return at most k hits visible to `role`, best first, ranks from 1."""
        ...


def visible_ids(chunks: Iterable[Chunk], role: str) -> frozenset[str]:
    """Chunk ids whose `allowed_roles` contains `role`.

    This is the only sanctioned way to build a candidate mask. `allowed_roles`
    was written at ingest by evaluating the policy engine, so this stays in
    step with the policy without re-evaluating it per query.
    """
    return frozenset(c.chunk_id for c in chunks if role in c.allowed_roles)
