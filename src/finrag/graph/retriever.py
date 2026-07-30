"""GraphRetriever: the graph walk behind the same `search` contract as BM25.

Same shape as `BM25Retriever` on purpose. It takes the chunk list and returns
ranked `Hit`s for a role, so the eval harness, the CLI and the API can score it
against the other modes with nothing changed but a mode name. The score is the
number of distinct paths that reached the chunk, which is not comparable with a
BM25 score and is not meant to be; only the ranking is compared.

The mask is `traverse.py`'s job and it happens inside the Cypher. This file adds
one more lock anyway when it turns a candidate into a `Hit`, because a Hit is the
thing that leaves the retrieval layer with chunk text in it.

Registered in `retrieval/registry.py` as mode "graphrag" (wired at integration,
per docs/r5_wiring.md).
"""

from __future__ import annotations

from collections.abc import Sequence

from finrag.graph.store import GraphStore
from finrag.graph.traverse import DEFAULT_MAX_HOPS, Candidate, traverse
from finrag.models import Chunk, Hit
from finrag.retrieval.base import visible_ids

MODE_NAME = "graphrag"


class GraphRetriever:
    """Graph traversal search over the chunk set, role masked during the walk."""

    name = MODE_NAME

    def __init__(
        self,
        chunks: Sequence[Chunk],
        store: GraphStore,
        max_hops: int = DEFAULT_MAX_HOPS,
    ) -> None:
        self.chunks = list(chunks)
        self.store = store
        self.max_hops = max_hops

        # Hits need the chunk's own attributes, and the graph stores ids, not
        # chunks. One lookup table, built once, keeps a query from scanning.
        self._by_id: dict[str, Chunk] = {}
        for chunk in self.chunks:
            self._by_id[chunk.chunk_id] = chunk

    def search(self, query: str, role: str, k: int = 10) -> list[Hit]:
        """Best `k` chunks this role may see, ranked by how many paths reach them."""
        allowed = visible_ids(self.chunks, role)
        result = traverse(
            self.store,
            query,
            role,
            self.chunks,
            k=k,
            max_hops=self.max_hops,
        )

        hits: list[Hit] = []
        rank = 1
        for candidate in result.candidates:
            chunk = self._by_id.get(candidate.chunk_id)
            if chunk is None:
                # The graph holds a chunk the current chunks.jsonl does not. That
                # means the graph is stale; skip it rather than serve text we can
                # no longer check the policy on.
                continue
            # Third lock, on the way out. A denied chunk must never become a Hit.
            if chunk.chunk_id not in allowed:
                continue
            hits.append(
                self._to_hit(
                    chunk,
                    rank,
                    candidate,
                    result.seed_terms,
                    result.n_blocked_late,
                )
            )
            rank += 1
        return hits

    def _to_hit(
        self,
        chunk: Chunk,
        rank: int,
        candidate: Candidate,
        seed_terms: Sequence[str],
        n_blocked_late: int,
    ) -> Hit:
        """Copy the chunk attributes the policy replay and the metrics need."""
        return Hit(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            section_slug=chunk.section_slug,
            rank=rank,
            score=float(candidate.n_paths),
            retriever=self.name,
            owner_dept=chunk.owner_dept,
            doc_type=chunk.doc_type,
            sensitivity=chunk.sensitivity,
            allowed_roles=chunk.allowed_roles,
            text=chunk.text,
            debug={
                "n_paths": candidate.n_paths,
                "max_hops": self.max_hops,
                "seed_terms": list(seed_terms),
                # Always 0 in a correct build; see TraversalResult.
                "n_blocked_late": n_blocked_late,
            },
        )
