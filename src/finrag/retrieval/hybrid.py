"""Hybrid search: BM25 and dense fused by Reciprocal Rank Fusion.

Why fuse at all. R1 and R2 each left a hole the other one fills. BM25 scores
words, so it cannot reach a paraphrase (T2 section hit rate 0.000 under bm25).
Dense scores meaning, so it cannot tell four near-identical quarterly reports
apart: R2 measured its top-ten cosine spread at 13.9 percent of the top score
against BM25's 49.6 percent, which is a polite way of saying dense is confident
about the neighbourhood and nearly indifferent inside it. Fusion keeps BM25's
sharp opinions and dense's reach.

Why RRF rather than adding the scores. A BM25 score and a cosine are not the
same kind of number: one is unbounded and corpus-dependent, the other sits in
[-1, 1]. Adding them needs a weight and a normalisation, both of which would be
guesses tuned on 50 questions. RRF throws the scores away and keeps only the
ranks, so there is nothing to tune: a chunk at rank 1 in either list is worth
1/(60+1), and a chunk both lists like beats a chunk only one list loves.

Where the 60 comes from: the constant in the original RRF paper. It flattens the
difference between the top few ranks, so a rank-1 and a rank-2 hit are close
together and neither list can dominate on its top pick alone.

THE INVARIANT: both input lists arrive already role-masked, because each
sub-retriever masks before it ranks. This file masks a third time anyway, one
`if` inside the fusion loop, since a fusion score built from a denied chunk
would be a leak even if the chunk itself never surfaced.
"""

from __future__ import annotations

from collections.abc import Sequence

from finrag.models import Chunk, Hit
from finrag.retrieval.base import Retriever, visible_ids

# The RRF constant. 60 is the value from the paper, and we keep it rather than
# tuning it: with 50 gold questions any tuned value would be fitted to noise.
RRF_K = 60

# How deep to ask each sub-retriever before fusing. Generous on purpose: a chunk
# that BM25 puts at rank 25 and dense puts at rank 2 is exactly the case fusion
# exists to rescue, and a depth of 10 would throw the BM25 half of that evidence
# away before the fusion ever saw it.
FUSION_DEPTH = 30


def rrf_weight(rank: int) -> float:
    """What one appearance at `rank` (counting from 1) contributes."""
    return 1.0 / (RRF_K + rank)


class HybridRetriever:
    """BM25 and dense fused by RRF, role masked before any ranking.

    Takes the two sub-retrievers rather than building them, so the registry
    decides how each one is loaded and the tests can hand in retrievers over a
    fake embedder.

    `chunks` must be the same list, in the same order, that the sub-retrievers
    hold. It is used for two things: the mask, and looking a chunk up by id to
    build the fused hit. A shorter list here can only ever drop a chunk from the
    output, never add a denied one, because the sub-retrievers have already
    refused everything the role may not see.
    """

    name = "hybrid"

    def __init__(
        self,
        chunks: Sequence[Chunk],
        bm25: Retriever,
        dense: Retriever,
        depth: int = FUSION_DEPTH,
    ) -> None:
        if depth < 1:
            raise ValueError("fusion depth must be at least 1")
        self.chunks = list(chunks)
        self.bm25 = bm25
        self.dense = dense
        self.depth = depth

        self._chunk_by_id: dict[str, Chunk] = {}
        for chunk in self.chunks:
            self._chunk_by_id[chunk.chunk_id] = chunk

    def search(self, query: str, role: str, k: int = 10) -> list[Hit]:
        """Best `k` chunks this role may see, ranked by fused RRF score."""
        # The mask comes FIRST, before anything is scored or cut. Each
        # sub-retriever does this too; this is the outer lock of three.
        allowed = visible_ids(self.chunks, role)
        if not allowed:
            return []

        lexical_hits = self.bm25.search(query, role, k=self.depth)
        semantic_hits = self.dense.search(query, role, k=self.depth)

        fused = self._fuse(lexical_hits, semantic_hits, allowed)

        # Higher fused score first; chunk_id breaks ties so two runs agree. Ties
        # are common here and not a corner case: RRF scores come from a small set
        # of sums of 1/(60+rank), so two chunks landing on the same total is
        # ordinary rather than rare.
        fused.sort(key=lambda item: (-item[0], item[1]))

        hits: list[Hit] = []
        rank = 1
        for score, chunk_id, ranks in fused[:k]:
            hits.append(self._to_hit(chunk_id, rank, score, ranks))
            rank += 1
        return hits

    def _fuse(
        self,
        lexical_hits: Sequence[Hit],
        semantic_hits: Sequence[Hit],
        allowed: frozenset[str],
    ) -> list[tuple[float, str, dict[str, int]]]:
        """Sum 1/(60+rank) over both lists, per chunk.

        Returns one tuple per chunk: its fused score, its id, and which rank it
        held in each input list. The ranks go into the hit's debug field, which
        is how a bad fusion is diagnosed later without re-running both modes.
        """
        scores: dict[str, float] = {}
        ranks: dict[str, dict[str, int]] = {}

        for label, hit_list in (("bm25", lexical_hits), ("dense", semantic_hits)):
            for hit in hit_list:
                # Third lock on the same door. Both lists are already masked, so
                # this cannot fire today; it is here because a denied chunk that
                # contributed to a fused score would be a leak even if it never
                # appeared in the output.
                if hit.chunk_id not in allowed:
                    continue
                # A chunk missing from our own list cannot be turned into a hit,
                # so skip it rather than guess its attributes.
                if hit.chunk_id not in self._chunk_by_id:
                    continue

                if hit.chunk_id not in scores:
                    scores[hit.chunk_id] = 0.0
                    ranks[hit.chunk_id] = {}
                scores[hit.chunk_id] += rrf_weight(hit.rank)
                ranks[hit.chunk_id][label] = hit.rank

        fused: list[tuple[float, str, dict[str, int]]] = []
        for chunk_id in scores:
            fused.append((scores[chunk_id], chunk_id, ranks[chunk_id]))
        return fused

    def _to_hit(
        self,
        chunk_id: str,
        rank: int,
        score: float,
        input_ranks: dict[str, int],
    ) -> Hit:
        """Copy the chunk attributes the policy replay and the metrics need."""
        chunk = self._chunk_by_id[chunk_id]
        return Hit(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            section_slug=chunk.section_slug,
            rank=rank,
            score=score,
            retriever=self.name,
            owner_dept=chunk.owner_dept,
            doc_type=chunk.doc_type,
            sensitivity=chunk.sensitivity,
            allowed_roles=chunk.allowed_roles,
            text=chunk.text,
            debug={
                "rrf_score": score,
                # Absent means the chunk was not in that list at all, which is
                # different from being at a bad rank in it.
                "bm25_rank": input_ranks.get("bm25"),
                "dense_rank": input_ranks.get("dense"),
                "fusion_depth": self.depth,
            },
        )
