"""The hybrid mode: the RRF arithmetic, the mask, and the tie-break.

Two kinds of test live here, and they need different tools.

The fusion arithmetic is checked against stub sub-retrievers that return a fixed
list of ids. A stub is the only way to hand-compute the expected score: with real
retrievers the input ranks depend on BM25 statistics and hash vectors, so the
"expected" number would have to be produced by the code under test.

The mask is checked twice over. Once with a deliberately LEAKY stub, which
returns a chunk the role may not see, to prove hybrid's own mask catches what a
broken sub-retriever hands it. Once with the real BM25 and dense retrievers, to
prove the assembled mode behaves.
"""

from __future__ import annotations

import pytest

from finrag.models import Chunk, Hit
from finrag.retrieval.bm25 import BM25Retriever
from finrag.retrieval.dense import DenseRetriever
from finrag.retrieval.embed import FakeEmbedder
from finrag.retrieval.hybrid import FUSION_DEPTH, RRF_K, HybridRetriever, rrf_weight
from finrag.retrieval.registry import IndexBundle, available_retrievers, build_retriever

QUERY = "how did customer churn move over the festive quarter"


def _chunk(chunk_id: str, text: str, allowed_roles: tuple[str, ...]) -> Chunk:
    """A chunk with only the fields these tests read."""
    return Chunk(
        chunk_id=chunk_id,
        doc_id=chunk_id.split("#")[0],
        section_path="Report > Section",
        section_slug=chunk_id.split("#")[1],
        text=text,
        owner_dept="finance" if chunk_id.startswith("finance/") else "marketing",
        doc_type="md",
        sensitivity="confidential" if chunk_id.startswith("finance/") else "internal",
        allowed_roles=allowed_roles,
    )


VISIBLE = ("marketing", "c_level")

A = _chunk("marketing/a#one", "seasonal churn notes for the festive quarter", VISIBLE)
B = _chunk("marketing/b#two", "brand colours and typeface", VISIBLE)
C = _chunk("marketing/c#three", "customer churn by region", VISIBLE)

# The bait. Its text is the query word for word, so any retriever that scores it
# puts it first. Only c_level may read it.
SECRET = _chunk("finance/secret#churn", QUERY, ("c_level",))

# The bait is listed first on purpose: source order must not save it either.
CORPUS = [SECRET, A, B, C]


class StubRetriever:
    """Returns a fixed id list as hits at ranks 1, 2, 3, ..., cut to `k`.

    Ignores the role entirely. That is the point for the leak test: a stub that
    hands hybrid a denied chunk is the broken sub-retriever hybrid's own mask
    exists to catch. `asked_for_k` records the depth hybrid requested.
    """

    def __init__(self, name: str, chunk_ids: list[str]) -> None:
        self.name = name
        self.chunk_ids = list(chunk_ids)
        self.asked_for_k: list[int] = []

    def search(self, query: str, role: str, k: int = 10) -> list[Hit]:
        self.asked_for_k.append(k)
        by_id: dict[str, Chunk] = {}
        for chunk in CORPUS:
            by_id[chunk.chunk_id] = chunk

        hits: list[Hit] = []
        rank = 1
        for chunk_id in self.chunk_ids[:k]:
            chunk = by_id[chunk_id]
            hits.append(
                Hit(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    section_slug=chunk.section_slug,
                    rank=rank,
                    score=1.0 / rank,
                    retriever=self.name,
                    owner_dept=chunk.owner_dept,
                    doc_type=chunk.doc_type,
                    sensitivity=chunk.sensitivity,
                    allowed_roles=chunk.allowed_roles,
                    text=chunk.text,
                )
            )
            rank += 1
        return hits


def ids_of(hits: list[Hit]) -> list[str]:
    out: list[str] = []
    for hit in hits:
        out.append(hit.chunk_id)
    return out


def stub_hybrid(lexical: list[str], semantic: list[str]) -> HybridRetriever:
    return HybridRetriever(
        CORPUS,
        StubRetriever("bm25", lexical),
        StubRetriever("dense", semantic),
    )


def real_hybrid(chunks: list[Chunk]) -> HybridRetriever:
    """Hybrid over the real BM25 and dense retrievers, offline."""
    return HybridRetriever(
        chunks,
        BM25Retriever(chunks),
        DenseRetriever(chunks, FakeEmbedder()),
    )


# --- the RRF arithmetic ------------------------------------------------------


def test_the_fused_score_is_the_sum_of_one_over_sixty_plus_rank() -> None:
    """Hand computed, on purpose, so the constant cannot drift unnoticed.

    bm25 returns A, B, C at ranks 1, 2, 3. dense returns C, A at ranks 1, 2.

      A = 1/(60+1) + 1/(60+2) = 0.0163934 + 0.0161290 = 0.0325225
      C = 1/(60+3) + 1/(60+1) = 0.0158730 + 0.0163934 = 0.0322664
      B = 1/(60+2)                                    = 0.0161290

    So A beats C by a quarter of a thousandth, and B, which only one list found,
    comes last despite sitting at rank 2 there.
    """
    hits = stub_hybrid([A.chunk_id, B.chunk_id, C.chunk_id], [C.chunk_id, A.chunk_id]).search(
        QUERY, "marketing", k=10
    )

    assert ids_of(hits) == [A.chunk_id, C.chunk_id, B.chunk_id]
    assert hits[0].score == pytest.approx(0.0325225, abs=1e-7)
    assert hits[1].score == pytest.approx(0.0322664, abs=1e-7)
    assert hits[2].score == pytest.approx(0.0161290, abs=1e-7)


def test_a_chunk_both_lists_found_beats_a_chunk_only_one_list_ranked_first() -> None:
    """The reason to fuse at all, stated as a test.

    bm25 returns B, A, C at ranks 1, 2, 3. dense returns C alone at rank 1.

      C = 1/63 + 1/61 = 0.0322664, found by both, last in one of them
      B = 1/61       = 0.0163934, rank 1 in one list, absent from the other
      A = 1/62       = 0.0161290

    Agreement beats a single confident vote, which is the whole bet of this rung.
    """
    hits = stub_hybrid([B.chunk_id, A.chunk_id, C.chunk_id], [C.chunk_id]).search(
        QUERY, "marketing", k=10
    )

    assert ids_of(hits) == [C.chunk_id, B.chunk_id, A.chunk_id]


def test_rrf_weight_uses_the_documented_constant() -> None:
    assert RRF_K == 60
    assert rrf_weight(1) == pytest.approx(1.0 / 61.0)
    assert rrf_weight(30) == pytest.approx(1.0 / 90.0)


def test_both_sub_retrievers_are_asked_for_the_full_fusion_depth() -> None:
    """A depth of k would throw away the evidence fusion exists to rescue."""
    lexical = StubRetriever("bm25", [A.chunk_id])
    semantic = StubRetriever("dense", [B.chunk_id])
    HybridRetriever(CORPUS, lexical, semantic).search(QUERY, "marketing", k=3)

    assert lexical.asked_for_k == [FUSION_DEPTH]
    assert semantic.asked_for_k == [FUSION_DEPTH]


# --- the invariant: mask before rank ----------------------------------------


def test_a_leaky_sub_retriever_cannot_smuggle_a_denied_chunk_through_fusion() -> None:
    """Hybrid's own mask is the point of this test, not the sub-retrievers'.

    The stub ignores the role and hands SECRET back at rank 1, which is what a
    broken or future sub-retriever would look like. k is 10, wider than the whole
    visible pool of three, so no cut-off is doing the hiding.
    """
    hits = stub_hybrid(
        [SECRET.chunk_id, A.chunk_id], [SECRET.chunk_id, B.chunk_id]
    ).search(QUERY, "marketing", k=10)

    assert SECRET.chunk_id not in ids_of(hits)
    # Ranks stay 1..2 with no gap, so the denied chunk did not consume a slot.
    assert [hit.rank for hit in hits] == [1, 2]


def test_a_denied_chunk_contributes_nothing_to_a_visible_chunk_score() -> None:
    """The leak that would survive a filter on the output.

    SECRET arrives at rank 1 in both lists, so A and B sit at rank 2 in one list
    each. Their fused scores must be 1/62 exactly: if the denied chunk had been
    dropped by re-ranking the survivors instead of being skipped, A and B would
    have been promoted to rank 1 and scored 1/61.
    """
    hits = stub_hybrid(
        [SECRET.chunk_id, A.chunk_id], [SECRET.chunk_id, B.chunk_id]
    ).search(QUERY, "marketing", k=10)

    for hit in hits:
        assert hit.score == pytest.approx(1.0 / 62.0)


def test_the_real_hybrid_hides_the_chunk_the_role_may_not_see() -> None:
    """The control below proves the bait was findable; the role is what stops it."""
    hits = real_hybrid(CORPUS).search(QUERY, "marketing", k=10)

    assert SECRET.chunk_id not in ids_of(hits)
    for hit in hits:
        assert "marketing" in hit.allowed_roles


def test_the_denied_chunk_really_was_the_top_hit_for_a_role_that_may_read_it() -> None:
    hits = real_hybrid(CORPUS).search(QUERY, "c_level", k=10)

    assert hits[0].chunk_id == SECRET.chunk_id


def test_the_mask_holds_at_every_k_including_one_that_is_too_small() -> None:
    for k in [1, 2, 3, 10, 50]:
        hits = real_hybrid(CORPUS).search(QUERY, "marketing", k=k)
        assert SECRET.chunk_id not in ids_of(hits)


def test_a_role_with_nothing_visible_gets_nothing() -> None:
    retriever = real_hybrid(CORPUS)
    assert retriever.search(QUERY, "engineering", k=10) == []
    assert retriever.search(QUERY, "not_a_role", k=10) == []


# --- ranking behaviour ------------------------------------------------------


def test_ties_break_on_chunk_id_not_on_which_list_came_first() -> None:
    """Equal fused scores are ordinary here, so the tie-break has to be pinned.

    B is rank 1 in the lexical list, A is rank 1 in the dense list, so both score
    1/61. B is listed first, and A still wins because its id sorts first.
    """
    hits = stub_hybrid([B.chunk_id], [A.chunk_id]).search(QUERY, "marketing", k=10)

    assert ids_of(hits) == [A.chunk_id, B.chunk_id]
    assert hits[0].score == pytest.approx(hits[1].score)


def test_two_identical_searches_return_identical_results() -> None:
    retriever = real_hybrid(CORPUS)
    first = retriever.search(QUERY, "marketing", k=10)
    second = retriever.search(QUERY, "marketing", k=10)

    assert ids_of(first) == ids_of(second)
    assert [hit.score for hit in first] == [hit.score for hit in second]


def test_k_cuts_the_list_and_ranks_start_at_one() -> None:
    hits = stub_hybrid([A.chunk_id, B.chunk_id, C.chunk_id], [C.chunk_id]).search(
        QUERY, "marketing", k=2
    )

    assert len(hits) == 2
    assert [hit.rank for hit in hits] == [1, 2]


def test_the_hit_records_which_rank_each_list_gave_it() -> None:
    """Absent from a list is recorded as None, not as a bad rank in it."""
    hits = stub_hybrid([A.chunk_id, B.chunk_id], [B.chunk_id]).search(QUERY, "marketing", k=10)

    by_id: dict[str, Hit] = {}
    for hit in hits:
        by_id[hit.chunk_id] = hit

    assert by_id[A.chunk_id].debug["bm25_rank"] == 1
    assert by_id[A.chunk_id].debug["dense_rank"] is None
    assert by_id[B.chunk_id].debug["bm25_rank"] == 2
    assert by_id[B.chunk_id].debug["dense_rank"] == 1


def test_every_hit_says_it_came_from_the_hybrid_retriever() -> None:
    for hit in real_hybrid(CORPUS).search(QUERY, "marketing", k=10):
        assert hit.retriever == "hybrid"


def test_an_empty_query_returns_nothing() -> None:
    assert real_hybrid(CORPUS).search("", "marketing", k=10) == []


def test_an_empty_corpus_returns_nothing() -> None:
    assert real_hybrid([]).search(QUERY, "marketing", k=10) == []


def test_a_fusion_depth_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="depth"):
        HybridRetriever(CORPUS, StubRetriever("bm25", []), StubRetriever("dense", []), depth=0)


# --- the registry -----------------------------------------------------------


def test_the_registry_builds_the_hybrid_mode_from_a_bundle() -> None:
    bundle = IndexBundle(chunks=CORPUS, embedder=FakeEmbedder())
    retriever = build_retriever("hybrid", bundle)

    assert retriever.name == "hybrid"
    assert SECRET.chunk_id not in ids_of(retriever.search(QUERY, "marketing", k=10))


def test_the_hybrid_mode_still_needs_an_embedder_for_its_dense_half() -> None:
    with pytest.raises(ValueError, match="embedder"):
        build_retriever("hybrid", IndexBundle(chunks=CORPUS))


def test_hybrid_is_a_mode_a_caller_may_ask_for() -> None:
    assert "hybrid" in available_retrievers()
