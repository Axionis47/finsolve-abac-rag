"""The naive baseline retriever and the mode registry, tested on their own.

`NaiveRetriever` is the floor every later mode has to beat, and the first place
the role mask is applied. Two things matter here and neither is covered by
scoring a whole eval run:

  * the mask runs BEFORE ranking. The proof below is direct: the chunk with the
    highest word overlap by a clear margin is one this role may not see, and `k`
    is set wider than the whole visible pool, so a post-filter would still have
    room to return it. It must not appear at any rank.
  * ties break the same way every run, or two runs of the same eval would post
    different numbers for no reason.

`build_retriever` is the one place that turns a mode string into an object, so
its unknown-mode message is part of the contract with the CLI.
"""

from __future__ import annotations

import pytest

from finrag.models import Chunk
from finrag.retrieval import naive
from finrag.retrieval.naive import NaiveRetriever
from finrag.retrieval.registry import IndexBundle, available_modes, build_retriever

# The bait: a chunk packed with the query words that only c_level may read.
SECRET = Chunk(
    chunk_id="finance/secret#margins",
    doc_id="finance/secret",
    section_path="Secret > Margins",
    section_slug="margins",
    text="quarterly revenue margin growth forecast target ratio",
    owner_dept="finance",
    doc_type="md",
    sensitivity="confidential",
    allowed_roles=("c_level",),
)

# Two chunks marketing may read. Each carries exactly one query word, so both
# score 1 and the tie has to be broken by chunk_id.
VISIBLE_B = Chunk(
    chunk_id="marketing/b#spend",
    doc_id="marketing/b",
    section_path="B > Spend",
    section_slug="spend",
    text="campaign revenue notes",
    owner_dept="marketing",
    doc_type="md",
    sensitivity="internal",
    allowed_roles=("marketing", "c_level"),
)

VISIBLE_A = Chunk(
    chunk_id="marketing/a#spend",
    doc_id="marketing/a",
    section_path="A > Spend",
    section_slug="spend",
    text="campaign margin notes",
    owner_dept="marketing",
    doc_type="md",
    sensitivity="internal",
    allowed_roles=("marketing", "c_level"),
)

# A chunk sharing no word with the query, so it never earns a slot.
UNRELATED = Chunk(
    chunk_id="marketing/c#logo",
    doc_id="marketing/c",
    section_path="C > Logo",
    section_slug="logo",
    text="brand colours and typeface",
    owner_dept="marketing",
    doc_type="md",
    sensitivity="internal",
    allowed_roles=("marketing", "c_level"),
)

# The bait is listed first on purpose: source order must not save it either.
CORPUS = [SECRET, VISIBLE_B, VISIBLE_A, UNRELATED]

QUERY = "quarterly revenue margin growth forecast target ratio"


def test_tokenize_folds_case_and_splits_on_punctuation() -> None:
    assert naive.tokenize("Q4 Revenue: 2.6 billion!") == ["q4", "revenue", "2", "6", "billion"]


def test_overlap_score_counts_distinct_query_words_only() -> None:
    query = naive.unique_terms("revenue revenue margin")
    assert naive.overlap_score(query, naive.unique_terms("revenue margin")) == 2
    # A chunk repeating the word still proves it once.
    assert naive.overlap_score(query, naive.unique_terms("revenue revenue revenue")) == 1
    assert naive.overlap_score(query, naive.unique_terms("brand colours")) == 0


def test_the_best_scoring_chunk_is_hidden_when_the_role_may_not_see_it() -> None:
    """The mask runs before ranking, so a denied chunk never occupies a slot.

    SECRET contains all seven query words; the two visible chunks contain one
    each. k is 10, wider than the whole visible pool of two, so nothing is being
    squeezed out by a cut-off. If the mask ran after scoring, SECRET would be
    sitting at rank 1 right now.
    """
    retriever = NaiveRetriever(CORPUS)

    hits = retriever.search(QUERY, "marketing", k=10)

    returned = [hit.chunk_id for hit in hits]
    assert SECRET.chunk_id not in returned
    assert returned == ["marketing/a#spend", "marketing/b#spend"]
    # And the same query as c_level does reach it, so the chunk really was
    # findable and it was the role that stopped it.
    c_level_hits = retriever.search(QUERY, "c_level", k=10)
    assert c_level_hits[0].chunk_id == SECRET.chunk_id


def test_a_denied_chunk_is_absent_even_when_it_would_have_placed_last() -> None:
    """Nothing about the ranking saves it: an unlisted role sees nothing at all."""
    retriever = NaiveRetriever(CORPUS)
    assert retriever.search(QUERY, "engineering", k=10) == []
    assert retriever.search(QUERY, "not_a_role", k=10) == []


def test_ties_break_on_chunk_id_so_two_runs_agree() -> None:
    """Both visible chunks score 1, and `marketing/a` sorts before `marketing/b`."""
    retriever = NaiveRetriever(CORPUS)

    first = retriever.search(QUERY, "marketing", k=10)
    second = retriever.search(QUERY, "marketing", k=10)

    assert [hit.chunk_id for hit in first] == [hit.chunk_id for hit in second]
    assert first[0].chunk_id == "marketing/a#spend"
    assert first[0].score == first[1].score == 1.0
    assert [hit.rank for hit in first] == [1, 2]


def test_ties_break_the_same_way_whatever_order_the_chunks_arrived_in() -> None:
    """Feeding the corpus backwards must not change the answer."""
    forwards = NaiveRetriever(CORPUS).search(QUERY, "marketing", k=10)
    backwards = NaiveRetriever(list(reversed(CORPUS))).search(QUERY, "marketing", k=10)
    assert [hit.chunk_id for hit in forwards] == [hit.chunk_id for hit in backwards]


def test_a_chunk_sharing_no_word_with_the_query_never_appears() -> None:
    hits = NaiveRetriever(CORPUS).search(QUERY, "marketing", k=10)
    assert UNRELATED.chunk_id not in [hit.chunk_id for hit in hits]


def test_an_empty_query_returns_nothing() -> None:
    """No terms means no evidence, so we return nothing rather than everything."""
    retriever = NaiveRetriever(CORPUS)
    assert retriever.search("", "c_level", k=10) == []
    assert retriever.search("   ", "c_level", k=10) == []
    # Punctuation alone tokenises to no words at all.
    assert retriever.search("?!,.", "c_level", k=10) == []


def test_an_empty_corpus_returns_nothing() -> None:
    assert NaiveRetriever([]).search(QUERY, "c_level", k=10) == []


def test_k_cuts_the_list_and_ranks_still_start_at_one() -> None:
    hits = NaiveRetriever(CORPUS).search(QUERY, "marketing", k=1)
    assert len(hits) == 1
    assert hits[0].rank == 1
    assert hits[0].chunk_id == "marketing/a#spend"


def test_a_hit_carries_what_the_policy_replay_and_the_metrics_need() -> None:
    hit = NaiveRetriever(CORPUS).search(QUERY, "marketing", k=1)[0]

    assert hit.doc_id == "marketing/a"
    assert hit.section_slug == "spend"
    assert hit.retriever == "naive"
    assert hit.owner_dept == "marketing"
    assert hit.doc_type == "md"
    assert hit.sensitivity == "internal"
    assert hit.allowed_roles == ("marketing", "c_level")
    assert hit.debug["matched_terms"] == ["margin"]


def test_available_modes_lists_naive() -> None:
    assert available_modes() == sorted(available_modes())
    assert "naive" in available_modes()
    # Each later rung adds its own mode here; test_bm25.py pins the full list.


def test_build_retriever_builds_the_naive_mode() -> None:
    retriever = build_retriever("naive", IndexBundle(chunks=CORPUS))

    assert isinstance(retriever, NaiveRetriever)
    assert retriever.name == "naive"
    assert retriever.search(QUERY, "marketing", k=10)[0].chunk_id == "marketing/a#spend"


def test_build_retriever_names_the_known_modes_when_asked_for_a_bad_one() -> None:
    """The message is what the CLI shows a person who mistyped `--mode`.

    The bad name is one no rung will ever add, so this test does not have to be
    edited every time a real mode lands.
    """
    with pytest.raises(KeyError) as caught:
        build_retriever("telepathy", IndexBundle(chunks=CORPUS))

    message = str(caught.value)
    assert "telepathy" in message
    for mode in available_modes():
        assert mode in message
