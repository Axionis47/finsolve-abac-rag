"""The hybrid_rerank mode: the prompt, the parser, and the fallback.

Nothing here calls a model. Two stand-ins do the work:

  * `FakeLLM` from `llm/fake.py`. Given a canned reply for an exact prompt it
    returns it, and given anything else it returns a hash stub, which is garbage
    as far as the parser is concerned. That makes it both the happy path and the
    bad-parse path with no extra class.
  * `RaisingLLM` below, for the transport failure.

The behaviours worth pinning are the ones a whole eval run would hide. A rerank
that silently fell back on every question still produces a full report; only the
fallback counter tells you. A rerank that let the model name a chunk id could
leak, and only the parser stops that.
"""

from __future__ import annotations

import pytest

from finrag.llm.fake import FakeLLM
from finrag.models import Chunk, Hit
from finrag.retrieval.bm25 import BM25Retriever
from finrag.retrieval.dense import DenseRetriever
from finrag.retrieval.embed import FakeEmbedder
from finrag.retrieval.hybrid import HybridRetriever
from finrag.retrieval.registry import IndexBundle, available_modes, build_retriever
from finrag.retrieval.rerank import (
    RERANK_DEPTH,
    RETRY_TRIES,
    LLMReranker,
    RerankRetriever,
    build_prompt,
    excerpt,
    parse_order,
)

QUERY = "how did customer churn move over the festive quarter"


def _chunk(chunk_id: str, text: str, allowed_roles: tuple[str, ...]) -> Chunk:
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

# The bait, readable by c_level only. Its text is a phrase no other chunk uses,
# so a test can assert it never reached the prompt.
SECRET_TEXT = "board only: the merger closes in March"
SECRET = _chunk("finance/secret#merger", SECRET_TEXT, ("c_level",))

CORPUS = [SECRET, A, B, C]


def hit_for(chunk: Chunk, rank: int) -> Hit:
    """One hybrid-shaped hit, with the fields the reranker copies."""
    return Hit(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        section_slug=chunk.section_slug,
        rank=rank,
        score=1.0 / (60 + rank),
        retriever="hybrid",
        owner_dept=chunk.owner_dept,
        doc_type=chunk.doc_type,
        sensitivity=chunk.sensitivity,
        allowed_roles=chunk.allowed_roles,
        text=chunk.text,
        debug={"rrf_score": 1.0 / (60 + rank)},
    )


THREE_HITS = [hit_for(A, 1), hit_for(B, 2), hit_for(C, 3)]


def ids_of(hits: list[Hit]) -> list[str]:
    out: list[str] = []
    for hit in hits:
        out.append(hit.chunk_id)
    return out


class RaisingLLM:
    """Fails every call, the way a missing credential or a dead endpoint does."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str, system: str | None = None) -> str:
        self.calls += 1
        raise RuntimeError("no credentials")


class LeakyRetriever:
    """Returns every chunk it was given, role or no role.

    Stands in for a broken or future inner retriever. The reranker must not show
    its output to the model, let alone return it.
    """

    name = "leaky"

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = list(chunks)

    def search(self, query: str, role: str, k: int = 10) -> list[Hit]:
        hits: list[Hit] = []
        rank = 1
        for chunk in self.chunks[:k]:
            hits.append(hit_for(chunk, rank))
            rank += 1
        return hits


def real_hybrid(chunks: list[Chunk]) -> HybridRetriever:
    return HybridRetriever(
        chunks, BM25Retriever(chunks), DenseRetriever(chunks, FakeEmbedder())
    )


def scripted_reranker(hits: list[Hit], reply: str) -> tuple[LLMReranker, FakeLLM]:
    """A reranker whose model answers `reply` to the prompt these hits produce."""
    llm = FakeLLM({build_prompt(QUERY, hits): reply})
    return LLMReranker(llm), llm


# --- the excerpt and the prompt ---------------------------------------------


def test_an_excerpt_is_cut_to_the_limit() -> None:
    assert excerpt("abcdefghij", limit=4) == "abcd"


def test_an_excerpt_is_flattened_onto_one_line() -> None:
    """The prompt numbers one passage per line, so a newline would split it."""
    assert excerpt("first line\n\nsecond   line") == "first line second line"


def test_the_prompt_numbers_every_passage_from_one() -> None:
    prompt = build_prompt(QUERY, THREE_HITS)

    assert QUERY in prompt
    assert "1. seasonal churn notes" in prompt
    assert "2. brand colours" in prompt
    assert "3. customer churn by region" in prompt


def test_the_prompt_asks_for_all_the_numbers_it_showed() -> None:
    prompt = build_prompt(QUERY, THREE_HITS)

    assert "Order all 3 passages" in prompt
    assert "JSON array of the 3 numbers" in prompt


def test_the_prompt_never_shows_a_chunk_id() -> None:
    """Ids are withheld so the model can only answer with positions we chose."""
    prompt = build_prompt(QUERY, THREE_HITS)

    for hit in THREE_HITS:
        assert hit.chunk_id not in prompt


# --- the parser --------------------------------------------------------------


def test_a_clean_permutation_parses() -> None:
    assert parse_order("[3, 1, 2]", 3) == [3, 1, 2]


def test_a_fenced_and_prefixed_reply_still_parses() -> None:
    """What models actually send when they have been told not to."""
    reply = "Here is the order:\n```json\n[2, 1, 3]\n```\nHope that helps."
    assert parse_order(reply, 3) == [2, 1, 3]


def test_numbers_sent_as_strings_still_parse() -> None:
    assert parse_order('["2", "1"]', 2) == [2, 1]


def test_an_array_wrapped_in_an_object_still_parses() -> None:
    """Falls out of taking the text between the first `[` and the last `]`.

    Worth stating as intended rather than leaving as an accident: a model that
    answers `{"order": [...]}` has told us the ordering, and the permutation
    check still has to pass before it is used.
    """
    assert parse_order('{"order": [1, 3, 2]}', 3) == [1, 3, 2]


@pytest.mark.parametrize(
    "reply",
    [
        "FAKE_LLM::abc::nonsense",  # no brackets at all
        "[1, 2",  # unterminated
        "[1, 2,]",  # not valid JSON
        "[1, 2]",  # too short for n=3
        "[1, 2, 3, 4]",  # too long for n=3
        "[1, 1, 2]",  # a repeat, so one passage would be dropped
        "[0, 1, 2]",  # out of range low
        "[1, 2, 4]",  # out of range high
        "[1, 2, true]",  # a bool is an int in Python; not a position
        "[1, 2, 3.5]",  # a float is not a position
        '["a", "b", "c"]',  # not numbers
    ],
)
def test_a_reply_that_is_not_an_exact_permutation_is_refused(reply: str) -> None:
    """Strict on purpose: a patched-up partial order would hide the failure."""
    assert parse_order(reply, 3) is None


# --- the reranker ------------------------------------------------------------


def test_the_model_order_is_the_returned_order() -> None:
    reranker, _llm = scripted_reranker(THREE_HITS, "[3, 1, 2]")

    ordered = reranker.rerank(QUERY, THREE_HITS)

    assert ids_of(ordered) == [C.chunk_id, A.chunk_id, B.chunk_id]
    assert reranker.fallbacks == 0
    assert reranker.calls == 1


def test_a_garbage_reply_falls_back_to_the_order_it_was_given() -> None:
    """FakeLLM with no canned answer returns its hash stub, which is garbage."""
    reranker = LLMReranker(FakeLLM())

    ordered = reranker.rerank(QUERY, THREE_HITS)

    assert ids_of(ordered) == ids_of(THREE_HITS)
    assert reranker.fallbacks == 1
    assert reranker.calls == RETRY_TRIES
    assert "unparsable" in (reranker.last_failure or "")


def test_a_failing_model_falls_back_to_the_order_it_was_given() -> None:
    llm = RaisingLLM()
    reranker = LLMReranker(llm)

    ordered = reranker.rerank(QUERY, THREE_HITS)

    assert ids_of(ordered) == ids_of(THREE_HITS)
    assert reranker.fallbacks == 1
    assert llm.calls == RETRY_TRIES
    assert "no credentials" in (reranker.last_failure or "")


def test_a_single_hit_is_not_worth_a_call() -> None:
    llm = FakeLLM()
    reranker = LLMReranker(llm)

    ordered = reranker.rerank(QUERY, [THREE_HITS[0]])

    assert ids_of(ordered) == [A.chunk_id]
    assert llm.calls == []
    assert reranker.fallbacks == 0


def test_the_reranked_list_is_always_a_permutation_of_the_input() -> None:
    """Never drops a hit, never invents one, whatever the model said."""
    reranker, _llm = scripted_reranker(THREE_HITS, "[2, 3, 1]")

    ordered = reranker.rerank(QUERY, THREE_HITS)

    assert sorted(ids_of(ordered)) == sorted(ids_of(THREE_HITS))


def test_a_rerank_depth_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="depth"):
        LLMReranker(FakeLLM(), depth=0)


# --- the retriever wrapper --------------------------------------------------


def test_the_inner_retriever_is_asked_deeper_than_the_caller_wanted() -> None:
    """The point of the rung: the right passage is often at rank 12, not 2."""
    inner = real_hybrid(CORPUS)
    asked: list[int] = []
    original = inner.search

    def spy(query: str, role: str, k: int = 10) -> list[Hit]:
        asked.append(k)
        return original(query, role, k=k)

    inner.search = spy  # type: ignore[method-assign]
    RerankRetriever(CORPUS, inner, LLMReranker(FakeLLM())).search(QUERY, "marketing", k=5)

    assert asked == [RERANK_DEPTH]


def test_ranks_are_renumbered_from_one_after_the_reorder() -> None:
    reranker, _llm = scripted_reranker(THREE_HITS, "[3, 1, 2]")
    retriever = RerankRetriever(CORPUS, LeakyRetriever([A, B, C]), reranker)

    hits = retriever.search(QUERY, "marketing", k=3)

    assert [hit.rank for hit in hits] == [1, 2, 3]
    assert hits[0].debug["hybrid_rank"] == 3
    assert hits[0].debug["reranked"] is True
    assert hits[0].retriever == "hybrid_rerank"


def test_a_fallback_is_reported_in_the_hit_debug_as_not_reranked() -> None:
    """`reranked: true` on a fallback would hide the thing we are counting."""
    retriever = RerankRetriever(CORPUS, LeakyRetriever([A, B, C]), LLMReranker(FakeLLM()))

    hits = retriever.search(QUERY, "marketing", k=3)

    assert ids_of(hits) == [A.chunk_id, B.chunk_id, C.chunk_id]
    for hit in hits:
        assert hit.debug["reranked"] is False


def test_k_still_cuts_the_list_after_reranking() -> None:
    reranker, _llm = scripted_reranker(THREE_HITS, "[3, 1, 2]")
    retriever = RerankRetriever(CORPUS, LeakyRetriever([A, B, C]), reranker)

    hits = retriever.search(QUERY, "marketing", k=2)

    assert ids_of(hits) == [C.chunk_id, A.chunk_id]


# --- the invariant: mask before the model sees anything ---------------------


def test_a_leaky_inner_retriever_cannot_get_a_denied_chunk_past_the_reranker() -> None:
    """k is 10, wider than the visible pool of three, so no cut-off hides it."""
    retriever = RerankRetriever(CORPUS, LeakyRetriever(CORPUS), LLMReranker(FakeLLM()))

    hits = retriever.search(QUERY, "marketing", k=10)

    assert SECRET.chunk_id not in ids_of(hits)
    assert [hit.rank for hit in hits] == [1, 2, 3]


def test_the_denied_chunk_text_is_never_put_in_the_prompt() -> None:
    """The sharper half of the same invariant: the model must not even read it."""
    llm = FakeLLM()
    retriever = RerankRetriever(CORPUS, LeakyRetriever(CORPUS), LLMReranker(llm))

    retriever.search(QUERY, "marketing", k=10)

    assert llm.calls != []
    for prompt, _system in llm.calls:
        assert SECRET_TEXT not in prompt


def test_a_role_with_nothing_visible_gets_nothing_and_costs_no_call() -> None:
    llm = FakeLLM()
    retriever = RerankRetriever(CORPUS, LeakyRetriever(CORPUS), LLMReranker(llm))

    assert retriever.search(QUERY, "engineering", k=10) == []
    assert llm.calls == []


def test_two_identical_searches_return_identical_results() -> None:
    reranker, _llm = scripted_reranker(THREE_HITS, "[2, 3, 1]")
    retriever = RerankRetriever(CORPUS, LeakyRetriever([A, B, C]), reranker)

    first = retriever.search(QUERY, "marketing", k=3)
    second = retriever.search(QUERY, "marketing", k=3)

    assert ids_of(first) == ids_of(second)


# --- the registry -----------------------------------------------------------


def test_the_registry_builds_the_rerank_mode_from_a_bundle() -> None:
    bundle = IndexBundle(chunks=CORPUS, embedder=FakeEmbedder(), llm=FakeLLM())
    retriever = build_retriever("hybrid_rerank", bundle)

    assert retriever.name == "hybrid_rerank"
    assert SECRET.chunk_id not in ids_of(retriever.search(QUERY, "marketing", k=10))


def test_the_rerank_mode_refuses_a_bundle_with_no_model() -> None:
    bundle = IndexBundle(chunks=CORPUS, embedder=FakeEmbedder())
    with pytest.raises(ValueError, match="llm"):
        build_retriever("hybrid_rerank", bundle)


def test_the_rerank_mode_is_registered() -> None:
    assert "hybrid_rerank" in available_modes()
    # The full list moved to tests/graph/test_graph_retriever.py when graphrag
    # landed, keeping the convention that the newest mode's file pins it.
