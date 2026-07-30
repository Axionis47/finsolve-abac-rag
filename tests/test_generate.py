"""The answer step: grounding, refusal, the extractive fallback, and the audit.

Offline throughout. The corpus is the six-chunk `micro_corpus` fixture and the
model is `FakeLLM`, whose canned replies are keyed on the exact prompt, so the
prompt builder and the citation parser are tested against each other rather than
against a pinned string.
"""

from __future__ import annotations

import json

from finrag.generate import Answerer
from finrag.llm.fake import FakeLLM
from finrag.policy.engine import PolicyEngine
from finrag.prompting import build_prompt
from finrag.retrieval.registry import IndexBundle

SPEND_QUESTION = "what was the campaign spend and impressions"
REVENUE_QUESTION = "what was Q4 revenue"
DENIED_FINANCE_CHUNK = "finance/quarterly_financial_report#q4-revenue"


def make_answerer(chunks, llm=None, policy=None) -> Answerer:
    """An answerer over an in-memory corpus. No artifacts, no network."""
    return Answerer(bundle=IndexBundle(chunks=list(chunks)), policy=policy, llm=llm)


def canned(answerer: Answerer, question: str, role: str, k: int, reply: str) -> None:
    """Point the answerer at a FakeLLM that replies `reply` to this exact prompt.

    The prompt is built from the real hits, so a change to the prompt builder
    fails here instead of silently making the canned reply unreachable.
    """
    hits = answerer.retriever("bm25").search(question, role, k=k)
    answerer.llm = FakeLLM({build_prompt(question, hits): reply})


def test_a_role_with_no_permitted_chunk_gets_no_answer(micro_corpus) -> None:
    answerer = make_answerer(micro_corpus, llm=FakeLLM())
    result = answerer.answer(SPEND_QUESTION, "employee", mode="bm25", k=5)
    assert result["answer"] is None
    assert result["reason"] == "no_answer"
    assert result["citations"] == []
    assert result["hits"] == []


def test_the_model_is_never_called_when_there_are_no_permitted_hits(micro_corpus) -> None:
    # The guard that matters: an empty context must not reach a model that would
    # cheerfully invent a number to fill it.
    llm = FakeLLM()
    answerer = make_answerer(micro_corpus, llm=llm)
    answerer.answer(SPEND_QUESTION, "employee", mode="bm25", k=5)
    assert llm.calls == []


def test_a_cited_answer_reports_the_chunk_it_cited(micro_corpus) -> None:
    answerer = make_answerer(micro_corpus, llm=FakeLLM())
    hits = answerer.retriever("bm25").search(SPEND_QUESTION, "marketing", k=5)
    canned(answerer, SPEND_QUESTION, "marketing", 5, "Spend was 2.0 million dollars [2].")

    result = answerer.answer(SPEND_QUESTION, "marketing", mode="bm25", k=5)
    assert result["citations"] == [hits[1].chunk_id]
    assert result["extractive"] is False
    assert result["reason"] == "ok"


def test_the_refusal_token_comes_back_as_no_answer(micro_corpus) -> None:
    answerer = make_answerer(micro_corpus, llm=FakeLLM())
    canned(answerer, SPEND_QUESTION, "marketing", 5, "NOT_IN_CONTEXT")
    result = answerer.answer(SPEND_QUESTION, "marketing", mode="bm25", k=5)
    assert result["answer"] is None
    assert result["reason"] == "no_answer"


def test_an_empty_reply_comes_back_as_no_answer(micro_corpus) -> None:
    answerer = make_answerer(micro_corpus, llm=FakeLLM())
    canned(answerer, SPEND_QUESTION, "marketing", 5, "   ")
    result = answerer.answer(SPEND_QUESTION, "marketing", mode="bm25", k=5)
    assert result["answer"] is None


def test_with_no_model_the_answer_quotes_the_top_hit(micro_corpus) -> None:
    answerer = make_answerer(micro_corpus, llm=None)
    result = answerer.answer(SPEND_QUESTION, "marketing", mode="bm25", k=5)
    assert result["extractive"] is True
    assert result["citations"] == [result["hits"][0]["chunk_id"]]
    assert "impressions" in result["answer"]


def test_the_audit_counts_the_chunks_the_mask_removed(micro_corpus) -> None:
    answerer = make_answerer(micro_corpus, llm=None)
    audit = answerer.answer(SPEND_QUESTION, "marketing", mode="bm25", k=5)["audit"]
    assert audit["corpus_chunks"] == 6
    assert audit["visible_to_role"] == 2
    assert audit["masked_out"] == 4


def test_the_audit_never_names_a_masked_chunk(micro_corpus) -> None:
    # Counts are fine to publish; the ids of chunks a role cannot see are exactly
    # what the mask exists to withhold.
    answerer = make_answerer(micro_corpus, llm=None)
    result = answerer.answer(SPEND_QUESTION, "marketing", mode="bm25", k=5)
    assert DENIED_FINANCE_CHUNK not in json.dumps(result["audit"])


def test_a_chunk_the_role_may_not_see_never_reaches_the_answer(micro_corpus) -> None:
    answerer = make_answerer(micro_corpus, llm=None)
    result = answerer.answer(REVENUE_QUESTION, "marketing", mode="bm25", k=5)
    for view in result["hits"]:
        assert view["owner_dept"] != "finance"


def test_the_same_question_as_c_level_does_reach_the_finance_chunk(micro_corpus) -> None:
    # The other half of the test above: the chunk is findable, so its absence for
    # marketing is the mask working, not the query failing to match.
    answerer = make_answerer(micro_corpus, llm=None)
    result = answerer.answer(REVENUE_QUESTION, "c_level", mode="bm25", k=5)
    ids = [view["chunk_id"] for view in result["hits"]]
    assert DENIED_FINANCE_CHUNK in ids


def test_the_answer_reports_the_mode_it_used(micro_corpus) -> None:
    answerer = make_answerer(micro_corpus, llm=None)
    result = answerer.answer(SPEND_QUESTION, "marketing", mode="naive", k=5)
    assert result["mode"] == "naive"
    assert result["hits"][0]["retriever"] == "naive"


def test_policy_refusal_names_the_rule_that_denies_the_role(micro_corpus, policy_path) -> None:
    policy = PolicyEngine.from_yaml(policy_path)
    answerer = make_answerer(micro_corpus, llm=None, policy=policy)
    refusal = answerer.policy_refusal("employee")
    assert refusal is not None
    assert "employee_non_general_deny" in refusal["rules"]


def test_policy_refusal_is_silent_for_a_role_that_may_see_something(
    micro_corpus, policy_path
) -> None:
    policy = PolicyEngine.from_yaml(policy_path)
    answerer = make_answerer(micro_corpus, llm=None, policy=policy)
    assert answerer.policy_refusal("c_level") is None


def test_policy_refusal_is_silent_when_no_policy_was_given(micro_corpus) -> None:
    # An answerer built without a policy still answers; it just cannot explain a
    # whole-corpus denial, so it says nothing rather than guessing at a rule.
    answerer = make_answerer(micro_corpus, llm=None)
    assert answerer.policy_refusal("employee") is None


# Which backends a mode needs. The serving layer builds them lazily, because
# building an embedder or a rerank client may open a Vertex connection and a
# process that only ever answers with bm25 should not pay for one.


def test_a_mode_that_needs_no_embedder_does_not_get_one(micro_corpus) -> None:
    answerer = make_answerer(micro_corpus, llm=FakeLLM())
    answerer.retriever("bm25")
    assert answerer.bundle.embedder is None
    assert answerer.bundle.llm is None


def test_the_fused_modes_all_need_an_embedder(micro_corpus) -> None:
    # hybrid and hybrid_rerank both run a dense retriever underneath. Leaving
    # them out of this set is what would make `/ask?mode=hybrid` a 500.
    from finrag.generate import MODES_NEEDING_AN_EMBEDDER

    assert MODES_NEEDING_AN_EMBEDDER == {"dense", "hybrid", "hybrid_rerank"}


def test_graphrag_with_the_server_down_refuses_instantly_and_says_how_to_fix_it(
    micro_corpus, monkeypatch
) -> None:
    # Without this probe the neo4j driver retries a dead server for ~30 seconds
    # and then raises something unreadable through the API as a bare 500.
    from finrag.generate import GraphUnavailableError

    monkeypatch.setattr("finrag.graph.store.bolt_is_open", lambda *a, **kw: False)
    answerer = make_answerer(micro_corpus, llm=FakeLLM())

    try:
        answerer.retriever("graphrag")
        raise AssertionError("a dead graph server was accepted")
    except GraphUnavailableError as err:
        assert "docker start finrag-neo4j" in str(err)


def test_an_offline_process_reranks_with_the_same_fake_it_answers_with(
    micro_corpus, monkeypatch
) -> None:
    # A process told to run on FakeLLM must not reach for credentials to rerank.
    # If it did, this test would raise on the missing project id.
    def explode(_settings):
        raise AssertionError("the rerank path built a real Vertex client")

    monkeypatch.setattr("finrag.llm.vertex.build_rerank_llm", explode)
    fake = FakeLLM()
    answerer = make_answerer(micro_corpus, llm=fake)
    answerer.bundle.embedder = _fake_embedder()

    retriever = answerer.retriever("hybrid_rerank")
    assert retriever.reranker.llm is fake


def _fake_embedder():
    from finrag.retrieval.embed import FakeEmbedder

    return FakeEmbedder(dim=32)
