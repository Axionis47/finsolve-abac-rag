"""Grounded answer generation over the modular search pipeline."""

from __future__ import annotations

import json

import pytest

from finrag.generate import Answerer
from finrag.llm.fake import FakeLLM, ScriptedLLM
from finrag.pipeline import StrategySelection, StrategyUnavailableError
from finrag.policy.engine import PolicyEngine
from finrag.prompting import build_prompt
from finrag.retrieval.registry import IndexBundle

SPEND_QUESTION = "what was the campaign spend and impressions"
REVENUE_QUESTION = "what was Q4 revenue"
DENIED_FINANCE_CHUNK = "finance/quarterly_financial_report#q4-revenue"


def make_answerer(chunks, llm=None, policy=None) -> Answerer:
    return Answerer(bundle=IndexBundle(chunks=list(chunks)), policy=policy, llm=llm)


def canned(answerer: Answerer, question: str, role: str, k: int, reply: str) -> list:
    selection = StrategySelection()
    hits = answerer.pipeline(selection).search(question, role, k=k).hits
    answerer.llm = FakeLLM({build_prompt(question, hits): reply})
    return hits


def test_a_role_with_no_permitted_chunk_gets_no_answer(micro_corpus) -> None:
    answerer = make_answerer(micro_corpus, llm=FakeLLM())
    result = answerer.answer(SPEND_QUESTION, "employee", k=5)
    assert result["answer"] is None
    assert result["reason"] == "no_answer"
    assert result["citations"] == []
    assert result["hits"] == []


def test_the_model_is_never_called_when_there_are_no_permitted_hits(micro_corpus) -> None:
    llm = FakeLLM()
    answerer = make_answerer(micro_corpus, llm=llm)
    answerer.answer(SPEND_QUESTION, "employee", k=5)
    assert llm.calls == []


def test_a_cited_answer_reports_the_chunk_it_cited(micro_corpus) -> None:
    answerer = make_answerer(micro_corpus, llm=FakeLLM())
    hits = canned(answerer, SPEND_QUESTION, "marketing", 5, "Spend was $2m [2].")
    result = answerer.answer(SPEND_QUESTION, "marketing", k=5)
    assert result["citations"] == [hits[1].chunk_id]
    assert result["extractive"] is False
    assert result["reason"] == "ok"


@pytest.mark.parametrize("reply", ["NOT_IN_CONTEXT", "   "])
def test_a_model_refusal_or_empty_reply_is_no_answer(micro_corpus, reply) -> None:
    answerer = make_answerer(micro_corpus, llm=FakeLLM())
    canned(answerer, SPEND_QUESTION, "marketing", 5, reply)
    result = answerer.answer(SPEND_QUESTION, "marketing", k=5)
    assert result["answer"] is None
    assert result["reason"] == "no_answer"


def test_with_no_model_the_answer_quotes_the_top_hit(micro_corpus) -> None:
    result = make_answerer(micro_corpus).answer(SPEND_QUESTION, "marketing", k=5)
    assert result["extractive"] is True
    assert result["citations"] == [result["hits"][0]["chunk_id"]]
    assert "impressions" in result["answer"]


def test_result_and_audit_report_all_three_axes(micro_corpus) -> None:
    answerer = make_answerer(micro_corpus)
    result = answerer.answer(
        SPEND_QUESTION,
        "marketing",
        retrieval_strategy="naive",
        query_strategy="direct",
        reranking_strategy="none",
        k=5,
    )
    assert result["pipeline_id"] == "direct__naive__none"
    assert result["retrieval_strategy"] == "naive"
    assert result["query_strategy"] == "direct"
    assert result["reranking_strategy"] == "none"
    assert result["audit"]["pipeline_id"] == "direct__naive__none"
    assert result["hits"][0]["retriever"] == "naive"


def test_audit_counts_masked_chunks_without_naming_them(micro_corpus) -> None:
    result = make_answerer(micro_corpus).answer(SPEND_QUESTION, "marketing", k=5)
    audit = result["audit"]
    assert audit["corpus_chunks"] == 6
    assert audit["visible_to_role"] == 2
    assert audit["masked_out"] == 4
    assert DENIED_FINANCE_CHUNK not in json.dumps(audit)


def test_role_mask_still_applies_before_every_strategy(micro_corpus) -> None:
    answerer = make_answerer(micro_corpus)
    marketing = answerer.answer(REVENUE_QUESTION, "marketing", k=5)
    executive = answerer.answer(REVENUE_QUESTION, "c_level", k=5)
    assert all(hit["owner_dept"] != "finance" for hit in marketing["hits"])
    assert DENIED_FINANCE_CHUNK in [hit["chunk_id"] for hit in executive["hits"]]


def test_policy_refusal_names_the_rule(micro_corpus, policy_path) -> None:
    answerer = make_answerer(
        micro_corpus, policy=PolicyEngine.from_yaml(policy_path)
    )
    refusal = answerer.policy_refusal("employee")
    assert refusal is not None
    assert "employee_non_general_deny" in refusal["rules"]
    assert answerer.policy_refusal("c_level") is None


def test_bm25_direct_none_creates_no_optional_dependency(micro_corpus) -> None:
    answerer = make_answerer(micro_corpus, llm=FakeLLM())
    answerer.pipeline(StrategySelection())
    assert answerer.bundle.embedder is None
    assert answerer.bundle.graph_store is None
    assert answerer.runtime.query_llm is None
    assert answerer.runtime.rerank_llm is None


def test_graph_dependency_failure_is_clear(micro_corpus, monkeypatch) -> None:
    monkeypatch.setattr("finrag.pipeline.bolt_is_open", lambda: False)
    answerer = make_answerer(micro_corpus, llm=FakeLLM())
    with pytest.raises(StrategyUnavailableError, match="docker start finrag-neo4j"):
        answerer.pipeline(StrategySelection(retrieval_strategy="graph_keyword"))


def test_fake_answer_backend_is_reused_for_querying_and_reranking(
    micro_corpus, monkeypatch
) -> None:
    def explode(_settings):
        raise AssertionError("a real Vertex client was created")

    monkeypatch.setattr("finrag.llm.vertex.build_query_llm", explode)
    monkeypatch.setattr("finrag.llm.vertex.build_rerank_llm", explode)
    fake = FakeLLM()
    answerer = make_answerer(micro_corpus, llm=fake)
    pipeline = answerer.pipeline(
        StrategySelection(
            retrieval_strategy="bm25",
            query_strategy="hyde",
            reranking_strategy="llm",
        )
    )
    assert pipeline.query_strategy.llm is fake
    assert pipeline.reranker.llm is fake


def test_react_selects_evidence_then_shared_generator_answers(micro_corpus) -> None:
    llm = ScriptedLLM(
        [
            '{"thought":"look","action":"search","query":"campaign spend"}',
            '{"thought":"enough","action":"finish","evidence":[1]}',
            "Spend was 2.0 million [1].",
        ]
    )
    answerer = make_answerer(micro_corpus, llm=llm)
    result = answerer.answer(
        "what was the campaign spend",
        "marketing",
        query_strategy="react",
        k=5,
    )
    assert result["answer"] == "Spend was 2.0 million [1]."
    assert result["citations"] == [result["hits"][0]["chunk_id"]]
    assert result["query_trace"][0]["query"] == "campaign spend"
    assert len(llm.calls) == 3


def test_hyde_trace_keeps_generated_query_but_final_answer_uses_original_question(
    micro_corpus,
) -> None:
    llm = ScriptedLLM(
        [
            "Campaign spend reached 2.0 million with 150 million impressions.",
            "Spend was 2.0 million [1].",
        ]
    )
    result = make_answerer(micro_corpus, llm=llm).answer(
        "what was the campaign spend",
        "marketing",
        query_strategy="hyde",
        k=5,
    )
    assert result["query_trace"][0]["kind"] == "hyde"
    assert "2.0 million" in result["query_trace"][0]["query"]
    assert "Question: what was the campaign spend" in llm.calls[-1][0]
