"""Independent reranker behavior and pipeline contract enforcement."""

from __future__ import annotations

from dataclasses import replace

import pytest

from finrag.llm.fake import ScriptedLLM
from finrag.pipeline import (
    SearchPipeline,
    StrategyContractError,
    StrategyRuntime,
    StrategySelection,
)
from finrag.reranking.llm import LLMReranker, build_prompt, parse_order
from finrag.reranking.registry import available_rerankers
from finrag.retrieval.registry import IndexBundle


def direct_pipeline(chunks) -> SearchPipeline:
    return SearchPipeline(
        StrategySelection(retrieval_strategy="bm25"),
        StrategyRuntime(IndexBundle(chunks=list(chunks))),
    )


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("[3, 1, 2]", [3, 1, 2]),
        ('["2", "1"]', [2, 1]),
        ("answer: [2, 1]", [2, 1]),
        ("[1, 1]", None),
        ("[1]", None),
        ("[true, 2]", None),
        ("not json", None),
    ],
)
def test_rerank_order_parser_requires_an_exact_permutation(reply, expected) -> None:
    n = 3 if reply == "[3, 1, 2]" else 2
    assert parse_order(reply, n) == expected


def test_llm_reranker_reorders_every_existing_hit_once(micro_corpus) -> None:
    hits = direct_pipeline(micro_corpus).search("campaign spend", "c_level", k=3).hits
    reranker = LLMReranker(ScriptedLLM(["[3, 1, 2]"]), tries=1)
    ordered = reranker.rerank("original question", hits)
    assert [hit.chunk_id for hit in ordered] == [
        hits[2].chunk_id,
        hits[0].chunk_id,
        hits[1].chunk_id,
    ]
    assert sorted(hit.chunk_id for hit in ordered) == sorted(hit.chunk_id for hit in hits)


def test_llm_reranker_falls_back_to_input_order_after_invalid_replies(micro_corpus) -> None:
    hits = direct_pipeline(micro_corpus).search("campaign spend", "c_level", k=3).hits
    reranker = LLMReranker(ScriptedLLM(["bad"]), tries=2)
    assert reranker.rerank("question", hits) == hits
    assert reranker.calls == 2
    assert reranker.fallbacks == 1


@pytest.mark.parametrize("kind", ["drop", "duplicate", "add"])
def test_pipeline_rejects_rerankers_that_change_the_candidate_set(
    micro_corpus, kind
) -> None:
    pipeline = direct_pipeline(micro_corpus)

    class InvalidReranker:
        name = "invalid"
        depth = 0

        def rerank(self, question, hits):
            values = list(hits)
            if kind == "drop":
                return values[:-1]
            if kind == "duplicate":
                return values[:-1] + [values[0]]
            return values + [replace(values[0], chunk_id="invented")]

    pipeline.reranker = InvalidReranker()
    with pytest.raises(StrategyContractError, match="exactly the input hits"):
        pipeline.search("campaign spend", "c_level", k=3)


def test_hyde_reranker_receives_original_question(micro_corpus) -> None:
    original = "what was the campaign spend"
    llm = ScriptedLLM(["A hypothetical campaign passage."])
    pipeline = SearchPipeline(
        StrategySelection(
            retrieval_strategy="bm25",
            query_strategy="hyde",
            reranking_strategy="none",
        ),
        StrategyRuntime(
            IndexBundle(chunks=list(micro_corpus)),
            answer_llm=llm,
            query_llm=llm,
        ),
    )

    class RecordingReranker:
        name = "recording"
        depth = 0

        def __init__(self):
            self.question = None

        def rerank(self, question, hits):
            self.question = question
            return list(hits)

    recording = RecordingReranker()
    pipeline.reranker = recording
    result = pipeline.search(original, "marketing", k=5)
    assert recording.question == original
    assert result.trace[0].query == "A hypothetical campaign passage."


def test_rerank_prompt_uses_the_original_question_and_all_candidates(micro_corpus) -> None:
    hits = direct_pipeline(micro_corpus).search("campaign", "c_level", k=2).hits
    prompt = build_prompt("the original question", hits)
    assert "Question: the original question" in prompt
    assert "1." in prompt and "2." in prompt


def test_reranker_registry_contains_only_the_independent_axis() -> None:
    assert available_rerankers() == ["llm", "none"]
