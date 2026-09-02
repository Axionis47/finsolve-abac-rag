"""Leakage sweep reports composed pipelines and remains offline."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from finrag.config import Settings
from finrag.eval import sweep
from finrag.eval.sweep import breakdown_lines, run_leakage_sweep, sweep_passed
from finrag.models import ROLES, Chunk, Hit
from finrag.pipeline import StrategySelection
from finrag.policy.engine import PolicyEngine
from finrag.querying.base import QueryResult
from finrag.retrieval.registry import available_retrievers

QUESTIONS: list[dict[str, Any]] = [
    {"id": "Q-REVENUE", "role": "c_level", "question": "What was Q4 revenue?"},
    {"id": "Q-SPEND", "role": "marketing", "question": "What was campaign spend?"},
]


class LeakyPipeline:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks

    def search(self, question: str, role: str, k: int = 10) -> QueryResult:
        return QueryResult(
            [
                Hit(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    section_slug=chunk.section_slug,
                    rank=rank,
                    score=1.0,
                    retriever="leaky",
                    owner_dept=chunk.owner_dept,
                    doc_type=chunk.doc_type,
                    sensitivity=chunk.sensitivity,
                    allowed_roles=chunk.allowed_roles,
                    text=chunk.text,
                )
                for rank, chunk in enumerate(self.chunks[:k], start=1)
            ]
        )


@pytest.fixture
def policy(policy_path) -> PolicyEngine:
    return PolicyEngine.from_yaml(policy_path)


def leaky_map(micro_corpus):
    selection = StrategySelection("leaky", "direct", "none")
    return {selection: LeakyPipeline(micro_corpus)}


def test_sweep_checks_every_question_role_pipeline_cell(policy, micro_corpus) -> None:
    report = run_leakage_sweep(leaky_map(micro_corpus), policy, QUESTIONS)
    assert report["cells_checked"] == len(QUESTIONS) * len(ROLES)
    assert report["pipelines"] == ["direct__leaky__none"]
    assert report["strategy_combinations"][0]["retrieval_strategy"] == "leaky"


def test_sweep_catches_and_names_a_pipeline_leak(policy, micro_corpus) -> None:
    report = run_leakage_sweep(leaky_map(micro_corpus), policy, QUESTIONS)
    assert report["violations"] > 0
    assert sweep_passed(report) is False
    first = report["violation_details"][0]
    assert first["pipeline_id"] == "direct__leaky__none"
    assert first["chunk_id"] in {chunk.chunk_id for chunk in micro_corpus}


def test_employee_empty_visibility_is_detected_as_leak_for_buggy_pipeline(
    policy, micro_corpus
) -> None:
    report = run_leakage_sweep(leaky_map(micro_corpus), policy, QUESTIONS)
    assert report["per_role"]["employee"]["violations"] == (
        len(QUESTIONS) * len(micro_corpus)
    )
    assert report["per_role"]["c_level"]["violations"] == 0


def test_breakdown_names_the_full_pipeline(policy, micro_corpus) -> None:
    lines = breakdown_lines(run_leakage_sweep(leaky_map(micro_corpus), policy, QUESTIONS))
    assert lines[0] == "direct__leaky__none"
    assert any(line.startswith("employee") for line in lines)
    assert lines[-1].startswith("ALL")


def test_sweep_builder_forms_the_requested_cartesian_product(
    monkeypatch, micro_corpus, tmp_path
) -> None:
    monkeypatch.setattr(sweep, "read_chunks", lambda _path: list(micro_corpus))
    settings = replace(Settings(), artifacts_dir=tmp_path, embedder="fake")
    pipelines = sweep.build_sweep_pipelines(
        settings,
        retrievers=["naive"],
        query_strategies=["direct", "hyde", "react"],
        rerankers=["none", "llm"],
    )
    assert len(pipelines) == 6
    assert {selection.pipeline_id for selection in pipelines} == {
        f"{query}__naive__{reranker}"
        for query in ["direct", "hyde", "react"]
        for reranker in ["none", "llm"]
    }


def test_graph_is_omitted_by_default_and_refused_when_explicit(monkeypatch, settings) -> None:
    monkeypatch.setattr(sweep, "bolt_is_open", lambda: False)
    graph_names = {"graph_keyword", "graph_entity", "graph_temporal"}
    assert graph_names.isdisjoint(sweep.sweepable_retrievers())
    assert set(sweep.offline_retrievers()) == set(available_retrievers()) - graph_names
    with pytest.raises(RuntimeError, match="docker start finrag-neo4j"):
        sweep.build_sweep_pipelines(
            settings,
            retrievers=["graph_entity"],
            query_strategies=["direct"],
            rerankers=["none"],
        )


def test_the_offline_llm_never_generates() -> None:
    with pytest.raises(RuntimeError):
        sweep.FallbackOnlyLLM().generate("rank these")


def _vertex_settings(tmp_path) -> Settings:
    return replace(
        Settings(),
        artifacts_dir=tmp_path,
        embedder="vertex",
        vertex_project="demo-project",
    )


def test_a_cold_cache_names_the_question_instead_of_calling_the_api(tmp_path) -> None:
    embedder = sweep.build_sweep_embedder(_vertex_settings(tmp_path))
    with pytest.raises(RuntimeError) as caught:
        embedder.embed_texts(["what was the campaign spend in Q3"])
    message = str(caught.value)
    assert "campaign spend in Q3" in message
    assert "finrag eval --retriever dense" in message


def test_a_warm_cache_answers_without_the_backend(tmp_path) -> None:
    from finrag.retrieval.embed import EMBED_DIM, FakeEmbedder
    from finrag.retrieval.embed_cache import CachingEmbedder, EmbeddingCache

    settings = _vertex_settings(tmp_path)
    cache = EmbeddingCache.load(
        settings.embed_cache_path, settings.vertex_embed_model, EMBED_DIM
    )
    CachingEmbedder(FakeEmbedder(), cache).embed_texts(["a cached question"])
    assert sweep.build_sweep_embedder(settings).embed_texts(
        ["a cached question"]
    ).shape == (1, EMBED_DIM)


def test_fake_embedder_needs_no_cache(tmp_path) -> None:
    settings = replace(Settings(), artifacts_dir=tmp_path, embedder="fake")
    assert sweep.build_sweep_embedder(settings).embed_texts(["new text"]).shape[0] == 1
