"""Composition matrix and invariants for the modular search pipeline."""

from __future__ import annotations

import json
import re

import pytest

from finrag.pipeline import (
    SearchPipeline,
    StrategyRuntime,
    StrategySelection,
    pipeline_requirements,
    strategy_catalog,
)
from finrag.querying.registry import available_query_strategies
from finrag.reranking.registry import available_rerankers
from finrag.retrieval.embed import FakeEmbedder
from finrag.retrieval.registry import (
    IndexBundle,
    available_retrievers,
    build_retriever,
)
from finrag.strategy import Requirement


class UniversalLLM:
    """Valid deterministic replies for HyDE, ReAct, and reranking prompts."""

    name = "universal"

    def generate(self, prompt: str, system: str | None = None) -> str:
        if system and "retrieval planner" in system:
            return '{"thought":"no extra search needed","action":"finish","evidence":[]}'
        if system and "rank search results" in system:
            match = re.search(r"Order all (\d+) passages", prompt)
            n = int(match.group(1)) if match else 0
            return json.dumps(list(range(1, n + 1)))
        return "campaign spend impressions"


class StubGraphStore:
    namespace = "FinragTest"

    def read(self, query, params=None):
        return []

    def layer_counts(self):
        return {"entity": 1, "temporal": 1}


def runtime(micro_corpus) -> StrategyRuntime:
    llm = UniversalLLM()
    return StrategyRuntime(
        bundle=IndexBundle(
            chunks=list(micro_corpus),
            embedder=FakeEmbedder(dim=32),
            graph_store=StubGraphStore(),
        ),
        answer_llm=llm,
        query_llm=llm,
        rerank_llm=llm,
    )


@pytest.mark.parametrize("retrieval_strategy", ["naive", "bm25", "dense", "hybrid"])
def test_direct_none_preserves_existing_retriever_hits_exactly(
    micro_corpus, retrieval_strategy
) -> None:
    shared = runtime(micro_corpus)
    expected = build_retriever(retrieval_strategy, shared.bundle).search(
        "campaign spend", "c_level", k=5
    )
    actual = SearchPipeline(
        StrategySelection(retrieval_strategy=retrieval_strategy), shared
    ).search("campaign spend", "c_level", k=5).hits
    assert [(hit.chunk_id, hit.rank, hit.score) for hit in actual] == [
        (hit.chunk_id, hit.rank, hit.score) for hit in expected
    ]


@pytest.mark.parametrize("query_strategy", available_query_strategies())
@pytest.mark.parametrize("retrieval_strategy", available_retrievers())
@pytest.mark.parametrize("reranking_strategy", available_rerankers())
def test_every_registered_strategy_combination_executes(
    micro_corpus, query_strategy, retrieval_strategy, reranking_strategy
) -> None:
    selection = StrategySelection(
        retrieval_strategy=retrieval_strategy,
        query_strategy=query_strategy,
        reranking_strategy=reranking_strategy,
    )
    result = SearchPipeline(selection, runtime(micro_corpus)).search(
        "campaign spend", "c_level", k=3
    )
    assert len(result.hits) <= 3
    assert len({hit.chunk_id for hit in result.hits}) == len(result.hits)
    assert [hit.rank for hit in result.hits] == list(range(1, len(result.hits) + 1))


def test_registry_catalog_exposes_defaults_descriptions_and_requirements() -> None:
    catalog = strategy_catalog()
    assert catalog["retrieval"]["default"] == "bm25"
    assert catalog["query"]["default"] == "direct"
    assert catalog["reranking"]["default"] == "none"
    for axis in catalog.values():
        for item in axis["items"]:
            assert item["description"]
            assert isinstance(item["requirements"], list)


def test_dependency_requirements_are_derived_from_registry_metadata() -> None:
    assert pipeline_requirements(StrategySelection()) == frozenset()
    requirements = pipeline_requirements(
        StrategySelection("hybrid", "hyde", "llm")
    )
    assert requirements == frozenset(
        {Requirement.EMBEDDER, Requirement.QUERY_LLM, Requirement.RERANK_LLM}
    )
    assert pipeline_requirements(
        StrategySelection(retrieval_strategy="graph_entity")
    ) == frozenset({Requirement.EMBEDDER, Requirement.GRAPH})


def test_pipeline_identifier_orders_query_retrieval_then_reranking() -> None:
    assert StrategySelection("hybrid", "hyde", "llm").pipeline_id == (
        "hyde__hybrid__llm"
    )
