"""Offline ABAC leakage checks across composed strategy pipelines.

The sweep deliberately uses a failing model backend and a cache-only real
embedder. Query strategies therefore exercise their documented direct fallback,
the LLM reranker preserves its input order, and no sweep can unexpectedly bill
an external service. Graph retrieval is included only when Neo4j is available.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product
from typing import Any

import numpy as np

from finrag.config import Settings
from finrag.graph.store import bolt_is_open
from finrag.ingest.pipeline import read_chunks
from finrag.models import ROLES
from finrag.pipeline import SearchPipeline, StrategyRuntime, StrategySelection
from finrag.policy.engine import PolicyEngine
from finrag.querying.registry import available_query_strategies
from finrag.reranking.registry import available_rerankers
from finrag.retrieval.embed import Embedder, build_embedder
from finrag.retrieval.registry import IndexBundle, available_retrievers
from finrag.retrieval.registry import retriever_requirements
from finrag.strategy import Requirement

MAX_RECORDED_VIOLATIONS = 100


class FallbackOnlyLLM:
    """A backend that proves fallback paths without making an API call."""

    name = "fallback-only"

    def generate(self, prompt: str, system: str | None = None) -> str:
        raise RuntimeError("the leakage sweep runs offline: no model is called")


class CacheOnlyEmbedder:
    """Serve saved query vectors and fail instead of calling a paid backend."""

    def __init__(self, cached: Any) -> None:
        self.cache = cached.cache
        self.name = f"cache-only-{cached.name}"
        self.model = cached.model
        self.dim = cached.dim

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, value in enumerate(texts):
            vector = self.cache.get(value)
            if vector is None:
                raise RuntimeError(
                    "The leakage sweep is offline and has no cached "
                    f"'{self.model}' vector for this text:\n  {value[:120]!r}\n"
                    "Run `finrag eval --retriever dense` once to fill the cache, "
                    "or set FINRAG_EMBEDDER=fake."
                )
            out[row] = vector
        return out


def build_sweep_embedder(settings: Settings) -> Embedder:
    embedder = build_embedder(settings)
    if settings.embedder == "fake":
        return embedder
    return CacheOnlyEmbedder(embedder)


def offline_retrievers() -> list[str]:
    """Retrievers whose backing service can be exercised without Neo4j."""
    return [
        name
        for name in available_retrievers()
        if Requirement.GRAPH not in retriever_requirements(name)
    ]


def sweepable_retrievers() -> list[str]:
    if bolt_is_open():
        return available_retrievers()
    return offline_retrievers()


def build_sweep_pipelines(
    settings: Settings,
    *,
    retrievers: Sequence[str] | None = None,
    query_strategies: Sequence[str] | None = None,
    rerankers: Sequence[str] | None = None,
) -> dict[StrategySelection, SearchPipeline]:
    """Build the requested Cartesian product over one shared artifact bundle."""
    explicit_retrievers = retrievers is not None
    retriever_names = tuple(retrievers or sweepable_retrievers())
    query_names = tuple(query_strategies or available_query_strategies())
    reranker_names = tuple(rerankers or available_rerankers())

    _validate_names("retrieval", retriever_names, available_retrievers())
    _validate_names("query", query_names, available_query_strategies())
    _validate_names("reranking", reranker_names, available_rerankers())
    _check_graph_retrievers(retriever_names, explicit=explicit_retrievers)

    bundle = IndexBundle(
        chunks=read_chunks(settings.chunks_path),
        bm25_dir=settings.bm25_dir,
        dense_path=settings.dense_path,
    )
    if any(Requirement.EMBEDDER in retriever_requirements(name) for name in retriever_names):
        bundle.embedder = build_sweep_embedder(settings)

    fallback_llm = FallbackOnlyLLM()
    runtime = StrategyRuntime(
        bundle=bundle,
        settings=settings,
        answer_llm=fallback_llm,
        query_llm=fallback_llm,
        rerank_llm=fallback_llm,
    )
    pipelines: dict[StrategySelection, SearchPipeline] = {}
    for query_name, retriever_name, reranker_name in product(
        query_names, retriever_names, reranker_names
    ):
        selection = StrategySelection(
            retrieval_strategy=retriever_name,
            query_strategy=query_name,
            reranking_strategy=reranker_name,
        )
        pipelines[selection] = SearchPipeline(selection, runtime)
    return pipelines


def _validate_names(kind: str, selected: Sequence[str], available: Sequence[str]) -> None:
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(
            f"unknown {kind} strategy '{unknown[0]}'. Known strategies: "
            f"{', '.join(available)}"
        )


def _check_graph_retrievers(retrievers: Sequence[str], *, explicit: bool) -> None:
    graph_retrievers = [
        name for name in retrievers if Requirement.GRAPH in retriever_requirements(name)
    ]
    if not graph_retrievers or bolt_is_open():
        return
    if not explicit:
        return
    raise RuntimeError(
        f"{', '.join(graph_retrievers)} need Neo4j and bolt is not answering. Start it with "
        "`docker start finrag-neo4j`, or leave --retrievers off."
    )


def _new_counts() -> dict[str, int]:
    return {"cells": 0, "hits": 0, "violations": 0, "empty_cells": 0}


def _record(counts: dict[str, int], n_hits: int, n_violations: int) -> None:
    counts["cells"] += 1
    counts["hits"] += n_hits
    counts["violations"] += n_violations
    if n_hits == 0:
        counts["empty_cells"] += 1


def _sum_counts(blocks: Sequence[dict[str, int]]) -> dict[str, int]:
    total = _new_counts()
    for block in blocks:
        for key in total:
            total[key] += block[key]
    return total


def run_leakage_sweep(
    pipelines: dict[StrategySelection, SearchPipeline],
    policy: PolicyEngine,
    gold: Sequence[dict[str, Any]],
    roles: Sequence[str] = ROLES,
    k: int = 10,
) -> dict[str, Any]:
    """Ask every question as every role through every composed pipeline."""
    ordered = sorted(pipelines.items(), key=lambda item: item[0].pipeline_id)
    pipeline_ids = [selection.pipeline_id for selection, _ in ordered]
    roles = tuple(roles)

    breakdown = {
        role: {pipeline_id: _new_counts() for pipeline_id in pipeline_ids}
        for role in roles
    }
    details: list[dict[str, Any]] = []
    n_violations = 0

    for entry in gold:
        for role in roles:
            for selection, pipeline in ordered:
                hits = pipeline.search(entry["question"], role, k=k).hits
                found = policy.replay(role, hits)
                pipeline_id = selection.pipeline_id
                _record(breakdown[role][pipeline_id], len(hits), len(found))
                n_violations += len(found)
                for violation in found:
                    if len(details) < MAX_RECORDED_VIOLATIONS:
                        details.append(
                            {
                                "question_id": entry["id"],
                                "gold_role": entry["role"],
                                "asked_as_role": role,
                                "pipeline_id": pipeline_id,
                                "retrieval_strategy": selection.retrieval_strategy,
                                "query_strategy": selection.query_strategy,
                                "reranking_strategy": selection.reranking_strategy,
                                "chunk_id": violation.chunk_id,
                                "denied_by_rule": violation.rule_name,
                            }
                        )

    per_role = {
        role: _sum_counts(list(breakdown[role].values())) for role in roles
    }
    per_pipeline = {
        pipeline_id: _sum_counts([breakdown[role][pipeline_id] for role in roles])
        for pipeline_id in pipeline_ids
    }
    totals = _sum_counts(list(per_role.values()))
    selections = [
        {
            "pipeline_id": selection.pipeline_id,
            "retrieval_strategy": selection.retrieval_strategy,
            "query_strategy": selection.query_strategy,
            "reranking_strategy": selection.reranking_strategy,
        }
        for selection, _ in ordered
    ]
    return {
        "pipeline_id": "sweep",
        "k": k,
        "roles": list(roles),
        "pipelines": pipeline_ids,
        "strategy_combinations": selections,
        "n_questions": len(gold),
        "cells_checked": totals["cells"],
        "hits_inspected": totals["hits"],
        "violations": n_violations,
        "violations_recorded": len(details),
        "empty_cells": totals["empty_cells"],
        "model_policy": "fallback only; no model API calls",
        "per_role": per_role,
        "per_pipeline": per_pipeline,
        "breakdown": breakdown,
        "violation_details": details,
    }


def sweep_passed(report: dict[str, Any]) -> bool:
    return report["violations"] == 0


def breakdown_lines(report: dict[str, Any]) -> list[str]:
    """Render a readable block per pipeline instead of one unbounded table."""
    lines: list[str] = []
    for pipeline_id in report["pipelines"]:
        lines.append(pipeline_id)
        lines.append(f"{'role':<14}{'hits/leaks':>14}")
        for role in report["roles"]:
            cell = report["breakdown"][role][pipeline_id]
            value = f"{cell['hits']}/{cell['violations']}"
            lines.append(f"{role:<14}{value:>14}")
        total = report["per_pipeline"][pipeline_id]
        value = f"{total['hits']}/{total['violations']}"
        lines.append(f"{'ALL':<14}{value:>14}")
        lines.append("")
    if lines:
        lines.pop()
    return lines
