"""Compose one query strategy, retriever and reranker into a search pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
from finrag.config import Settings
from finrag.graph.store import GraphStore, bolt_is_open
from finrag.graph.seeded import GraphLayerUnavailableError
from finrag.llm.base import LLMBackend
from finrag.models import Hit
from finrag.querying.base import QueryRequest, QueryResult
from finrag.querying.registry import (
    DEFAULT_QUERY_STRATEGY,
    QueryBundle,
    build_query_strategy,
    query_strategy_catalog,
    query_strategy_requirements,
)
from finrag.reranking.registry import (
    DEFAULT_RERANKER,
    RerankingBundle,
    build_reranker,
    reranker_catalog,
    reranker_requirements,
)
from finrag.retrieval.base import visible_ids
from finrag.retrieval.embed import Embedder, build_embedder
from finrag.retrieval.registry import (
    DEFAULT_RETRIEVER,
    IndexBundle,
    build_retriever,
    retriever_catalog,
    retriever_requirements,
)
from finrag.strategy import Requirement


class StrategyUnavailableError(RuntimeError):
    """A selected strategy's external dependency is unavailable."""


class StrategyContractError(RuntimeError):
    """A strategy violated a plug-in contract and cannot safely continue."""


@dataclass(frozen=True)
class StrategySelection:
    retrieval_strategy: str = DEFAULT_RETRIEVER
    query_strategy: str = DEFAULT_QUERY_STRATEGY
    reranking_strategy: str = DEFAULT_RERANKER

    @property
    def pipeline_id(self) -> str:
        return (
            f"{self.query_strategy}__{self.retrieval_strategy}__"
            f"{self.reranking_strategy}"
        )


@dataclass
class StrategyRuntime:
    """Artifacts and lazily-created backends shared by composed pipelines."""

    bundle: IndexBundle
    settings: Settings | None = None
    answer_llm: LLMBackend | None = None
    query_llm: LLMBackend | None = None
    rerank_llm: LLMBackend | None = None

    def prepare(self, requirements: frozenset[Requirement]) -> None:
        if Requirement.EMBEDDER in requirements and self.bundle.embedder is None:
            try:
                self.bundle.embedder = build_embedder(self._settings())
            except RuntimeError as error:
                raise StrategyUnavailableError(str(error)) from error
        if Requirement.GRAPH in requirements and self.bundle.graph_store is None:
            if not bolt_is_open():
                raise StrategyUnavailableError(
                    "the selected graph retriever needs Neo4j, but it is not answering. "
                    "Start it with `docker start finrag-neo4j`."
                )
            self.bundle.graph_store = GraphStore()
        if Requirement.QUERY_LLM in requirements and self.query_llm is None:
            try:
                self.query_llm = self._specialised_llm("query")
            except RuntimeError as error:
                raise StrategyUnavailableError(str(error)) from error
        if Requirement.RERANK_LLM in requirements and self.rerank_llm is None:
            try:
                self.rerank_llm = self._specialised_llm("rerank")
            except RuntimeError as error:
                raise StrategyUnavailableError(str(error)) from error

    def _specialised_llm(self, kind: str) -> LLMBackend:
        # Offline/test backends are safe to reuse and keep every path credential-free.
        if self.answer_llm is not None and getattr(self.answer_llm, "name", "") != "vertex":
            return self.answer_llm
        from finrag.llm.vertex import build_query_llm, build_rerank_llm

        return build_query_llm(self._settings()) if kind == "query" else build_rerank_llm(
            self._settings()
        )

    def _settings(self) -> Settings:
        if self.settings is None:
            self.settings = Settings.from_env()
        return self.settings


class SearchPipeline:
    """The stable execution path shared by API, CLI, evaluation and sweep."""

    def __init__(self, selection: StrategySelection, runtime: StrategyRuntime) -> None:
        self.selection = selection
        self.runtime = runtime
        requirements = pipeline_requirements(selection)
        runtime.prepare(requirements)
        try:
            self.retriever = build_retriever(selection.retrieval_strategy, runtime.bundle)
        except GraphLayerUnavailableError as error:
            raise StrategyUnavailableError(str(error)) from error
        self.query_strategy = build_query_strategy(
            selection.query_strategy,
            QueryBundle(chunks=runtime.bundle.chunks, llm=runtime.query_llm),
        )
        self.reranker = build_reranker(
            selection.reranking_strategy,
            RerankingBundle(llm=runtime.rerank_llm),
        )

    @property
    def name(self) -> str:
        return self.selection.pipeline_id

    def search(self, question: str, role: str, k: int = 10) -> QueryResult:
        candidate_k = max(k, self.reranker.depth)
        result = self.query_strategy.execute(
            QueryRequest(question=question, role=role, k=candidate_k), self.retriever
        )

        allowed = visible_ids(self.runtime.bundle.chunks, role)
        candidates: list[Hit] = []
        seen: set[str] = set()
        for hit in result.hits:
            if hit.chunk_id in allowed and hit.chunk_id not in seen:
                candidates.append(hit)
                seen.add(hit.chunk_id)

        ordered = self.reranker.rerank(question, candidates)
        self._validate_reranker(candidates, ordered)

        hits: list[Hit] = []
        for rank, hit in enumerate(ordered[:k], start=1):
            debug = dict(hit.debug)
            debug.update(
                {
                    "pre_rerank_rank": hit.rank,
                    "query_strategy": self.selection.query_strategy,
                    "reranking_strategy": self.selection.reranking_strategy,
                }
            )
            hits.append(replace(hit, rank=rank, debug=debug))
        result.hits = hits
        return result

    def _validate_reranker(self, before: list[Hit], after: list[Hit]) -> None:
        before_ids = [hit.chunk_id for hit in before]
        after_ids = [hit.chunk_id for hit in after]
        if len(before_ids) != len(after_ids) or sorted(before_ids) != sorted(after_ids):
            raise StrategyContractError(
                f"reranker '{self.selection.reranking_strategy}' must return exactly "
                "the input hits once each"
            )


def pipeline_requirements(selection: StrategySelection) -> frozenset[Requirement]:
    return frozenset(
        set(retriever_requirements(selection.retrieval_strategy))
        | set(query_strategy_requirements(selection.query_strategy))
        | set(reranker_requirements(selection.reranking_strategy))
    )


def strategy_catalog() -> dict[str, dict[str, object]]:
    return {
        "retrieval": {"default": DEFAULT_RETRIEVER, "items": retriever_catalog()},
        "query": {"default": DEFAULT_QUERY_STRATEGY, "items": query_strategy_catalog()},
        "reranking": {"default": DEFAULT_RERANKER, "items": reranker_catalog()},
    }


def load_runtime(
    settings: Settings,
    *,
    answer_llm: LLMBackend | None = None,
    embedder: Embedder | None = None,
) -> StrategyRuntime:
    """Load saved artifacts once; strategy-specific dependencies stay lazy."""
    from finrag.ingest.pipeline import read_chunks

    bundle = IndexBundle(
        chunks=read_chunks(settings.chunks_path),
        bm25_dir=settings.bm25_dir,
        dense_path=settings.dense_path,
        embedder=embedder,
    )
    return StrategyRuntime(bundle=bundle, settings=settings, answer_llm=answer_llm)


def report_filename_part(selection: StrategySelection) -> str:
    """Filesystem-safe pipeline id kept separate for report naming tests."""
    return selection.pipeline_id
