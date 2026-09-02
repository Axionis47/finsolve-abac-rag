"""Explicit registry for independently selectable retrieval strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from finrag.graph.retriever import GraphRetriever
from finrag.graph.schema import LAYER_ENTITY, LAYER_TEMPORAL
from finrag.graph.seeded import SeededGraphRetriever
from finrag.graph.store import GraphStore
from finrag.models import Chunk
from finrag.retrieval.base import Retriever
from finrag.retrieval.bm25 import BM25Retriever
from finrag.retrieval.dense import DenseRetriever
from finrag.retrieval.embed import Embedder
from finrag.retrieval.hybrid import HybridRetriever
from finrag.retrieval.naive import NaiveRetriever
from finrag.strategy import Requirement, StrategySpec

DEFAULT_RETRIEVER = "bm25"


@dataclass
class IndexBundle:
    """Everything a retriever might need, loaded from `artifacts/` once.

    Optional dependency objects are injected only when registry metadata says
    the selected pipeline needs them.
    """

    chunks: list[Chunk] = field(default_factory=list)
    bm25_dir: Path | None = None
    dense_path: Path | None = None
    embedder: Embedder | None = None
    graph_store: GraphStore | None = None


def make_naive(bundle: IndexBundle) -> Retriever:
    """The term-overlap baseline. Needs nothing but the chunks."""
    return NaiveRetriever(bundle.chunks)


def make_bm25(bundle: IndexBundle) -> Retriever:
    """Lexical BM25. Uses the index ingest wrote, when the bundle names one.

    With `bm25_dir` set the saved artifact is loaded, which is the real path the
    CLI and API take. With it unset the retriever builds its own index in memory,
    which keeps small in-process callers and tests from needing a file on disk.
    """
    return BM25Retriever(bundle.chunks, index_dir=bundle.bm25_dir)


def make_dense(bundle: IndexBundle) -> Retriever:
    """Cosine search over embeddings. Needs an embedder for the query side.

    With `dense_path` set the saved matrix is loaded, which is the path the CLI
    takes. With it unset the chunks are embedded here and now, which is what the
    tests do with a fake embedder.
    """
    if bundle.embedder is None:
        raise ValueError(
            "dense retrieval needs an embedder in the bundle. Build one with "
            "finrag.retrieval.embed.build_embedder(settings)."
        )
    return DenseRetriever(bundle.chunks, bundle.embedder, dense_path=bundle.dense_path)


def make_hybrid(bundle: IndexBundle) -> Retriever:
    """BM25 and dense fused by RRF. Builds both sub-retrievers from the bundle.

    They are built here rather than passed in so that `hybrid` loads exactly the
    same artifacts `bm25` and `dense` load on their own, which is what makes the
    three modes comparable in the eval table.
    """
    return HybridRetriever(bundle.chunks, make_bm25(bundle), make_dense(bundle))


def make_graph_keyword(bundle: IndexBundle) -> Retriever:
    """Two-hop traversal over the Neo4j graph `finrag graph` built.

    The graph lives in Neo4j rather than in `artifacts/`. The runtime normally
    injects one lazily-created store when this retriever is selected.
    """
    return GraphRetriever(bundle.chunks, bundle.graph_store or GraphStore())


def _make_seeded_graph(bundle: IndexBundle, name: str, layer: str) -> Retriever:
    """Dense entry followed by traversal over one explicit graph layer."""
    store = bundle.graph_store or GraphStore()
    return SeededGraphRetriever(
        name=name,
        layer=layer,
        chunks=bundle.chunks,
        dense=make_dense(bundle),
        store=store,
    )


def make_graph_entity(bundle: IndexBundle) -> Retriever:
    return _make_seeded_graph(bundle, "graph_entity", LAYER_ENTITY)


def make_graph_temporal(bundle: IndexBundle) -> Retriever:
    return _make_seeded_graph(bundle, "graph_temporal", LAYER_TEMPORAL)


RETRIEVERS: dict[str, StrategySpec[Retriever, IndexBundle]] = {
    "naive": StrategySpec("naive", "Simple word-overlap baseline.", make_naive),
    "bm25": StrategySpec("bm25", "Lexical BM25 over the saved index.", make_bm25),
    "dense": StrategySpec(
        "dense",
        "Cosine similarity over text embeddings.",
        make_dense,
        frozenset({Requirement.EMBEDDER}),
    ),
    "hybrid": StrategySpec(
        "hybrid",
        "Reciprocal Rank Fusion of BM25 and dense retrieval.",
        make_hybrid,
        frozenset({Requirement.EMBEDDER}),
    ),
    "graph_keyword": StrategySpec(
        "graph_keyword",
        "Keyword-seeded path-count traversal over all graph layers.",
        make_graph_keyword,
        frozenset({Requirement.GRAPH}),
    ),
    "graph_entity": StrategySpec(
        "graph_entity",
        "Dense seeds expanded through entity and claim relationships.",
        make_graph_entity,
        frozenset({Requirement.EMBEDDER, Requirement.GRAPH}),
    ),
    "graph_temporal": StrategySpec(
        "graph_temporal",
        "Dense seeds expanded through quarter and metric relationships.",
        make_graph_temporal,
        frozenset({Requirement.EMBEDDER, Requirement.GRAPH}),
    ),
}


def available_retrievers() -> list[str]:
    return sorted(RETRIEVERS)


def retriever_catalog() -> list[dict[str, object]]:
    return [RETRIEVERS[name].public_view() for name in available_retrievers()]


def retriever_requirements(name: str) -> frozenset[Requirement]:
    return _spec(name).requirements


def build_retriever(name: str, bundle: IndexBundle) -> Retriever:
    return _spec(name).factory(bundle)


def _spec(name: str) -> StrategySpec[Retriever, IndexBundle]:
    if name not in RETRIEVERS:
        known = ", ".join(available_retrievers())
        raise KeyError(f"unknown retrieval strategy '{name}'. Known strategies: {known}")
    return RETRIEVERS[name]
