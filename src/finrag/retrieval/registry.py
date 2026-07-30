"""Mode name to retriever: the one place that knows what "naive" means.

The CLI, the eval harness and the API all pick a retriever by a short string
("naive", "bm25", "dense", "hybrid", "hybrid_rerank", "graphrag", "react"). They should not import
retriever classes directly, or every new mode would mean edits in three places.
Instead they load the artifacts once into an `IndexBundle`, then call a factory
from `FACTORIES`.

Adding a mode is two lines here: write the factory, put it in the dict.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from finrag.graph.retriever import GraphRetriever
from finrag.graph.store import GraphStore
from finrag.llm.base import LLMBackend
from finrag.models import Chunk
from finrag.react import REACT_BASE_MODE, ReactAgent
from finrag.react_retriever import ReactRetriever
from finrag.retrieval.base import Retriever
from finrag.retrieval.bm25 import BM25Retriever
from finrag.retrieval.dense import DenseRetriever
from finrag.retrieval.embed import Embedder
from finrag.retrieval.hybrid import HybridRetriever
from finrag.retrieval.naive import NaiveRetriever
from finrag.retrieval.rerank import LLMReranker, RerankRetriever


@dataclass
class IndexBundle:
    """Everything a retriever might need, loaded from `artifacts/` once.

    `chunks`, `bm25_dir`, `dense_path`, `embedder` and `llm` are the ones anything
    reads. `graph_path` is not: the graph ended up in Neo4j rather than in a file,
    so `make_graphrag` opens a `GraphStore` instead. The field stays because
    removing it would edit every caller for nothing.

    `embedder` and `llm` are objects rather than paths because building either
    may open a Vertex client, and only the modes that need one should pay for
    that. The caller sets them when it asks for such a mode; the rest leave them
    None.
    """

    chunks: list[Chunk] = field(default_factory=list)
    bm25_dir: Path | None = None
    dense_path: Path | None = None
    graph_path: Path | None = None
    embedder: Embedder | None = None
    llm: LLMBackend | None = None


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
            "dense mode needs an embedder in the bundle. Build one with "
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


def make_graphrag(bundle: IndexBundle) -> Retriever:
    """Two-hop traversal over the Neo4j graph `finrag graph` built.

    The graph lives in Neo4j rather than in `artifacts/`, so `graph_path` stays
    unused and there is nothing to load here. `GraphStore()` opens a driver, not a
    connection, so a caller that never asks for this mode pays nothing for the
    import; the first search is what needs the container up.
    """
    return GraphRetriever(bundle.chunks, GraphStore())


def make_hybrid_rerank(bundle: IndexBundle) -> Retriever:
    """Hybrid, then one Gemini call to reorder the top twenty."""
    if bundle.llm is None:
        raise ValueError(
            "hybrid_rerank mode needs an llm in the bundle. Build one with "
            "finrag.llm.vertex.build_llm(settings)."
        )
    return RerankRetriever(bundle.chunks, make_hybrid(bundle), LLMReranker(bundle.llm))


def make_react(bundle: IndexBundle) -> Retriever:
    """The ReAct agent, searching with whichever mode `REACT_BASE_MODE` names.

    The base retriever is built HERE, from the registry, rather than inside the
    agent. That is the ABAC guarantee: the agent's only way to reach a chunk is a
    retriever this file built, which masks before it ranks, so the agent inherits
    the invariant instead of restating it.
    """
    if bundle.llm is None:
        raise ValueError(
            "react mode needs an llm in the bundle. Build one with "
            "finrag.llm.vertex.build_agent_llm(settings)."
        )
    base = build_retriever(REACT_BASE_MODE, bundle)
    return ReactRetriever(bundle.chunks, ReactAgent(bundle.chunks, base, bundle.llm))


FACTORIES: dict[str, Callable[[IndexBundle], Retriever]] = {
    "bm25": make_bm25,
    "dense": make_dense,
    "graphrag": make_graphrag,
    "hybrid": make_hybrid,
    "hybrid_rerank": make_hybrid_rerank,
    "naive": make_naive,
    "react": make_react,
}


def available_modes() -> list[str]:
    """Mode names a caller may ask for, in a stable order for menus and help."""
    return sorted(FACTORIES)


def build_retriever(mode: str, bundle: IndexBundle) -> Retriever:
    """Build the retriever for `mode`, or say plainly which modes exist."""
    if mode not in FACTORIES:
        known = ", ".join(available_modes())
        raise KeyError(f"unknown retrieval mode '{mode}'. Known modes: {known}")
    return FACTORIES[mode](bundle)
