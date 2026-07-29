"""Mode name to retriever: the one place that knows what "naive" means.

The CLI, the eval harness and the API all pick a retriever by a short string
("naive", "bm25", later "dense", "hybrid", "graph"). They should not import
retriever classes directly, or every new mode would mean edits in three places.
Instead they load the artifacts once into an `IndexBundle`, then call a factory
from `FACTORIES`.

Adding a mode is two lines here: write the factory, put it in the dict.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from finrag.models import Chunk
from finrag.retrieval.base import Retriever
from finrag.retrieval.bm25 import BM25Retriever
from finrag.retrieval.naive import NaiveRetriever


@dataclass
class IndexBundle:
    """Everything a retriever might need, loaded from `artifacts/` once.

    `chunks` and `bm25_dir` are filled in today. The other paths are here so the
    later modes can take an index without changing this signature and breaking
    every caller.
    """

    chunks: list[Chunk] = field(default_factory=list)
    bm25_dir: Path | None = None
    dense_path: Path | None = None
    graph_path: Path | None = None


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


FACTORIES: dict[str, Callable[[IndexBundle], Retriever]] = {
    "bm25": make_bm25,
    "naive": make_naive,
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
