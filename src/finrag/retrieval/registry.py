"""Mode name to retriever: the one place that knows what "naive" means.

The CLI, the eval harness and the API all pick a retriever by a short string
("naive", later "bm25", "dense", "hybrid", "graph"). They should not import
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
from finrag.retrieval.naive import NaiveRetriever


@dataclass
class IndexBundle:
    """Everything a retriever might need, loaded from `artifacts/` once.

    Only `chunks` is filled in today. The paths are here so the later modes can
    take an index without changing this signature and breaking every caller.
    """

    chunks: list[Chunk] = field(default_factory=list)
    bm25_dir: Path | None = None
    dense_path: Path | None = None
    graph_path: Path | None = None


def make_naive(bundle: IndexBundle) -> Retriever:
    """The term-overlap baseline. Needs nothing but the chunks."""
    return NaiveRetriever(bundle.chunks)


FACTORIES: dict[str, Callable[[IndexBundle], Retriever]] = {
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
