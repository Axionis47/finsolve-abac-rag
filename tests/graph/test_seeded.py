"""Embedding-seeded graph retrieval: score math, layers, and role masks."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from finrag.graph.build_structural import extract
from finrag.graph.schema import LAYER_ENTITY, LAYER_TEMPORAL
from finrag.graph.seeded import (
    GraphLayerUnavailableError,
    SeededGraphRetriever,
    traverse_from_seeds,
)
from finrag.graph.store import GraphStore
from finrag.models import Chunk, Hit
from finrag.retrieval.base import Retriever


def _hit(chunk: Chunk, rank: int) -> Hit:
    return Hit(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        section_slug=chunk.section_slug,
        rank=rank,
        score=1.0 / rank,
        retriever="scripted_dense",
        owner_dept=chunk.owner_dept,
        doc_type=chunk.doc_type,
        sensitivity=chunk.sensitivity,
        allowed_roles=chunk.allowed_roles,
        text=chunk.text,
    )


class ScriptedDense:
    name = "scripted_dense"

    def __init__(self, chunks: list[Chunk]) -> None:
        self.hits = [_hit(chunk, rank) for rank, chunk in enumerate(chunks, start=1)]
        self.calls: list[tuple[str, str, int]] = []

    def search(self, query: str, role: str, k: int = 10) -> list[Hit]:
        self.calls.append((query, role, k))
        return list(self.hits[:k])


class ScriptedStore:
    namespace = "FinragTest"

    def __init__(
        self,
        paths: list[dict[str, object]] | None = None,
        layer_counts: dict[str, int] | None = None,
    ) -> None:
        self.paths = list(paths or [])
        self._layer_counts = dict(layer_counts or {LAYER_ENTITY: 1, LAYER_TEMPORAL: 1})
        self.calls: list[tuple[str, dict[str, object]]] = []

    def layer_counts(self) -> dict[str, int]:
        return dict(self._layer_counts)

    def read(self, query: str, params: dict[str, object] | None = None):
        self.calls.append((query, dict(params or {})))
        if "min(length(path)) AS hops" in query:
            return list(self.paths)
        return []


def test_missing_entity_layer_names_the_build_command(micro_corpus: list[Chunk]) -> None:
    with pytest.raises(GraphLayerUnavailableError, match="finrag graph --llm"):
        SeededGraphRetriever(
            "graph_entity",
            LAYER_ENTITY,
            micro_corpus,
            ScriptedDense(micro_corpus),
            ScriptedStore(layer_counts={LAYER_TEMPORAL: 4}),
        )


def test_seeded_retriever_satisfies_the_shared_contract(micro_corpus: list[Chunk]) -> None:
    retriever = SeededGraphRetriever(
        "graph_temporal",
        LAYER_TEMPORAL,
        micro_corpus,
        ScriptedDense(micro_corpus),
        ScriptedStore(),
    )
    assert isinstance(retriever, Retriever)


def test_scores_merge_direct_seeds_and_shortest_path_contributions(
    micro_corpus: list[Chunk],
) -> None:
    first, second, expanded = micro_corpus[:3]
    store = ScriptedStore(
        paths=[
            {
                "seed_chunk_id": first.chunk_id,
                "seed_rank": 1,
                "node_id": "chunk:" + expanded.chunk_id,
                "hops": 2,
            },
            {
                "seed_chunk_id": second.chunk_id,
                "seed_rank": 2,
                "node_id": "chunk:" + expanded.chunk_id,
                "hops": 2,
            },
        ]
    )
    retriever = SeededGraphRetriever(
        "graph_entity",
        LAYER_ENTITY,
        micro_corpus,
        ScriptedDense([first, second]),
        store,
    )

    hits = retriever.search("linked facts", "c_level", k=3)
    by_id = {hit.chunk_id: hit for hit in hits}

    assert by_id[first.chunk_id].score == 1.0
    assert by_id[second.chunk_id].score == 0.5
    assert by_id[expanded.chunk_id].score == pytest.approx(1 / 3 + 1 / 6)
    assert by_id[expanded.chunk_id].debug["graph_contribution"] == pytest.approx(0.5)
    assert by_id[expanded.chunk_id].debug["graph_layer"] == LAYER_ENTITY
    assert by_id[expanded.chunk_id].debug["max_hops"] == 3


def test_role_mask_removes_bad_seeds_and_bad_targets(micro_corpus: list[Chunk]) -> None:
    forbidden = micro_corpus[0]
    visible = micro_corpus[2]
    store = ScriptedStore(
        paths=[
            {
                "seed_chunk_id": visible.chunk_id,
                "seed_rank": 2,
                "node_id": "chunk:" + forbidden.chunk_id,
                "hops": 2,
            }
        ]
    )
    retriever = SeededGraphRetriever(
        "graph_temporal",
        LAYER_TEMPORAL,
        micro_corpus,
        ScriptedDense([forbidden, visible]),
        store,
    )

    hits = retriever.search("quarter spend", "marketing", k=5)

    assert [hit.chunk_id for hit in hits] == [visible.chunk_id]
    path_call = next(call for call in store.calls if "min(length(path))" in call[0])
    params = path_call[1]
    assert [seed["chunk_id"] for seed in params["seeds"]] == [visible.chunk_id]
    assert forbidden.chunk_id not in params["visible_ids"]
    assert hits[0].debug["n_blocked_late"] == 1


def test_ties_break_by_chunk_id(micro_corpus: list[Chunk]) -> None:
    seeds = [micro_corpus[3], micro_corpus[2]]
    scripted = ScriptedDense(seeds)
    scripted.hits[1] = _hit(seeds[1], 1)
    retriever = SeededGraphRetriever(
        "graph_temporal",
        LAYER_TEMPORAL,
        micro_corpus,
        scripted,
        ScriptedStore(),
    )
    hits = retriever.search("same score", "marketing", k=2)
    assert [hit.chunk_id for hit in hits] == sorted(hit.chunk_id for hit in hits)


@pytest.mark.parametrize("role", ["marketing", "c_level"])
def test_live_walk_uses_only_the_selected_layer_and_visible_edges(
    graph_store: GraphStore,
    micro_corpus: list[Chunk],
    role: str,
) -> None:
    nodes, edges = extract(micro_corpus)
    graph_store.write_graph(nodes, edges)
    visible_chunks = [chunk for chunk in micro_corpus if role in chunk.allowed_roles]
    seeds = [_hit(chunk, rank) for rank, chunk in enumerate(visible_chunks[:2], start=1)]

    result = traverse_from_seeds(
        graph_store,
        seeds,
        role,
        micro_corpus,
        LAYER_TEMPORAL,
        collect_walked=True,
    )

    visible_ids = {chunk.chunk_id for chunk in visible_chunks}
    assert all(edge.layer == LAYER_TEMPORAL for edge in result.walked)
    assert all(edge.source_chunk_id in visible_ids for edge in result.walked)
    assert all(path.chunk_id in visible_ids for path in result.paths)
