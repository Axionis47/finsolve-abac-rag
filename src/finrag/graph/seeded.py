"""Embedding-seeded traversal over one explicit graph layer.

Dense retrieval answers where to enter the graph. Neo4j then answers which
permitted chunks are connected through the selected relationship layer. The
same class backs the entity and temporal strategies so their only meaningful
difference is the graph they are allowed to walk.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from finrag.graph.schema import chunk_id_from_node_id, chunk_node_id
from finrag.graph.store import RELATIONSHIP_TYPE, GraphStore
from finrag.graph.traverse import WalkedEdge, _check_hops
from finrag.models import Chunk, Hit
from finrag.retrieval.base import Retriever, visible_ids

DEFAULT_SEED_K = 5
DEFAULT_MAX_HOPS = 3


class GraphLayerUnavailableError(RuntimeError):
    """The graph exists, but the selected strategy's edge layer does not."""


@dataclass(frozen=True)
class SeedPath:
    """The shortest permitted path from one dense seed to one chunk."""

    seed_chunk_id: str
    seed_rank: int
    chunk_id: str
    hops: int

    @property
    def contribution(self) -> float:
        return 1.0 / (self.seed_rank * (self.hops + 1))


@dataclass(frozen=True)
class SeedTraversalResult:
    paths: list[SeedPath] = field(default_factory=list)
    walked: list[WalkedEdge] = field(default_factory=list)
    n_blocked_late: int = 0


def _path_query(namespace: str, max_hops: int) -> str:
    """Shortest visible path for each seed-to-target pair in one layer."""
    return (
        "UNWIND $seeds AS seed_row "
        f"MATCH (seed:{namespace} {{id: seed_row.node_id}}) "
        f"MATCH path = (seed)-[rels:{RELATIONSHIP_TYPE}*1..{max_hops}]-(target:{namespace}) "
        "WHERE target.kind = 'chunk' "
        "  AND target.id <> seed.id "
        "  AND all(r IN rels WHERE r.layer = $layer "
        "      AND r.source_chunk_id IN $visible_ids) "
        "RETURN seed_row.chunk_id AS seed_chunk_id, "
        "  seed_row.rank AS seed_rank, target.id AS node_id, "
        "  min(length(path)) AS hops "
        "ORDER BY seed_rank ASC, node_id ASC"
    )


def _walked_query(namespace: str, max_hops: int) -> str:
    """Edges used by any permitted seed-to-chunk path in one layer."""
    return (
        "UNWIND $seeds AS seed_row "
        f"MATCH (seed:{namespace} {{id: seed_row.node_id}}) "
        f"MATCH path = (seed)-[rels:{RELATIONSHIP_TYPE}*1..{max_hops}]-(target:{namespace}) "
        "WHERE target.kind = 'chunk' "
        "  AND target.id <> seed.id "
        "  AND all(r IN rels WHERE r.layer = $layer "
        "      AND r.source_chunk_id IN $visible_ids) "
        "UNWIND rels AS r "
        "RETURN DISTINCT startNode(r).id AS source, endNode(r).id AS target, "
        "  r.relation AS relation, r.layer AS layer, "
        "  r.source_chunk_id AS source_chunk_id"
    )


def traverse_from_seeds(
    store: GraphStore,
    seeds: Sequence[Hit],
    role: str,
    chunks: Sequence[Chunk],
    layer: str,
    max_hops: int = DEFAULT_MAX_HOPS,
    collect_walked: bool = False,
) -> SeedTraversalResult:
    """Walk one graph layer from role-visible dense seeds."""
    hops = _check_hops(max_hops)
    allowed = visible_ids(chunks, role)

    safe_seeds: list[dict[str, object]] = []
    seed_ids: set[str] = set()
    for seed in seeds:
        if seed.chunk_id not in allowed or seed.chunk_id in seed_ids:
            continue
        safe_seeds.append(
            {
                "node_id": chunk_node_id(seed.chunk_id),
                "chunk_id": seed.chunk_id,
                "rank": int(seed.rank),
            }
        )
        seed_ids.add(seed.chunk_id)
    if not safe_seeds:
        return SeedTraversalResult()

    params = {
        "seeds": safe_seeds,
        "layer": layer,
        "visible_ids": sorted(allowed),
    }
    paths: list[SeedPath] = []
    n_blocked_late = 0
    for row in store.read(_path_query(store.namespace, hops), params):
        target_id = chunk_id_from_node_id(str(row["node_id"]))
        seed_id = str(row["seed_chunk_id"])
        if not target_id or seed_id not in seed_ids:
            continue
        if target_id not in allowed:
            n_blocked_late += 1
            continue
        paths.append(
            SeedPath(
                seed_chunk_id=seed_id,
                seed_rank=int(row["seed_rank"]),
                chunk_id=target_id,
                hops=int(row["hops"]),
            )
        )

    walked: list[WalkedEdge] = []
    if collect_walked:
        for row in store.read(_walked_query(store.namespace, hops), params):
            walked.append(
                WalkedEdge(
                    source=str(row["source"]),
                    target=str(row["target"]),
                    relation=str(row["relation"]),
                    layer=str(row["layer"]),
                    source_chunk_id=str(row["source_chunk_id"]),
                )
            )

    return SeedTraversalResult(
        paths=paths,
        walked=walked,
        n_blocked_late=n_blocked_late,
    )


class SeededGraphRetriever:
    """Dense seeds plus expansion through one named graph layer."""

    def __init__(
        self,
        name: str,
        layer: str,
        chunks: Sequence[Chunk],
        dense: Retriever,
        store: GraphStore,
        seed_k: int = DEFAULT_SEED_K,
        max_hops: int = DEFAULT_MAX_HOPS,
    ) -> None:
        if seed_k <= 0:
            raise ValueError("seed_k must be positive")
        self.name = name
        self.layer = layer
        self.chunks = list(chunks)
        self.dense = dense
        self.store = store
        self.seed_k = seed_k
        self.max_hops = _check_hops(max_hops)
        self._by_id = {chunk.chunk_id: chunk for chunk in self.chunks}

        if store.layer_counts().get(layer, 0) == 0:
            hint = " Run `finrag graph --llm`." if layer == "entity" else " Run `finrag graph`."
            raise GraphLayerUnavailableError(
                f"{name} needs the Neo4j '{layer}' edge layer, but it is missing."
                f"{hint}"
            )

    def search(self, query: str, role: str, k: int = 10) -> list[Hit]:
        if k <= 0 or not query.strip():
            return []

        allowed = visible_ids(self.chunks, role)
        seeds = self.dense.search(query, role, k=self.seed_k)
        if not seeds:
            return []

        traversal = traverse_from_seeds(
            self.store,
            seeds,
            role,
            self.chunks,
            self.layer,
            max_hops=self.max_hops,
        )

        scores: dict[str, float] = {}
        direct_scores: dict[str, float] = {}
        path_details: dict[str, list[dict[str, object]]] = {}
        seed_ranks = {seed.chunk_id: seed.rank for seed in seeds if seed.chunk_id in allowed}

        for seed_id, rank in seed_ranks.items():
            direct = 1.0 / rank
            scores[seed_id] = direct
            direct_scores[seed_id] = direct

        for path in traversal.paths:
            scores[path.chunk_id] = scores.get(path.chunk_id, 0.0) + path.contribution
            path_details.setdefault(path.chunk_id, []).append(
                {
                    "seed_chunk_id": path.seed_chunk_id,
                    "seed_rank": path.seed_rank,
                    "hops": path.hops,
                    "contribution": path.contribution,
                }
            )

        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        hits: list[Hit] = []
        for chunk_id, score in ordered:
            if len(hits) >= k:
                break
            chunk = self._by_id.get(chunk_id)
            if chunk is None or chunk_id not in allowed:
                continue
            details = sorted(
                path_details.get(chunk_id, []),
                key=lambda item: (int(item["seed_rank"]), str(item["seed_chunk_id"])),
            )
            hits.append(
                Hit(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    section_slug=chunk.section_slug,
                    rank=len(hits) + 1,
                    score=score,
                    retriever=self.name,
                    owner_dept=chunk.owner_dept,
                    doc_type=chunk.doc_type,
                    sensitivity=chunk.sensitivity,
                    allowed_roles=chunk.allowed_roles,
                    text=chunk.text,
                    debug={
                        "graph_layer": self.layer,
                        "seed_k": self.seed_k,
                        "max_hops": self.max_hops,
                        "seed_rank": seed_ranks.get(chunk_id),
                        "direct_seed_score": direct_scores.get(chunk_id, 0.0),
                        "graph_contribution": sum(
                            float(item["contribution"]) for item in details
                        ),
                        "paths": details,
                        "n_blocked_late": traversal.n_blocked_late,
                    },
                )
            )
        return hits

    def configuration(self) -> dict[str, object]:
        """Stable settings copied into evaluation reports."""
        return {
            "name": self.name,
            "graph_layer": self.layer,
            "seed_k": self.seed_k,
            "max_hops": self.max_hops,
            "embedder": getattr(getattr(self.dense, "embedder", None), "name", None),
            "embedding_model": getattr(getattr(self.dense, "embedder", None), "model", None),
            "graph_layer_counts": self.store.layer_counts(),
            "score": "direct=1/rank; graph=1/(rank*(hops+1)); shortest path per seed",
        }
