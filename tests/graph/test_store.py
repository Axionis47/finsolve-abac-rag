"""GraphStore against the live dev container, plus the namespace guard.

These tests write only under :FinragTest and :FinragTestOther and delete only
those, so the real :Finrag graph is never touched. They skip when bolt is closed.
"""

from __future__ import annotations

import pytest

from finrag.graph.build_structural import extract
from finrag.graph.schema import Edge, LAYER_ENTITY, Node
from finrag.graph.store import DEFAULT_NAMESPACE, GraphStore, check_namespace
from finrag.models import Chunk


def test_a_namespace_that_is_not_a_plain_label_is_refused() -> None:
    # It is pasted into Cypher, so anything but letters and digits is refused.
    with pytest.raises(ValueError, match="namespace"):
        check_namespace("Finrag) DETACH DELETE (n")


def test_a_namespace_starting_with_a_digit_is_refused() -> None:
    with pytest.raises(ValueError, match="namespace"):
        check_namespace("1finrag")


def test_the_default_namespace_is_the_real_one() -> None:
    assert check_namespace(DEFAULT_NAMESPACE) == "Finrag"


def test_a_written_graph_is_read_back_with_the_same_counts(
    graph_store: GraphStore, micro_corpus: list[Chunk]
) -> None:
    nodes, edges = extract(micro_corpus)
    counts = graph_store.write_graph(nodes, edges)
    assert counts.n_nodes == len(nodes)
    assert counts.n_edges == len(edges)


def test_every_relationship_keeps_the_chunk_it_came_from(
    graph_store: GraphStore, micro_corpus: list[Chunk]
) -> None:
    nodes, edges = extract(micro_corpus)
    graph_store.write_graph(nodes, edges)
    rows = graph_store.read(
        f"MATCH (:{graph_store.namespace})-[r:REL]->(:{graph_store.namespace}) "
        "RETURN count(r) AS total, count(r.source_chunk_id) AS with_chunk"
    )
    assert rows[0]["total"] == rows[0]["with_chunk"]


def test_relationship_layers_are_stored_and_counted(
    graph_store: GraphStore, micro_corpus: list[Chunk]
) -> None:
    nodes, edges = extract(micro_corpus)
    graph_store.write_graph(nodes, edges)
    assert graph_store.layer_counts() == {
        "structural": sum(edge.layer == "structural" for edge in edges),
        "temporal": sum(edge.layer == "temporal" for edge in edges),
    }


def test_a_node_keeps_a_folded_label_for_the_seed_match(
    graph_store: GraphStore, micro_corpus: list[Chunk]
) -> None:
    nodes, edges = extract(micro_corpus)
    graph_store.write_graph(nodes, edges)
    rows = graph_store.read(
        f"MATCH (n:{graph_store.namespace} {{id: 'dept:finance'}}) "
        "RETURN n.label AS label, n.label_lower AS folded, n.kind AS kind"
    )
    assert rows[0]["folded"] == rows[0]["label"].lower()
    assert rows[0]["kind"] == "dept"


def test_writing_again_replaces_the_graph_instead_of_adding_to_it(
    graph_store: GraphStore, micro_corpus: list[Chunk]
) -> None:
    nodes, edges = extract(micro_corpus)
    graph_store.write_graph(nodes, edges)

    smaller_nodes, smaller_edges = extract(micro_corpus[:2])
    counts = graph_store.write_graph(smaller_nodes, smaller_edges)
    assert counts.n_nodes == len(smaller_nodes)
    assert counts.n_edges == len(smaller_edges)


def test_a_wipe_of_one_namespace_leaves_the_other_alone(
    graph_store: GraphStore,
    other_graph_store: GraphStore,
    micro_corpus: list[Chunk],
) -> None:
    # The real graph is protected by exactly this: a namespace is wiped on its own.
    nodes, edges = extract(micro_corpus)
    graph_store.write_graph(nodes, edges)
    other_graph_store.write_graph(nodes, edges)

    graph_store.wipe()

    assert graph_store.counts().n_nodes == 0
    assert other_graph_store.counts().n_nodes == len(nodes)


def test_an_edge_pointing_at_a_missing_node_stops_the_write(
    graph_store: GraphStore,
) -> None:
    nodes = [Node(id="entity:revenue", kind="entity", label="revenue")]
    edges = [
        Edge(
            source="entity:revenue",
            target="entity:ghost",
            relation="reported_in",
            layer=LAYER_ENTITY,
            source_chunk_id="finance/report#q4",
        )
    ]
    with pytest.raises(ValueError, match="named by an edge"):
        graph_store.write_graph(nodes, edges)


def test_a_small_batch_size_writes_the_same_graph(
    graph_store: GraphStore, micro_corpus: list[Chunk]
) -> None:
    nodes, edges = extract(micro_corpus)
    counts = graph_store.write_graph(nodes, edges, batch_size=2)
    assert counts.n_nodes == len(nodes)
    assert counts.n_edges == len(edges)


def test_the_same_relation_from_two_chunks_stays_two_relationships(
    graph_store: GraphStore,
) -> None:
    nodes = [
        Node(id="entity:revenue", kind="entity", label="revenue"),
        Node(id="entity:q4", kind="entity", label="q4"),
    ]
    edges = [
        Edge(
            source="entity:revenue",
            target="entity:q4",
            relation="reported_in",
            layer=LAYER_ENTITY,
            source_chunk_id="finance/report#q4",
        ),
        Edge(
            source="entity:revenue",
            target="entity:q4",
            relation="reported_in",
            layer=LAYER_ENTITY,
            source_chunk_id="marketing/report#q4",
        ),
    ]
    counts = graph_store.write_graph(nodes, edges)
    # Two chunks said it, and each role may walk only its own chunk's edge.
    assert counts.n_edges == 2


def test_the_store_answers_a_ping_when_the_container_is_up(graph_store: GraphStore) -> None:
    assert graph_store.ping() is True
