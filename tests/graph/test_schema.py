"""The graph shapes: id conventions, validation, and the chunk-id rule on edges."""

from __future__ import annotations

import pytest

from finrag.graph.schema import (
    Edge,
    Node,
    chunk_edges_are_self_labelled,
    chunk_id_from_node_id,
    dedupe_edges,
    dedupe_nodes,
    make_chunk_node,
    make_dept_node,
    make_doc_node,
    make_entity_node,
    make_quarter_node,
    missing_endpoints,
    normalize_entity,
    normalize_relation,
)
from finrag.models import Chunk


def test_an_edge_without_a_source_chunk_id_is_refused() -> None:
    with pytest.raises(ValueError, match="source_chunk_id"):
        Edge(source="entity:revenue", target="entity:q4", relation="reported", source_chunk_id="")


def test_an_edge_keeps_the_chunk_that_produced_it() -> None:
    edge = Edge(
        source="entity:revenue",
        target="entity:north america",
        relation="reported_in",
        source_chunk_id="finance/report#q4-revenue",
    )
    assert edge.source_chunk_id == "finance/report#q4-revenue"


def test_a_node_of_an_unknown_kind_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown node kind"):
        Node(id="thing:1", kind="widget", label="a widget")


def test_a_node_needs_a_label_to_be_seeded_on() -> None:
    with pytest.raises(ValueError, match="non-empty label"):
        Node(id="entity:x", kind="entity", label="  ")


def test_a_chunk_node_is_labelled_with_its_section_path(micro_corpus: list[Chunk]) -> None:
    chunk = micro_corpus[0]
    node = make_chunk_node(chunk)
    assert node.id == "chunk:" + chunk.chunk_id
    assert node.kind == "chunk"
    assert node.label == chunk.section_path


def test_ids_are_prefixed_so_two_kinds_cannot_collide() -> None:
    dept = make_dept_node("finance")
    entity = make_entity_node("finance")
    doc = make_doc_node("finance")
    quarter = make_quarter_node("Q4-2024")
    assert len({dept.id, entity.id, doc.id, quarter.id}) == 4


def test_a_chunk_id_can_be_read_back_out_of_a_chunk_node_id() -> None:
    assert chunk_id_from_node_id("chunk:hr/hr_data#row-1") == "hr/hr_data#row-1"


def test_a_non_chunk_node_id_yields_no_chunk_id() -> None:
    assert chunk_id_from_node_id("dept:finance") == ""


def test_entity_names_are_folded_so_two_spellings_are_one_node() -> None:
    assert make_entity_node("Gross  Margin").id == make_entity_node("gross margin").id


def test_normalize_entity_collapses_whitespace_and_case() -> None:
    assert normalize_entity("  Cash   FLOW ") == "cash flow"


def test_normalize_relation_gives_snake_case_words() -> None:
    assert normalize_relation("Reported Revenue For") == "reported_revenue_for"


def test_dedupe_nodes_keeps_the_first_of_each_id() -> None:
    first = Node(id="dept:hr", kind="dept", label="hr")
    second = Node(id="dept:hr", kind="dept", label="Human Resources")
    assert dedupe_nodes([first, second, first]) == [first]


def test_dedupe_edges_keeps_edges_that_differ_only_by_source_chunk() -> None:
    one = Edge(source="entity:a", target="entity:b", relation="r", source_chunk_id="c1")
    two = Edge(source="entity:a", target="entity:b", relation="r", source_chunk_id="c2")
    assert dedupe_edges([one, two, one]) == [one, two]


def test_missing_endpoints_names_the_nodes_an_edge_invented() -> None:
    nodes = [Node(id="entity:a", kind="entity", label="a")]
    edges = [Edge(source="entity:a", target="entity:ghost", relation="r", source_chunk_id="c1")]
    assert missing_endpoints(nodes, edges) == ["entity:ghost"]


def test_an_edge_touching_a_chunk_must_carry_that_chunk_s_id() -> None:
    wrong = Edge(
        source="chunk:finance/report#q4",
        target="dept:finance",
        relation="owned_by",
        source_chunk_id="marketing/report#spend",
    )
    assert chunk_edges_are_self_labelled([wrong]) == [wrong]


def test_an_edge_between_two_entities_is_free_of_that_rule() -> None:
    fine = Edge(
        source="entity:revenue",
        target="entity:q4",
        relation="reported_in",
        source_chunk_id="finance/report#q4",
    )
    assert chunk_edges_are_self_labelled([fine]) == []
