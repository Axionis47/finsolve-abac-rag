"""The structural build: what it extracts, and the chunk-id rule it must keep.

Server-free: this build is pure string work on chunks, so none of these tests
need Neo4j.
"""

from __future__ import annotations

from finrag.graph.build_structural import (
    extract,
    mentioned_quarters,
    metrics_in_heading,
)
from finrag.graph.schema import (
    LAYER_STRUCTURAL,
    LAYER_TEMPORAL,
    chunk_edges_are_self_labelled,
    missing_endpoints,
)
from finrag.models import Chunk


def test_a_quarter_is_found_however_the_text_spells_it() -> None:
    assert mentioned_quarters("Revenue in Q4 2024 rose") == ["Q4-2024"]
    assert mentioned_quarters("see the Q4-2024 report") == ["Q4-2024"]
    assert mentioned_quarters("the fourth quarter 2024 close") == ["Q4-2024"]


def test_a_quarter_named_twice_gives_one_label() -> None:
    assert mentioned_quarters("Q1 2024 and again Q1-2024") == ["Q1-2024"]


def test_a_year_we_do_not_track_is_not_a_quarter() -> None:
    assert mentioned_quarters("Q1 2025 guidance") == []


def test_a_metric_heading_is_read_off_the_section_path() -> None:
    assert metrics_in_heading("Quarterly Financial Report > Q4 > Revenue") == ["revenue"]


def test_a_longer_metric_name_wins_over_the_shorter_one_inside_it() -> None:
    assert metrics_in_heading("Financials > Operating Expenses") == ["operating expenses"]


def test_a_heading_with_no_metric_gives_none() -> None:
    assert metrics_in_heading("Employee Handbook > Dress Code") == []


def test_every_chunk_becomes_a_node_and_hangs_off_its_document(
    micro_corpus: list[Chunk],
) -> None:
    nodes, edges = extract(micro_corpus)
    chunk_nodes = [node for node in nodes if node.kind == "chunk"]
    contains = [edge for edge in edges if edge.relation == "contains"]
    assert len(chunk_nodes) == len(micro_corpus)
    assert len(contains) == len(micro_corpus)


def test_a_document_shared_by_two_chunks_is_one_node(micro_corpus: list[Chunk]) -> None:
    nodes, _edges = extract(micro_corpus)
    doc_ids = [node.id for node in nodes if node.kind == "doc"]
    assert doc_ids.count("doc:finance/quarterly_financial_report") == 1


def test_each_chunk_is_owned_by_its_department(micro_corpus: list[Chunk]) -> None:
    _nodes, edges = extract(micro_corpus)
    owned = [edge for edge in edges if edge.relation == "owned_by"]
    assert len(owned) == len(micro_corpus)
    assert {edge.target for edge in owned} == {"dept:finance", "dept:marketing", "dept:hr"}


def test_a_quarter_mentioned_in_the_text_becomes_an_edge(micro_corpus: list[Chunk]) -> None:
    _nodes, edges = extract(micro_corpus)
    mentions = [edge for edge in edges if edge.relation == "mentions"]
    targets = {edge.target for edge in mentions}
    assert "quarter:Q1-2024" in targets
    assert "quarter:Q3-2024" in targets


def test_a_metric_heading_becomes_an_entity_the_llm_build_can_share(
    micro_corpus: list[Chunk],
) -> None:
    nodes, edges = extract(micro_corpus)
    reports = [edge for edge in edges if edge.relation == "reports"]
    entity_ids = {node.id for node in nodes if node.kind == "entity"}
    assert "entity:revenue" in entity_ids
    assert "entity:revenue" in {edge.target for edge in reports}


def test_every_edge_names_the_chunk_it_came_from(micro_corpus: list[Chunk]) -> None:
    _nodes, edges = extract(micro_corpus)
    for edge in edges:
        assert edge.source_chunk_id


def test_structural_and_temporal_edges_are_separate_layers(
    micro_corpus: list[Chunk],
) -> None:
    _nodes, edges = extract(micro_corpus)
    layers_by_relation = {edge.relation: edge.layer for edge in edges}
    assert layers_by_relation["contains"] == LAYER_STRUCTURAL
    assert layers_by_relation["owned_by"] == LAYER_STRUCTURAL
    assert layers_by_relation["mentions"] == LAYER_TEMPORAL
    assert layers_by_relation["reports"] == LAYER_TEMPORAL


def test_no_edge_touching_a_chunk_carries_another_chunk_s_id(
    micro_corpus: list[Chunk],
) -> None:
    # This is what lets the traversal police arrival at a chunk; see traverse.py.
    _nodes, edges = extract(micro_corpus)
    assert chunk_edges_are_self_labelled(edges) == []


def test_no_edge_points_at_a_node_the_build_forgot(micro_corpus: list[Chunk]) -> None:
    nodes, edges = extract(micro_corpus)
    assert missing_endpoints(nodes, edges) == []


def test_extracting_twice_gives_the_same_lists(micro_corpus: list[Chunk]) -> None:
    first_nodes, first_edges = extract(micro_corpus)
    second_nodes, second_edges = extract(micro_corpus)
    assert first_nodes == second_nodes
    assert first_edges == second_edges


def test_an_empty_corpus_gives_an_empty_graph() -> None:
    assert extract([]) == ([], [])
