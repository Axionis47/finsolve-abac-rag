"""The masked walk, and the leakage test for graphs.

The important test in this file is `test_what_the_walk_touches_is_exactly_what_the_role_may_walk`:
for every role, the relationships the Cypher actually walked must equal the
relationships of the subgraph induced by "source_chunk_id is visible to this
role", and the ranking must match a walk of that subgraph computed here in
Python. If the mask were applied after the walk instead of during it, the walked
set would be larger than the induced set and this test would fail.

Two leaks are planted as well: one correctly labelled edge into a chunk the role
cannot see (the Cypher mask must refuse it), and one edge mislabelled with a
visible chunk's id (the Cypher mask cannot know, so the Python double lock must
refuse it). Each planted leak is first shown to be genuinely reachable in two
hops by a control query with no mask at all.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from finrag.graph import traverse as traverse_module
from finrag.graph.build_structural import extract
from finrag.graph.schema import Edge, Node, chunk_id_from_node_id, chunk_node_id
from finrag.graph.store import RELATIONSHIP_TYPE, GraphStore
from finrag.graph.traverse import matches_terms, seed_terms, traverse
from finrag.models import Chunk
from finrag.retrieval.base import visible_ids

# The clause the two "removing the mask" tests take out of the query.
MASK_CLAUSE = "AND all(r IN rels WHERE r.source_chunk_id IN $visible_ids) "

# A query that touches most of the micro corpus: quarters, metrics, departments.
WIDE_QUERY = "Q1 2024 campaign spend revenue risk salary finance"

# Roles the micro corpus knows about, including two that may see nothing.
ROLES = ("c_level", "marketing", "hr", "finance", "employee")

FINANCE_REVENUE_CHUNK = "finance/quarterly_financial_report#q4-revenue"


def reference_walk(
    nodes: Sequence[Node],
    edges: Sequence[Edge],
    query: str,
    visible: frozenset[str],
    max_hops: int = 2,
) -> tuple[dict[str, int], set[tuple[str, str, str, str]]]:
    """What the traversal should do, worked out in Python from the same lists.

    Deliberately written from the definition rather than from the Cypher: seeds
    are nodes whose label contains a query term, only edges whose source chunk is
    visible may be walked, no walk uses the same edge twice, and a chunk reached
    at one or two hops is one path. Returns the path count per chunk_id and every
    edge any counted path used.
    """
    kinds = {node.id: node.kind for node in nodes}
    labels = {node.id: node.label for node in nodes}
    terms = seed_terms(query)

    neighbours: dict[str, list[Edge]] = {}
    for edge in edges:
        if edge.source_chunk_id not in visible:
            continue
        neighbours.setdefault(edge.source, []).append(edge)
        neighbours.setdefault(edge.target, []).append(edge)

    counts: dict[str, int] = {}
    walked: set[tuple[str, str, str, str]] = set()

    for seed_id in sorted(kinds):
        if not matches_terms(labels[seed_id], terms):
            continue
        stack: list[tuple[str, tuple[tuple[str, str, str, str], ...]]] = [(seed_id, ())]
        while stack:
            current, used = stack.pop()
            if used and kinds[current] == "chunk":
                chunk_id = chunk_id_from_node_id(current)
                counts[chunk_id] = counts.get(chunk_id, 0) + 1
                for key in used:
                    walked.add(key)
            if len(used) == max_hops:
                continue
            for edge in neighbours.get(current, []):
                if edge.key() in used:
                    continue
                other = edge.target if edge.source == current else edge.source
                stack.append((other, used + (edge.key(),)))

    return counts, walked


def unmasked_chunk_ids(store: GraphStore, query: str, max_hops: int = 2) -> set[str]:
    """Chunks reachable with no role mask at all: the control for the leak tests."""
    rows = store.read(
        f"MATCH (seed:{store.namespace}) "
        f"WHERE any(term IN $terms WHERE seed.label_lower CONTAINS term) "
        f"MATCH (seed)-[:{RELATIONSHIP_TYPE}*1..{max_hops}]-(target:{store.namespace}) "
        f"WHERE target.kind = 'chunk' "
        f"RETURN DISTINCT target.id AS node_id",
        {"terms": seed_terms(query)},
    )
    found: set[str] = set()
    for row in rows:
        found.add(chunk_id_from_node_id(str(row["node_id"])))
    return found


def test_seed_terms_drop_stopwords_and_single_letters() -> None:
    assert seed_terms("What was the Q4 revenue in 2024?") == ["q4", "revenue", "2024"]


def test_seed_terms_keep_each_word_once() -> None:
    assert seed_terms("revenue revenue REVENUE") == ["revenue"]


def test_a_label_matches_a_term_whatever_the_case() -> None:
    assert matches_terms("Quarterly Financial Report > Q4 > Revenue", ["revenue"]) is True
    assert matches_terms("Employee Handbook > Dress Code", ["revenue"]) is False


def test_the_walk_finds_the_chunks_a_role_may_see(
    graph_store: GraphStore, micro_corpus: list[Chunk]
) -> None:
    nodes, edges = extract(micro_corpus)
    graph_store.write_graph(nodes, edges)

    result = traverse(graph_store, "campaign spend", "marketing", micro_corpus, k=10)

    found = {candidate.chunk_id for candidate in result.candidates}
    assert found == {
        "marketing/market_report_q1_2024#campaign-spend",
        "marketing/market_report_q3_2024#campaign-spend",
    }


def test_the_walk_returns_nothing_for_a_role_with_no_visible_chunks(
    graph_store: GraphStore, micro_corpus: list[Chunk]
) -> None:
    nodes, edges = extract(micro_corpus)
    graph_store.write_graph(nodes, edges)

    result = traverse(graph_store, WIDE_QUERY, "engineering", micro_corpus, k=10)
    assert result.candidates == []


def test_a_query_of_only_stopwords_seeds_nothing(
    graph_store: GraphStore, micro_corpus: list[Chunk]
) -> None:
    nodes, edges = extract(micro_corpus)
    graph_store.write_graph(nodes, edges)

    result = traverse(graph_store, "what is the", "c_level", micro_corpus, k=10)
    assert result.candidates == []
    assert result.seed_terms == []


def test_the_ranking_puts_the_most_reached_chunk_first(
    graph_store: GraphStore, micro_corpus: list[Chunk]
) -> None:
    nodes, edges = extract(micro_corpus)
    graph_store.write_graph(nodes, edges)

    result = traverse(graph_store, WIDE_QUERY, "c_level", micro_corpus, k=50)

    path_counts = [candidate.n_paths for candidate in result.candidates]
    assert path_counts == sorted(path_counts, reverse=True)


def test_two_runs_of_the_same_walk_agree(
    graph_store: GraphStore, micro_corpus: list[Chunk]
) -> None:
    nodes, edges = extract(micro_corpus)
    graph_store.write_graph(nodes, edges)

    first = traverse(graph_store, WIDE_QUERY, "c_level", micro_corpus, k=50)
    second = traverse(graph_store, WIDE_QUERY, "c_level", micro_corpus, k=50)
    assert first.candidates == second.candidates


def test_k_limits_how_many_chunks_come_back(
    graph_store: GraphStore, micro_corpus: list[Chunk]
) -> None:
    nodes, edges = extract(micro_corpus)
    graph_store.write_graph(nodes, edges)

    result = traverse(graph_store, WIDE_QUERY, "c_level", micro_corpus, k=2)
    assert len(result.candidates) == 2


def test_a_hop_count_outside_the_allowed_range_is_refused(
    graph_store: GraphStore, micro_corpus: list[Chunk]
) -> None:
    with pytest.raises(ValueError, match="max_hops"):
        traverse(graph_store, "revenue", "c_level", micro_corpus, k=5, max_hops=9)


@pytest.mark.parametrize("role", ROLES)
def test_what_the_walk_touches_is_exactly_what_the_role_may_walk(
    graph_store: GraphStore, micro_corpus: list[Chunk], role: str
) -> None:
    # The per-role subgraph equivalence test: the leakage test for graphs.
    nodes, edges = extract(micro_corpus)
    graph_store.write_graph(nodes, edges)
    visible = visible_ids(micro_corpus, role)

    result = traverse(graph_store, WIDE_QUERY, role, micro_corpus, k=50, collect_walked=True)
    expected_counts, expected_walked = reference_walk(nodes, edges, WIDE_QUERY, visible)

    walked = {
        (edge.source, edge.target, edge.relation, edge.source_chunk_id) for edge in result.walked
    }
    counts = {candidate.chunk_id: candidate.n_paths for candidate in result.candidates}

    assert walked == expected_walked
    assert counts == expected_counts


@pytest.mark.parametrize("role", ROLES)
def test_no_walked_relationship_names_a_chunk_outside_the_role_s_view(
    graph_store: GraphStore, micro_corpus: list[Chunk], role: str
) -> None:
    nodes, edges = extract(micro_corpus)
    graph_store.write_graph(nodes, edges)
    visible = visible_ids(micro_corpus, role)

    result = traverse(graph_store, WIDE_QUERY, role, micro_corpus, k=50, collect_walked=True)

    for edge in result.walked:
        assert edge.source_chunk_id in visible
    for candidate in result.candidates:
        assert candidate.chunk_id in visible


def test_a_planted_edge_into_a_chunk_the_role_cannot_see_is_not_walked(
    graph_store: GraphStore, micro_corpus: list[Chunk]
) -> None:
    # The leak: entity "spend" is one hop from a marketing seed, and this edge
    # runs from it into a finance chunk only c_level may see. The edge carries
    # that finance chunk's id, so the mask inside the Cypher is what refuses it.
    nodes, edges = extract(micro_corpus)
    leak = Edge(
        source="entity:spend",
        target=chunk_node_id(FINANCE_REVENUE_CHUNK),
        relation="planted_leak",
        source_chunk_id=FINANCE_REVENUE_CHUNK,
    )
    graph_store.write_graph(nodes, list(edges) + [leak])

    # Control: without a mask the leak really is reachable, so this is a live leak
    # and not a path that never existed.
    assert FINANCE_REVENUE_CHUNK in unmasked_chunk_ids(graph_store, "campaign spend")

    result = traverse(
        graph_store, "campaign spend", "marketing", micro_corpus, k=50, collect_walked=True
    )

    assert FINANCE_REVENUE_CHUNK not in {c.chunk_id for c in result.candidates}
    assert "planted_leak" not in {edge.relation for edge in result.walked}


def test_a_planted_edge_mislabelled_with_a_visible_chunk_is_caught_by_the_double_lock(
    graph_store: GraphStore, micro_corpus: list[Chunk]
) -> None:
    # The nastier leak: the edge touches a finance chunk marketing cannot see, but
    # claims a marketing chunk as its source. The Cypher mask has no way to know,
    # so the Python double lock is the thing that must refuse it.
    nodes, edges = extract(micro_corpus)
    mislabelled = Edge(
        source="entity:spend",
        target=chunk_node_id(FINANCE_REVENUE_CHUNK),
        relation="planted_mislabelled_leak",
        source_chunk_id="marketing/market_report_q1_2024#campaign-spend",
    )
    graph_store.write_graph(nodes, list(edges) + [mislabelled])

    result = traverse(graph_store, "campaign spend", "marketing", micro_corpus, k=50)

    assert FINANCE_REVENUE_CHUNK not in {c.chunk_id for c in result.candidates}
    assert result.n_blocked_late == 1


def test_removing_the_mask_from_the_cypher_lets_the_planted_leak_be_walked(
    graph_store: GraphStore, micro_corpus: list[Chunk], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Proof that the two tests above are not passing for some other reason: with
    # the mask taken out of the query, the same planted edge IS walked, and only
    # the Python double lock keeps the chunk out of the results.
    nodes, edges = extract(micro_corpus)
    leak = Edge(
        source="entity:spend",
        target=chunk_node_id(FINANCE_REVENUE_CHUNK),
        relation="planted_leak",
        source_chunk_id=FINANCE_REVENUE_CHUNK,
    )
    graph_store.write_graph(nodes, list(edges) + [leak])
    _remove_the_mask(monkeypatch)

    result = traverse(
        graph_store, "campaign spend", "marketing", micro_corpus, k=50, collect_walked=True
    )

    assert "planted_leak" in {edge.relation for edge in result.walked}
    assert result.n_blocked_late == 1


def test_removing_the_mask_breaks_the_per_role_equivalence(
    graph_store: GraphStore, micro_corpus: list[Chunk], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Same proof for the equivalence test: unmasked, marketing walks relationships
    # from chunks it cannot see, so the walked set stops matching its subgraph.
    nodes, edges = extract(micro_corpus)
    graph_store.write_graph(nodes, edges)
    visible = visible_ids(micro_corpus, "marketing")
    _remove_the_mask(monkeypatch)

    result = traverse(graph_store, WIDE_QUERY, "marketing", micro_corpus, k=50, collect_walked=True)
    _expected_counts, expected_walked = reference_walk(nodes, edges, WIDE_QUERY, visible)

    walked = {
        (edge.source, edge.target, edge.relation, edge.source_chunk_id) for edge in result.walked
    }
    assert walked != expected_walked


def _remove_the_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    """Break the traversal on purpose: same queries, without the visible-ids filter."""
    original_rank = traverse_module._rank_query
    original_walked = traverse_module._walked_query
    monkeypatch.setattr(
        traverse_module,
        "_rank_query",
        lambda namespace, hops: original_rank(namespace, hops).replace(MASK_CLAUSE, ""),
    )
    monkeypatch.setattr(
        traverse_module,
        "_walked_query",
        lambda namespace, hops: original_walked(namespace, hops).replace(MASK_CLAUSE, ""),
    )
