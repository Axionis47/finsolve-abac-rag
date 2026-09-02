"""The structural graph: what the corpus already tells us, with no model involved.

Every fact here is a fact ingest already knew, so this build is free, repeatable
and cannot hallucinate. It is also the floor the LLM build in `build_llm.py` has
to beat, the same way naive retrieval is the floor for BM25.

What it makes, per chunk:
  doc  -contains->  chunk        the document a chunk came from
  chunk -mentions-> quarter      a quarter named in the chunk text
  chunk -owned_by-> dept         the department folder that owns the chunk
  chunk -reports->  entity       a metric named in the chunk's heading

Metric nodes are `entity` nodes, the same kind the LLM build produces, so a
triple about "revenue" attaches to the node this scan already made.

Every edge carries the chunk_id it came from, and every edge that touches a
chunk node carries *that* chunk's id, which is what makes the role mask in
`traverse.py` able to police the walk. See `schema.chunk_edges_are_self_labelled`.

The matching is deliberately plain string work: lower case, `in`, a hand-written
list of quarter spellings and metric names. No regex cleverness, because the
next person has to be able to predict what this produces.
"""

from __future__ import annotations

from collections.abc import Sequence

from finrag.graph.schema import (
    Edge,
    LAYER_STRUCTURAL,
    LAYER_TEMPORAL,
    Node,
    chunk_node_id,
    dedupe_edges,
    dedupe_nodes,
    make_chunk_node,
    make_dept_node,
    make_doc_node,
    make_entity_node,
    make_quarter_node,
    normalize_entity,
)
from finrag.graph.store import GraphCounts, GraphStore
from finrag.models import Chunk

# The quarters we make nodes for, and the spellings that count as naming one.
# 2024 only: that is the year the corpus reports on, and a node for a quarter
# nothing mentions would just sit there.
QUARTER_SPELLINGS: dict[str, tuple[str, ...]] = {
    "Q1-2024": ("q1 2024", "q1-2024", "q1 fy2024", "first quarter 2024"),
    "Q2-2024": ("q2 2024", "q2-2024", "q2 fy2024", "second quarter 2024"),
    "Q3-2024": ("q3 2024", "q3-2024", "q3 fy2024", "third quarter 2024"),
    "Q4-2024": ("q4 2024", "q4-2024", "q4 fy2024", "fourth quarter 2024"),
}

# Metric names we look for in a heading. A hand list, kept short on purpose: a
# long list drags in words that are not metrics ("plan", "review") and every
# extra name is one more way for a query term to hit an unhelpful seed.
METRIC_HEADINGS: tuple[str, ...] = (
    "revenue",
    "gross margin",
    "operating expenses",
    "net income",
    "cash flow",
    "expenses",
    "budget",
    "spend",
    "customer acquisition cost",
    "conversion",
    "churn",
    "roi",
    "attrition",
    "headcount",
    "salary",
    "risk",
    "vendor",
    "compliance",
)

RELATION_CONTAINS = "contains"
RELATION_MENTIONS = "mentions"
RELATION_OWNED_BY = "owned_by"
RELATION_REPORTS = "reports"


def mentioned_quarters(text: str) -> list[str]:
    """The quarter labels this text names, in QUARTER_SPELLINGS order."""
    folded = text.lower()
    found: list[str] = []
    for label, spellings in QUARTER_SPELLINGS.items():
        for spelling in spellings:
            if spelling in folded:
                found.append(label)
                break
    return found


def metrics_in_heading(section_path: str) -> list[str]:
    """The metric names this heading names, longest match first.

    Longest first, and a shorter name inside one already accepted is skipped, so
    the heading "Operating Expenses" gives the node "operating expenses" and not
    also "expenses". Two nodes for one phrase would split the evidence between
    them and make both weaker seeds.
    """
    folded = section_path.lower()
    found: list[str] = []
    for metric in sorted(METRIC_HEADINGS, key=len, reverse=True):
        if metric not in folded:
            continue
        already_covered = False
        for accepted in found:
            if metric in accepted:
                already_covered = True
        if not already_covered:
            found.append(normalize_entity(metric))
    return found


def extract(chunks: Sequence[Chunk]) -> tuple[list[Node], list[Edge]]:
    """Turn chunks into the structural node and edge lists. No I/O, no model.

    Returns plain lists so this can be tested with nothing running, and so the
    LLM build's lists can simply be added to these before one write.
    """
    nodes: list[Node] = []
    edges: list[Edge] = []

    for chunk in chunks:
        this_chunk = chunk_node_id(chunk.chunk_id)
        nodes.append(make_chunk_node(chunk))

        doc = make_doc_node(chunk.doc_id)
        nodes.append(doc)
        edges.append(
            Edge(
                source=doc.id,
                target=this_chunk,
                relation=RELATION_CONTAINS,
                layer=LAYER_STRUCTURAL,
                source_chunk_id=chunk.chunk_id,
            )
        )

        dept = make_dept_node(chunk.owner_dept)
        nodes.append(dept)
        edges.append(
            Edge(
                source=this_chunk,
                target=dept.id,
                relation=RELATION_OWNED_BY,
                layer=LAYER_STRUCTURAL,
                source_chunk_id=chunk.chunk_id,
            )
        )

        for label in mentioned_quarters(chunk.text):
            quarter = make_quarter_node(label)
            nodes.append(quarter)
            edges.append(
                Edge(
                    source=this_chunk,
                    target=quarter.id,
                    relation=RELATION_MENTIONS,
                    layer=LAYER_TEMPORAL,
                    source_chunk_id=chunk.chunk_id,
                )
            )

        for metric in metrics_in_heading(chunk.section_path):
            entity = make_entity_node(metric)
            nodes.append(entity)
            edges.append(
                Edge(
                    source=this_chunk,
                    target=entity.id,
                    relation=RELATION_REPORTS,
                    layer=LAYER_TEMPORAL,
                    source_chunk_id=chunk.chunk_id,
                )
            )

    return dedupe_nodes(nodes), dedupe_edges(edges)


def push(store: GraphStore, nodes: Sequence[Node], edges: Sequence[Edge]) -> GraphCounts:
    """Write these lists into the store's namespace, replacing what was there.

    Separate from `extract` so the extraction can be tested with no server, and
    so a caller can extract twice (structural plus LLM) and write once.
    """
    return store.write_graph(nodes, edges)


def build_and_push(chunks: Sequence[Chunk], store: GraphStore) -> GraphCounts:
    """Extract the structural graph for these chunks and write it. The whole build."""
    nodes, edges = extract(chunks)
    return push(store, nodes, edges)
