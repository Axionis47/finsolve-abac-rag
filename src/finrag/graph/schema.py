"""The graph shapes: a Node, an Edge, and the id conventions both builds share.

Storage-agnostic on purpose. The structural build and the LLM build produce
plain dataclasses so they can be tested with no server running; `store.py` is
the only file that knows Neo4j exists.

THE RULE THAT SHAPES THIS FILE: every Edge carries the chunk_id of the chunk it
was extracted from. Permissions in this project live on chunks
(`Chunk.allowed_roles`), never on entities, so the only way to police a graph
walk is to ask "may this role see the chunk that produced this edge?". An edge
without that id could not be policed at all, so `Edge` refuses to be built
without one.

Ids are prefixed by kind ("chunk:", "doc:", "dept:", "quarter:", "entity:") so
two kinds can never collide on the same name: the department "finance" and an
entity called "finance" are different nodes.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from finrag.models import Chunk

# What a node can be. "entity" covers anything named that is not part of the
# corpus skeleton: a metric heading found by the structural build, or a subject
# or object of an LLM triple. Both land in the same namespace on purpose, so a
# triple about "revenue" attaches to the node the heading scan already made.
NODE_KINDS: tuple[str, ...] = ("entity", "chunk", "doc", "dept", "quarter")

# One Neo4j namespace holds three graph views. Retrieval selects a view by
# choosing which relationship layers it may walk; nodes can be shared safely.
LAYER_STRUCTURAL = "structural"
LAYER_TEMPORAL = "temporal"
LAYER_ENTITY = "entity"
GRAPH_LAYERS: tuple[str, ...] = (LAYER_STRUCTURAL, LAYER_TEMPORAL, LAYER_ENTITY)

CHUNK_PREFIX = "chunk:"
DOC_PREFIX = "doc:"
DEPT_PREFIX = "dept:"
QUARTER_PREFIX = "quarter:"
ENTITY_PREFIX = "entity:"

# Runs of letters, digits and underscores; everything else becomes a separator.
# Used to fold a relation name into snake_case.
_WORD = re.compile(r"\w+")


@dataclass(frozen=True)
class Node:
    """One thing in the graph: a chunk, a document, a department, a quarter, an entity.

    `id` is the stable key writes merge on. `label` is the human-readable text a
    query is matched against when the traversal picks its seeds, so it should be
    short: a section heading, a department name, a metric name. Never the whole
    chunk body, or every chunk would match every query.
    """

    id: str
    kind: str
    label: str

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("a Node needs a non-empty id")
        if self.kind not in NODE_KINDS:
            known = ", ".join(NODE_KINDS)
            raise ValueError(f"unknown node kind '{self.kind}'. Known kinds: {known}")
        if not self.label.strip():
            raise ValueError(f"node {self.id} needs a non-empty label")


@dataclass(frozen=True)
class Edge:
    """One relationship, plus the chunk that is answerable for it.

    `source_chunk_id` is not decoration and not provenance for its own sake: it
    is the access-control key. A role may walk this edge only if that chunk is
    in the role's visible set. See the module docstring and `traverse.py`.
    """

    source: str
    target: str
    relation: str
    layer: str
    source_chunk_id: str

    def __post_init__(self) -> None:
        if not self.source.strip() or not self.target.strip():
            raise ValueError("an Edge needs a non-empty source and target")
        if not self.relation.strip():
            raise ValueError(f"edge {self.source} -> {self.target} needs a relation")
        if self.layer not in GRAPH_LAYERS:
            known = ", ".join(GRAPH_LAYERS)
            raise ValueError(
                f"edge {self.source} -> {self.target} has unknown layer "
                f"'{self.layer}'. Known layers: {known}"
            )
        if not self.source_chunk_id.strip():
            # Refusing here is the cheapest place to catch it. An edge with no
            # chunk id would either leak (if traversal let it through) or be
            # invisible to every role (if it did not); both are silent bugs.
            raise ValueError(
                f"edge {self.source} -{self.relation}-> {self.target} has no "
                "source_chunk_id. Access control rides on chunks, so every edge "
                "must name the chunk it came from."
            )

    def key(self) -> tuple[str, str, str, str, str]:
        """What makes this edge the same edge as another one."""
        return (self.source, self.target, self.relation, self.layer, self.source_chunk_id)


def chunk_node_id(chunk_id: str) -> str:
    return CHUNK_PREFIX + chunk_id


def chunk_id_from_node_id(node_id: str) -> str:
    """The chunk_id inside a chunk node id, or "" for any other kind of node."""
    if not node_id.startswith(CHUNK_PREFIX):
        return ""
    return node_id[len(CHUNK_PREFIX) :]


def make_chunk_node(chunk: Chunk) -> Node:
    """A node for one chunk, labelled with its section path.

    The section path, not the text: the label is what seed matching reads, and a
    whole chunk body would match nearly any query.
    """
    return Node(id=chunk_node_id(chunk.chunk_id), kind="chunk", label=chunk.section_path)


def make_doc_node(doc_id: str) -> Node:
    return Node(id=DOC_PREFIX + doc_id, kind="doc", label=doc_id)


def make_dept_node(dept: str) -> Node:
    return Node(id=DEPT_PREFIX + dept, kind="dept", label=dept)


def make_quarter_node(quarter_label: str) -> Node:
    return Node(id=QUARTER_PREFIX + quarter_label, kind="quarter", label=quarter_label)


def normalize_entity(name: str) -> str:
    """Fold an entity name for its id: lower case, single spaces, trimmed.

    Two spellings of the same name should be one node, or the graph fills up
    with near-duplicates that no traversal can join.
    """
    return " ".join(name.lower().split())


def make_entity_node(name: str) -> Node:
    """An entity node keyed on the folded name, labelled with the folded name.

    The label is folded too, so seed matching never depends on how the LLM or a
    heading happened to capitalise it.
    """
    folded = normalize_entity(name)
    if not folded:
        raise ValueError("an entity needs a non-empty name")
    return Node(id=ENTITY_PREFIX + folded, kind="entity", label=folded)


def normalize_relation(relation: str) -> str:
    """Fold a relation name to snake_case words, e.g. "Reported Revenue" -> "reported_revenue"."""
    words = _WORD.findall(relation.lower())
    return "_".join(words)


def dedupe_nodes(nodes: Iterable[Node]) -> list[Node]:
    """The distinct nodes by id, keeping the first spelling seen."""
    seen: set[str] = set()
    out: list[Node] = []
    for node in nodes:
        if node.id in seen:
            continue
        seen.add(node.id)
        out.append(node)
    return out


def dedupe_edges(edges: Iterable[Edge]) -> list[Edge]:
    """The distinct edges, keeping the first of each identical edge."""
    seen: set[tuple[str, str, str, str, str]] = set()
    out: list[Edge] = []
    for edge in edges:
        if edge.key() in seen:
            continue
        seen.add(edge.key())
        out.append(edge)
    return out


def missing_endpoints(nodes: Iterable[Node], edges: Iterable[Edge]) -> list[str]:
    """Node ids an edge names but no node defines, sorted.

    A write that silently drops such edges would give a graph that scores worse
    for no visible reason, so callers check this and raise instead.
    """
    known: set[str] = set()
    for node in nodes:
        known.add(node.id)

    unknown: set[str] = set()
    for edge in edges:
        if edge.source not in known:
            unknown.add(edge.source)
        if edge.target not in known:
            unknown.add(edge.target)
    return sorted(unknown)


def chunk_edges_are_self_labelled(edges: Iterable[Edge]) -> list[Edge]:
    """Edges touching a chunk node that carry a *different* chunk's id.

    Why this matters: a role reaches a chunk node only by walking an edge that
    touches it, and traversal only walks edges whose source_chunk_id the role may
    see. So as long as every edge touching chunk X carries X's own id, arriving at
    chunk X proves the role may see chunk X. This function returns the edges that
    break that promise; builds keep the list empty and tests assert it.
    """
    offenders: list[Edge] = []
    for edge in edges:
        for endpoint in (edge.source, edge.target):
            chunk_id = chunk_id_from_node_id(endpoint)
            if chunk_id and chunk_id != edge.source_chunk_id:
                offenders.append(edge)
                break
    return offenders
