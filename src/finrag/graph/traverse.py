"""The masked graph walk: seed on query terms, expand two hops, rank chunks.

THE INVARIANT, and why it is written in Cypher rather than in Python. The role's
visible chunk ids are computed first, from `retrieval/base.visible_ids`, and then
handed to the server as a parameter. Every relationship in every path is checked
against that list *inside* the query, so a denied chunk's edge is never walked,
never lengthens a path, and never contributes to a rank. Filtering after the walk
would be too late: a denied edge would already have carried the walk to a
neighbour it should not have reached, and the ranking would carry that shape out
even if the denied chunk itself were dropped from the results.

Arriving at a chunk therefore proves the role may see it, because every edge that
touches a chunk node carries that chunk's own id (see
`schema.chunk_edges_are_self_labelled`, asserted by the builds' tests).

Ranking: how many distinct paths reach the chunk, then chunk_id to break ties.
Path count is a crude but honest signal, and it is the graph's own signal rather
than a text score borrowed from another retriever.

Known weakness, for whoever tunes this after the first eval: a department node
joins every chunk it owns, so at two hops any chunk in a seed's department is
reachable and the count rewards big departments. If graphrag scores badly, the
first thing to try is leaving `owned_by` out of the walk, not more hops.

Then the double lock. Back in Python the returned chunk ids are filtered against
the same visible set a second time. If the query were ever edited wrongly, this
still refuses to build a Hit for a chunk the role may not see.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from finrag.graph.schema import chunk_id_from_node_id
from finrag.graph.store import RELATIONSHIP_TYPE, GraphStore
from finrag.models import Chunk
from finrag.retrieval.base import visible_ids
from finrag.retrieval.naive import tokenize

# Two hops is the default because that is the shape of the questions this mode
# exists for: chunk -> entity -> other chunk. Three is allowed for experiments;
# more than that returns most of the graph for any query and ranks it by noise.
DEFAULT_MAX_HOPS = 2
MAX_HOPS_ALLOWED = 3

# Query words shorter than this are dropped, so "of" cannot seed. Two, not three,
# because "q4" is a real seed term in this corpus.
MIN_TERM_LENGTH = 2

# Words that match many labels and mean nothing as a seed.
STOPWORDS: frozenset[str] = frozenset(
    {
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "our",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "which",
        "who",
        "why",
        "with",
    }
)


@dataclass(frozen=True)
class Candidate:
    """One chunk the walk reached, and how many distinct paths reached it."""

    chunk_id: str
    n_paths: int


@dataclass(frozen=True)
class WalkedEdge:
    """One relationship the walk actually used, as the test compares them."""

    source: str
    target: str
    relation: str
    source_chunk_id: str


@dataclass(frozen=True)
class TraversalResult:
    """What one masked walk produced.

    `n_blocked_late` counts chunks the Cypher mask returned but the Python double
    lock refused. It should always be zero; anything else means the query and the
    mask have drifted apart, so it is carried into `Hit.debug` rather than hidden.
    """

    candidates: list[Candidate] = field(default_factory=list)
    walked: list[WalkedEdge] = field(default_factory=list)
    seed_terms: list[str] = field(default_factory=list)
    n_blocked_late: int = 0


def seed_terms(query: str) -> list[str]:
    """The query words worth seeding on: folded, distinct, short ones dropped.

    Same tokenizer as naive and BM25 use, deliberately. A different one here
    would fold "InstantPay" differently from the way the labels were written and
    lose the match with no error at all.
    """
    terms: list[str] = []
    for token in tokenize(query):
        if len(token) < MIN_TERM_LENGTH or token in STOPWORDS:
            continue
        if token not in terms:
            terms.append(token)
    return terms


def matches_terms(label: str, terms: Sequence[str]) -> bool:
    """The seed rule, in Python: does this label contain any seed term?

    The Cypher below asks exactly this of `label_lower`. It lives here as a
    function so the equivalence test can seed the same way the server does
    instead of guessing.
    """
    folded = label.lower()
    for term in terms:
        if term in folded:
            return True
    return False


def _check_hops(max_hops: int) -> int:
    """Hop counts are pasted into the query, so only a small integer is allowed.

    Cypher cannot take the bound of a variable-length pattern as a parameter,
    which is the one place this module builds query text from a value. Keeping it
    to 1..3 means nothing else can arrive there.
    """
    if not isinstance(max_hops, int) or isinstance(max_hops, bool):
        raise TypeError(f"max_hops must be an int, got {type(max_hops).__name__}")
    if max_hops < 1 or max_hops > MAX_HOPS_ALLOWED:
        raise ValueError(f"max_hops must be between 1 and {MAX_HOPS_ALLOWED}, got {max_hops}")
    return max_hops


def _rank_query(namespace: str, max_hops: int) -> str:
    """Chunks reached by a fully visible path, ranked by how many paths reach them."""
    return (
        f"MATCH (seed:{namespace}) "
        f"WHERE any(term IN $terms WHERE seed.label_lower CONTAINS term) "
        f"MATCH path = (seed)-[rels:{RELATIONSHIP_TYPE}*1..{max_hops}]-(target:{namespace}) "
        f"WHERE target.kind = 'chunk' "
        f"  AND all(r IN rels WHERE r.source_chunk_id IN $visible_ids) "
        f"RETURN target.id AS node_id, count(DISTINCT path) AS n_paths "
        f"ORDER BY n_paths DESC, node_id ASC "
        f"LIMIT $limit"
    )


def _walked_query(namespace: str, max_hops: int) -> str:
    """Every relationship used by any path the ranking query would count."""
    return (
        f"MATCH (seed:{namespace}) "
        f"WHERE any(term IN $terms WHERE seed.label_lower CONTAINS term) "
        f"MATCH path = (seed)-[rels:{RELATIONSHIP_TYPE}*1..{max_hops}]-(target:{namespace}) "
        f"WHERE target.kind = 'chunk' "
        f"  AND all(r IN rels WHERE r.source_chunk_id IN $visible_ids) "
        f"UNWIND rels AS r "
        f"RETURN DISTINCT startNode(r).id AS source, endNode(r).id AS target, "
        f"  r.relation AS relation, r.source_chunk_id AS source_chunk_id"
    )


def traverse(
    store: GraphStore,
    query: str,
    role: str,
    chunks: Sequence[Chunk],
    k: int = 10,
    max_hops: int = DEFAULT_MAX_HOPS,
    collect_walked: bool = False,
) -> TraversalResult:
    """Walk the graph for `query` as `role`, and return the chunks it reached.

    `collect_walked` runs a second query for the relationships the walk used. The
    retriever does not need them; the per-role subgraph test does, which is how it
    can compare what was walked against what the role was allowed to walk.
    """
    hops = _check_hops(max_hops)

    # The mask comes FIRST. Nothing below runs before this line.
    allowed = visible_ids(chunks, role)

    terms = seed_terms(query)
    if not terms or k <= 0:
        return TraversalResult(seed_terms=terms)

    params = {
        "terms": terms,
        "visible_ids": sorted(allowed),
        "limit": int(k),
    }
    rows = store.read(_rank_query(store.namespace, hops), params)

    candidates: list[Candidate] = []
    n_blocked_late = 0
    for row in rows:
        chunk_id = chunk_id_from_node_id(str(row["node_id"]))
        if not chunk_id:
            continue
        # The double lock: the same door, checked again on this side of the wire.
        if chunk_id not in allowed:
            n_blocked_late += 1
            continue
        candidates.append(Candidate(chunk_id=chunk_id, n_paths=int(row["n_paths"])))

    walked: list[WalkedEdge] = []
    if collect_walked:
        for row in store.read(_walked_query(store.namespace, hops), params):
            walked.append(
                WalkedEdge(
                    source=str(row["source"]),
                    target=str(row["target"]),
                    relation=str(row["relation"]),
                    source_chunk_id=str(row["source_chunk_id"]),
                )
            )

    return TraversalResult(
        candidates=candidates,
        walked=walked,
        seed_terms=terms,
        n_blocked_late=n_blocked_late,
    )
