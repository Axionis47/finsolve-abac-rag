"""GraphStore: the only file that talks to Neo4j.

Sid's ruling for R5 is a real graph database, not an in-memory graph library, so
the traversal and the role mask are one Cypher query the server runs, not a
Python loop we could accidentally write in the wrong order.

TWO IDEAS RUN THROUGH THIS FILE.

Namespace label. Every node this app writes carries one extra label, ":Finrag"
by default, and a rebuild wipes exactly that label. The tests pass
namespace="FinragTest" and delete only that, so a test run can never wipe the
real graph, and a real rebuild can never wipe a test's fixture.

One relationship type. Cypher cannot take a relationship type as a parameter, so
a graph with a type per relation would need string interpolation on data the LLM
wrote. Instead every relationship is :REL and carries `relation` as a property.
One static query shape, nothing from the model reaching the query text, and the
traversal stays a single readable statement.
"""

from __future__ import annotations

import os
import re
import socket
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Self
from urllib.parse import urlparse

from neo4j import Driver, GraphDatabase
from neo4j.exceptions import DriverError, Neo4jError

from finrag.graph.schema import Edge, Node, missing_endpoints

# Defaults are the local dev container's credentials on purpose. This password
# opens nothing but a throwaway Neo4j on this laptop (docker container
# finrag-neo4j, no data anyone else can reach), and having defaults means the
# tests and `finrag graph` work with no setup. Any real deployment sets
# NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD and never touches these.
DEFAULT_URI = "bolt://localhost:7687"
DEFAULT_USER = "neo4j"
DEFAULT_PASSWORD = "finragdev"
DEFAULT_DATABASE = "neo4j"

DEFAULT_NAMESPACE = "Finrag"
TEST_NAMESPACE = "FinragTest"

# The one relationship type; see the module docstring.
RELATIONSHIP_TYPE = "REL"

# Batch size for the UNWIND writes. Small enough that one failed batch is cheap
# to reason about, large enough that the whole corpus is a handful of round trips.
DEFAULT_BATCH_SIZE = 500

# A namespace becomes part of the query text, so it is checked hard: letters and
# digits, starting with a letter. Nothing else is ever interpolated into Cypher.
_SAFE_LABEL = re.compile(r"^[A-Za-z][A-Za-z0-9]*$")


@dataclass(frozen=True)
class GraphCounts:
    """How much is in the graph, or how much a write put there."""

    n_nodes: int
    n_edges: int


def check_namespace(namespace: str) -> str:
    """Allow a namespace only if it is safe to paste into a query."""
    if not _SAFE_LABEL.match(namespace):
        raise ValueError(
            f"namespace '{namespace}' must be letters and digits starting with a "
            "letter; it is pasted into Cypher as a label, so nothing else is allowed."
        )
    return namespace


def bolt_is_open(uri: str | None = None, timeout: float = 1.0) -> bool:
    """True if something is listening on the bolt port.

    A plain socket check, not a driver handshake, because the tests call this
    once to decide whether to skip and should not pay for a connection or wait
    on a retry policy.
    """
    target = uri or os.environ.get("NEO4J_URI", DEFAULT_URI)
    parsed = urlparse(target)
    host = parsed.hostname or "localhost"
    port = parsed.port or 7687
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class GraphStore:
    """Reads and writes one namespaced graph in Neo4j."""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        namespace: str = DEFAULT_NAMESPACE,
        database: str = DEFAULT_DATABASE,
    ) -> None:
        self.uri = uri or os.environ.get("NEO4J_URI", DEFAULT_URI)
        self.namespace = check_namespace(namespace)
        self.database = database
        auth_user = user or os.environ.get("NEO4J_USER", DEFAULT_USER)
        auth_password = password or os.environ.get("NEO4J_PASSWORD", DEFAULT_PASSWORD)
        self._driver: Driver = GraphDatabase.driver(self.uri, auth=(auth_user, auth_password))

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def ping(self) -> bool:
        """True if the server answers and the credentials work."""
        try:
            self._driver.verify_connectivity()
            return True
        except (DriverError, Neo4jError, OSError):
            # A closed port, a refused handshake or bad credentials all mean the
            # same thing to the caller: not usable. It only wants a yes or no.
            return False

    def read(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Run a read query and return its rows as plain dicts."""
        result = self._driver.execute_query(
            query,
            parameters_=params or {},
            database_=self.database,
        )
        rows: list[dict[str, Any]] = []
        for record in result.records:
            rows.append(record.data())
        return rows

    def write(self, query: str, params: dict[str, Any] | None = None) -> None:
        """Run a write query and drop the summary; callers ask `counts()` instead."""
        self._driver.execute_query(
            query,
            parameters_=params or {},
            database_=self.database,
        )

    def ensure_indexes(self) -> None:
        """Index the node id, and the folded label the seed match reads.

        Without the id index every MERGE in the write scans the namespace, which
        turns a rebuild from seconds into minutes.
        """
        name = self.namespace.lower()
        self.write(
            f"CREATE INDEX {name}_node_id IF NOT EXISTS "
            f"FOR (n:{self.namespace}) ON (n.id)"
        )
        self.write(
            f"CREATE INDEX {name}_node_label_lower IF NOT EXISTS "
            f"FOR (n:{self.namespace}) ON (n.label_lower)"
        )

    def wipe(self) -> None:
        """Delete every node in this namespace, and their relationships with them.

        A rebuild is a wipe and a fresh write, never a patch: the alternative is
        stale edges from a document that has since changed, which would keep
        answering queries with text that is no longer in the corpus.
        """
        self.write(f"MATCH (n:{self.namespace}) DETACH DELETE n")

    def write_nodes(self, nodes: Sequence[Node], batch_size: int = DEFAULT_BATCH_SIZE) -> int:
        """Merge nodes in batches; returns how many were sent."""
        query = (
            f"UNWIND $rows AS row "
            f"MERGE (n:{self.namespace} {{id: row.id}}) "
            f"SET n.kind = row.kind, n.label = row.label, n.label_lower = row.label_lower"
        )
        rows: list[dict[str, str]] = []
        for node in nodes:
            rows.append(
                {
                    "id": node.id,
                    "kind": node.kind,
                    "label": node.label,
                    # Folded once at write time so the seed match can be a plain
                    # CONTAINS and still be case-insensitive.
                    "label_lower": node.label.lower(),
                }
            )
        for batch in _batches(rows, batch_size):
            self.write(query, {"rows": batch})
        return len(rows)

    def write_edges(self, edges: Sequence[Edge], batch_size: int = DEFAULT_BATCH_SIZE) -> int:
        """Merge relationships in batches; returns how many were sent.

        Both endpoints must already be in this namespace. `source_chunk_id` is
        part of the MERGE key, not a later SET, so the same relation extracted
        from two different chunks stays two relationships: each one is walkable
        by whoever may see its own chunk.
        """
        query = (
            f"UNWIND $rows AS row "
            f"MATCH (a:{self.namespace} {{id: row.source}}) "
            f"MATCH (b:{self.namespace} {{id: row.target}}) "
            f"MERGE (a)-[r:{RELATIONSHIP_TYPE} "
            f"{{relation: row.relation, layer: row.layer, "
            f"source_chunk_id: row.source_chunk_id}}]->(b)"
        )
        rows: list[dict[str, str]] = []
        for edge in edges:
            rows.append(
                {
                    "source": edge.source,
                    "target": edge.target,
                    "relation": edge.relation,
                    "layer": edge.layer,
                    "source_chunk_id": edge.source_chunk_id,
                }
            )
        for batch in _batches(rows, batch_size):
            self.write(query, {"rows": batch})
        return len(rows)

    def write_graph(
        self,
        nodes: Sequence[Node],
        edges: Sequence[Edge],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> GraphCounts:
        """Rebuild this namespace from scratch: wipe, then write nodes and edges."""
        unknown = missing_endpoints(nodes, edges)
        if unknown:
            raise ValueError(
                f"{len(unknown)} node ids are named by an edge but not defined by a "
                f"node, so those edges would be dropped silently. First few: "
                f"{unknown[:3]}"
            )

        self.wipe()
        self.ensure_indexes()
        self.write_nodes(nodes, batch_size)
        self.write_edges(edges, batch_size)
        return self.counts()

    def counts(self) -> GraphCounts:
        """How many nodes and relationships this namespace holds now."""
        node_rows = self.read(f"MATCH (n:{self.namespace}) RETURN count(n) AS n")
        edge_rows = self.read(
            f"MATCH (:{self.namespace})-[r:{RELATIONSHIP_TYPE}]->(:{self.namespace}) "
            f"RETURN count(r) AS n"
        )
        return GraphCounts(n_nodes=int(node_rows[0]["n"]), n_edges=int(edge_rows[0]["n"]))

    def layer_counts(self) -> dict[str, int]:
        """Relationship counts by graph layer, omitting empty layers."""
        rows = self.read(
            f"MATCH (:{self.namespace})-[r:{RELATIONSHIP_TYPE}]->(:{self.namespace}) "
            "RETURN r.layer AS layer, count(r) AS n ORDER BY layer"
        )
        counts: dict[str, int] = {}
        for row in rows:
            layer = row.get("layer")
            if layer is not None:
                counts[str(layer)] = int(row["n"])
        return counts


def _batches(rows: Sequence[dict[str, str]], size: int) -> list[list[dict[str, str]]]:
    """Split rows into lists of at most `size`."""
    if size <= 0:
        raise ValueError("batch size must be positive")
    out: list[list[dict[str, str]]] = []
    start = 0
    while start < len(rows):
        out.append(list(rows[start : start + size]))
        start += size
    return out
