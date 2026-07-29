"""Fixtures for the graph tests, including the skip when Neo4j is not running.

R5 is the one part of this suite that may talk to a server: a graph database
traversal that is checked in Python is not the thing we ship, and the role mask
lives inside the Cypher. So these tests use the local dev container when it is up
and skip with one line when it is not. `pytest -q` is green either way.

Everything written here goes under the :FinragTest label, never :Finrag, and the
teardown deletes only that namespace. A test run cannot touch the real graph.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from finrag.graph.store import TEST_NAMESPACE, GraphStore, bolt_is_open

# A second test namespace, used to prove a wipe of one leaves the other alone.
OTHER_TEST_NAMESPACE = "FinragTestOther"


@pytest.fixture(scope="session")
def bolt_open() -> bool:
    """Whether the dev container answers, checked once per session."""
    return bolt_is_open()


@pytest.fixture
def graph_store(bolt_open: bool) -> Iterator[GraphStore]:
    """A store on the :FinragTest namespace, emptied before and after the test."""
    if not bolt_open:
        pytest.skip("Neo4j is not reachable on bolt://localhost:7687; graph store tests skipped")

    store = GraphStore(namespace=TEST_NAMESPACE)
    store.wipe()
    try:
        yield store
    finally:
        store.wipe()
        store.close()


@pytest.fixture
def other_graph_store(bolt_open: bool) -> Iterator[GraphStore]:
    """A store on a second test namespace, for the isolation test."""
    if not bolt_open:
        pytest.skip("Neo4j is not reachable on bolt://localhost:7687; graph store tests skipped")

    store = GraphStore(namespace=OTHER_TEST_NAMESPACE)
    store.wipe()
    try:
        yield store
    finally:
        store.wipe()
        store.close()
