"""GraphRetriever: the same search contract as BM25, backed by the masked walk."""

from __future__ import annotations

from finrag.graph.build_structural import extract
from finrag.graph.retriever import GraphRetriever
from finrag.graph.store import GraphStore
from finrag.models import Chunk
from finrag.retrieval.base import Retriever


def _retriever(store: GraphStore, chunks: list[Chunk]) -> GraphRetriever:
    nodes, edges = extract(chunks)
    store.write_graph(nodes, edges)
    return GraphRetriever(chunks, store)


def test_the_retriever_satisfies_the_retriever_protocol(
    graph_store: GraphStore, micro_corpus: list[Chunk]
) -> None:
    retriever = _retriever(graph_store, micro_corpus)
    assert isinstance(retriever, Retriever)
    assert retriever.name == "graphrag"


def test_a_search_returns_hits_ranked_from_one(
    graph_store: GraphStore, micro_corpus: list[Chunk]
) -> None:
    retriever = _retriever(graph_store, micro_corpus)
    hits = retriever.search("campaign spend", "marketing", k=10)
    assert [hit.rank for hit in hits] == list(range(1, len(hits) + 1))


def test_a_hit_carries_the_chunk_attributes_the_metrics_need(
    graph_store: GraphStore, micro_corpus: list[Chunk]
) -> None:
    retriever = _retriever(graph_store, micro_corpus)
    hit = retriever.search("campaign spend", "marketing", k=1)[0]
    chunk = next(c for c in micro_corpus if c.chunk_id == hit.chunk_id)
    assert hit.doc_id == chunk.doc_id
    assert hit.section_slug == chunk.section_slug
    assert hit.owner_dept == chunk.owner_dept
    assert hit.sensitivity == chunk.sensitivity
    assert hit.allowed_roles == chunk.allowed_roles
    assert hit.text == chunk.text


def test_the_score_is_the_number_of_paths_that_reached_the_chunk(
    graph_store: GraphStore, micro_corpus: list[Chunk]
) -> None:
    retriever = _retriever(graph_store, micro_corpus)
    hit = retriever.search("campaign spend", "marketing", k=1)[0]
    assert hit.score == float(hit.debug["n_paths"])
    assert hit.debug["n_paths"] >= 1


def test_the_debug_field_records_the_seeds_and_the_hop_limit(
    graph_store: GraphStore, micro_corpus: list[Chunk]
) -> None:
    retriever = _retriever(graph_store, micro_corpus)
    hit = retriever.search("campaign spend", "marketing", k=1)[0]
    assert hit.debug["seed_terms"] == ["campaign", "spend"]
    assert hit.debug["max_hops"] == 2
    assert hit.debug["n_blocked_late"] == 0


def test_a_role_never_sees_a_chunk_it_may_not_see(
    graph_store: GraphStore, micro_corpus: list[Chunk]
) -> None:
    retriever = _retriever(graph_store, micro_corpus)
    for role in ("c_level", "marketing", "hr", "finance", "employee"):
        for hit in retriever.search("Q1 2024 revenue campaign spend salary risk", role, k=10):
            assert role in hit.allowed_roles


def test_two_roles_asking_the_same_question_get_different_answers(
    graph_store: GraphStore, micro_corpus: list[Chunk]
) -> None:
    retriever = _retriever(graph_store, micro_corpus)
    query = "Q4 revenue and campaign spend"
    c_level = {hit.chunk_id for hit in retriever.search(query, "c_level", k=10)}
    marketing = {hit.chunk_id for hit in retriever.search(query, "marketing", k=10)}
    assert marketing < c_level


def test_k_cuts_the_result_list(graph_store: GraphStore, micro_corpus: list[Chunk]) -> None:
    retriever = _retriever(graph_store, micro_corpus)
    hits = retriever.search("Q1 2024 revenue campaign spend salary risk", "c_level", k=2)
    assert len(hits) == 2


def test_a_query_with_no_usable_words_returns_nothing(
    graph_store: GraphStore, micro_corpus: list[Chunk]
) -> None:
    retriever = _retriever(graph_store, micro_corpus)
    assert retriever.search("the and of", "c_level", k=10) == []


def test_a_chunk_the_graph_knows_but_the_corpus_dropped_is_skipped(
    graph_store: GraphStore, micro_corpus: list[Chunk]
) -> None:
    # A stale graph must not serve text we can no longer check the policy on.
    nodes, edges = extract(micro_corpus)
    graph_store.write_graph(nodes, edges)
    shrunk = [chunk for chunk in micro_corpus if chunk.owner_dept != "marketing"]
    retriever = GraphRetriever(shrunk, graph_store)

    hits = retriever.search("campaign spend", "marketing", k=10)
    assert hits == []
