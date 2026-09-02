"""The LLM triple build: parsing, the fallback count, and the graph it makes.

Offline. Every answer here comes from FakeLLM, keyed on the exact prompt, so no
credentials and no network are involved.
"""

from __future__ import annotations

import json
from dataclasses import replace

from finrag.graph.build_llm import (
    LLMBuildStats,
    Triple,
    build_prompt,
    extract,
    parse_triples,
    triples_to_graph,
)
from finrag.graph.schema import LAYER_ENTITY, chunk_edges_are_self_labelled, missing_endpoints
from finrag.llm.fake import FakeLLM
from finrag.models import Chunk


def _answer(*triples: tuple[str, str, str]) -> str:
    """A well-formed model answer for the given triples."""
    items = []
    for subject, relation, object_ in triples:
        items.append({"subject": subject, "relation": relation, "object": object_})
    return json.dumps(items)


def test_the_prompt_asks_for_json_and_carries_the_passage(micro_corpus: list[Chunk]) -> None:
    chunk = micro_corpus[0]
    prompt = build_prompt(chunk)
    assert "JSON array" in prompt
    assert chunk.text in prompt
    assert chunk.section_path in prompt


def test_a_clean_answer_parses_into_triples() -> None:
    parsed = parse_triples(_answer(("Q4 revenue", "reached", "2.6 billion dollars")))
    assert parsed.triples == [
        Triple(subject="Q4 revenue", relation="reached", object="2.6 billion dollars")
    ]


def test_a_relation_is_folded_to_snake_case() -> None:
    parsed = parse_triples(_answer(("revenue", "Reported In", "Q4 2024")))
    assert parsed.triples is not None
    assert parsed.triples[0].relation == "reported_in"


def test_json_wrapped_in_prose_and_a_code_fence_still_parses() -> None:
    raw = "Sure! Here you go:\n```json\n" + _answer(("a", "rel", "b")) + "\n```\nHope that helps."
    parsed = parse_triples(raw)
    assert parsed.triples is not None
    assert len(parsed.triples) == 1


def test_a_number_in_a_field_is_read_as_its_text() -> None:
    raw = json.dumps([{"subject": "headcount", "relation": "is", "object": 1200}])
    parsed = parse_triples(raw)
    assert parsed.triples is not None
    assert parsed.triples[0].object == "1200"


def test_an_answer_with_no_array_is_unusable() -> None:
    assert parse_triples("I cannot help with that request.").usable is False


def test_a_broken_array_is_unusable() -> None:
    assert parse_triples('[{"subject": "a", "relation": ').usable is False


def test_an_empty_array_is_usable_and_means_no_facts() -> None:
    assert parse_triples("[]").triples == []


def test_a_malformed_item_is_dropped_but_the_rest_is_kept() -> None:
    raw = json.dumps(
        [
            {"subject": "revenue", "relation": "reached", "object": "2.6 billion"},
            {"subject": "revenue", "relation": "reached"},
            "not a triple",
            {"subject": " ", "relation": "x", "object": "y"},
        ]
    )
    parsed = parse_triples(raw)
    assert parsed.triples is not None
    assert len(parsed.triples) == 1
    assert parsed.n_dropped == 3


def test_a_triple_becomes_two_entities_and_a_link_between_them(
    micro_corpus: list[Chunk],
) -> None:
    chunk = micro_corpus[0]
    triple = Triple(subject="Q4 revenue", relation="reached", object="2.6 billion dollars")
    nodes, edges = triples_to_graph(chunk, [triple])

    entity_ids = {node.id for node in nodes if node.kind == "entity"}
    assert entity_ids == {"entity:q4 revenue", "entity:2.6 billion dollars"}
    relations = {edge.relation for edge in edges}
    assert "reached" in relations


def test_the_chunk_is_linked_to_both_ends_so_the_walk_can_reach_it(
    micro_corpus: list[Chunk],
) -> None:
    chunk = micro_corpus[0]
    triple = Triple(subject="revenue", relation="reached", object="2.6 billion")
    _nodes, edges = triples_to_graph(chunk, [triple])
    mentions = [edge for edge in edges if edge.relation == "mentions_entity"]
    assert len(mentions) == 2
    for edge in mentions:
        assert edge.source == "chunk:" + chunk.chunk_id


def test_a_triple_pointing_at_itself_is_dropped(micro_corpus: list[Chunk]) -> None:
    chunk = micro_corpus[0]
    triple = Triple(subject="Revenue", relation="equals", object="revenue")
    _nodes, edges = triples_to_graph(chunk, [triple])
    assert edges == []


def test_every_edge_from_a_triple_carries_the_chunk_it_came_from(
    micro_corpus: list[Chunk],
) -> None:
    chunk = micro_corpus[0]
    triple = Triple(subject="revenue", relation="reached", object="2.6 billion")
    _nodes, edges = triples_to_graph(chunk, [triple])
    for edge in edges:
        assert edge.source_chunk_id == chunk.chunk_id
        assert edge.layer == LAYER_ENTITY


def test_the_build_walks_every_chunk_and_joins_the_graphs(
    micro_corpus: list[Chunk],
) -> None:
    responses = {}
    for chunk in micro_corpus:
        responses[build_prompt(chunk)] = _answer(("revenue", "reported_in", chunk.section_slug))
    llm = FakeLLM(responses)

    nodes, edges, stats = extract(llm, micro_corpus)

    assert len(llm.calls) == len(micro_corpus)
    assert stats.n_parsed == len(micro_corpus)
    assert stats.n_skipped == 0
    assert stats.n_triples == len(micro_corpus)
    assert missing_endpoints(nodes, edges) == []
    assert chunk_edges_are_self_labelled(edges) == []


def test_a_chunk_whose_answer_is_garbage_is_skipped_and_counted(
    micro_corpus: list[Chunk],
) -> None:
    # FakeLLM with no canned answer returns its stub, which is not JSON, so every
    # chunk falls back. That is the shape of a bad prompt or a bad model.
    _nodes, edges, stats = extract(FakeLLM(), micro_corpus)
    assert edges == []
    assert stats.n_skipped == len(micro_corpus)
    assert stats.fallback_rate == 1.0


def test_the_fallback_rate_is_the_share_of_unreadable_chunks(
    micro_corpus: list[Chunk],
) -> None:
    good = micro_corpus[0]
    responses = {build_prompt(good): _answer(("revenue", "reached", "2.6 billion"))}
    _nodes, _edges, stats = extract(FakeLLM(responses), micro_corpus[:2])
    assert stats.n_parsed == 1
    assert stats.n_skipped == 1
    assert stats.fallback_rate == 0.5


def test_a_dropped_item_is_counted_without_failing_the_chunk(
    micro_corpus: list[Chunk],
) -> None:
    chunk = micro_corpus[0]
    raw = json.dumps(
        [
            {"subject": "revenue", "relation": "reached", "object": "2.6 billion"},
            {"nonsense": True},
        ]
    )
    _nodes, _edges, stats = extract(FakeLLM({build_prompt(chunk): raw}), [chunk])
    assert stats.n_parsed == 1
    assert stats.n_triples == 1
    assert stats.n_dropped_items == 1


def test_the_fallback_rate_of_an_empty_run_is_zero() -> None:
    stats = LLMBuildStats(n_chunks=0, n_parsed=0, n_skipped=0, n_triples=0, n_dropped_items=0)
    assert stats.fallback_rate == 0.0


def test_a_second_build_reuses_cached_triples(
    micro_corpus: list[Chunk], tmp_path
) -> None:
    chunks = micro_corpus[:2]
    responses = {
        build_prompt(chunk): _answer(("revenue", "reported_in", chunk.section_slug))
        for chunk in chunks
    }
    cache_path = tmp_path / "graph_llm_cache.jsonl"

    first_llm = FakeLLM(responses)
    _nodes, first_edges, first_stats = extract(
        first_llm,
        chunks,
        cache_path=cache_path,
        model="extractor-v1",
    )
    second_llm = FakeLLM()
    _nodes, second_edges, second_stats = extract(
        second_llm,
        chunks,
        cache_path=cache_path,
        model="extractor-v1",
    )

    assert len(first_llm.calls) == 2
    assert first_stats.n_model_calls == 2
    assert first_stats.n_cache_hits == 0
    assert second_llm.calls == []
    assert second_stats.n_model_calls == 0
    assert second_stats.n_cache_hits == 2
    assert second_edges == first_edges


def test_changing_one_chunk_invalidates_only_that_cache_row(
    micro_corpus: list[Chunk], tmp_path
) -> None:
    original = micro_corpus[:2]
    responses = {
        build_prompt(chunk): _answer(("revenue", "reported_in", chunk.section_slug))
        for chunk in original
    }
    cache_path = tmp_path / "graph_llm_cache.jsonl"
    extract(FakeLLM(responses), original, cache_path=cache_path, model="extractor-v1")

    changed = replace(original[1], text=original[1].text + " Updated fact.")
    changed_answer = _answer(("revenue", "changed_in", changed.section_slug))
    llm = FakeLLM({build_prompt(changed): changed_answer})
    _nodes, _edges, stats = extract(
        llm,
        [original[0], changed],
        cache_path=cache_path,
        model="extractor-v1",
    )

    assert len(llm.calls) == 1
    assert stats.n_cache_hits == 1
    assert stats.n_model_calls == 1


def test_changing_the_model_invalidates_the_cache(
    micro_corpus: list[Chunk], tmp_path
) -> None:
    chunk = micro_corpus[0]
    response = {build_prompt(chunk): _answer(("revenue", "reported_in", "Q4"))}
    cache_path = tmp_path / "graph_llm_cache.jsonl"
    extract(FakeLLM(response), [chunk], cache_path=cache_path, model="extractor-v1")

    llm = FakeLLM(response)
    _nodes, _edges, stats = extract(
        llm,
        [chunk],
        cache_path=cache_path,
        model="extractor-v2",
    )

    assert len(llm.calls) == 1
    assert stats.n_cache_hits == 0
    assert stats.model == "extractor-v2"
