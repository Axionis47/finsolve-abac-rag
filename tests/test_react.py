"""Planner-only ReAct query strategy behavior and failure boundaries."""

from __future__ import annotations

from dataclasses import replace

from finrag.llm.fake import ScriptedLLM
from finrag.models import Chunk, Hit
from finrag.querying.base import QueryRequest
from finrag.querying.react import MAX_HOPS, ReactQueryStrategy
from finrag.querying.react_prompt import parse_step
from finrag.querying.registry import available_query_strategies


class RecordingRetriever:
    name = "recording"

    def __init__(self, chunks: list[Chunk], routes: dict[str, list[int]] | None = None):
        self.chunks = chunks
        self.routes = routes or {}
        self.calls: list[tuple[str, str, int]] = []

    def search(self, query: str, role: str, k: int = 10) -> list[Hit]:
        self.calls.append((query, role, k))
        indexes = self.routes.get(query, list(range(len(self.chunks))))
        hits: list[Hit] = []
        for rank, index in enumerate(indexes[:k], start=1):
            chunk = self.chunks[index]
            hits.append(
                Hit(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    section_slug=chunk.section_slug,
                    rank=rank,
                    score=1.0 / rank,
                    retriever=self.name,
                    owner_dept=chunk.owner_dept,
                    doc_type=chunk.doc_type,
                    sensitivity=chunk.sensitivity,
                    allowed_roles=chunk.allowed_roles,
                    text=chunk.text,
                )
            )
        return hits


def request(question: str = "combine the facts") -> QueryRequest:
    return QueryRequest(question=question, role="c_level", k=10)


def test_finish_parser_requires_structured_existing_unique_evidence() -> None:
    step, problem = parse_step(
        '{"thought":"done","action":"finish","evidence":[2,1]}', 2
    )
    assert problem is None
    assert step is not None and step.evidence == (2, 1)

    step, problem = parse_step(
        '{"thought":"done","action":"finish","evidence":[1,1]}', 2
    )
    assert step is None
    assert problem is not None and problem.kind == "invalid_evidence"


def test_react_runs_focused_searches_and_returns_only_selected_evidence(
    micro_corpus,
) -> None:
    llm = ScriptedLLM(
        [
            '{"thought":"first","action":"search","query":"revenue"}',
            '{"thought":"second","action":"search","query":"campaign spend"}',
            '{"thought":"done","action":"finish","evidence":[2]}',
        ]
    )
    retriever = RecordingRetriever(
        micro_corpus, {"revenue": [0], "campaign spend": [2]}
    )
    result = ReactQueryStrategy(micro_corpus, llm).execute(request(), retriever)

    assert [call[0] for call in retriever.calls] == ["revenue", "campaign spend"]
    assert [hit.chunk_id for hit in result.hits] == [micro_corpus[2].chunk_id]
    assert [item.query for item in result.trace] == ["revenue", "campaign spend"]
    assert result.stats["hops"] == 2


def test_invalid_finish_gets_exactly_one_retry(micro_corpus) -> None:
    llm = ScriptedLLM(
        [
            '{"thought":"look","action":"search","query":"revenue"}',
            '{"thought":"bad","action":"finish","evidence":[99]}',
            '{"thought":"fixed","action":"finish","evidence":[1]}',
        ]
    )
    retriever = RecordingRetriever(micro_corpus, {"revenue": [0]})
    result = ReactQueryStrategy(micro_corpus, llm).execute(request(), retriever)
    assert len(llm.calls) == 3
    assert result.stats["invalid_evidence"] == 1
    assert [hit.chunk_id for hit in result.hits] == [micro_corpus[0].chunk_id]


def test_repeated_query_is_rejected_then_the_retry_can_finish(micro_corpus) -> None:
    llm = ScriptedLLM(
        [
            '{"thought":"look","action":"search","query":"  Revenue "}',
            '{"thought":"again","action":"search","query":"revenue"}',
            '{"thought":"done","action":"finish","evidence":[1]}',
        ]
    )
    retriever = RecordingRetriever(micro_corpus, {"  Revenue ": [0]})
    result = ReactQueryStrategy(micro_corpus, llm).execute(request(), retriever)
    assert len(retriever.calls) == 1
    assert result.stats["repeated_queries"] == 1
    assert len(result.hits) == 1


def test_model_failure_falls_back_to_one_direct_search(micro_corpus) -> None:
    class BrokenLLM:
        name = "broken"

        def generate(self, prompt: str, system: str | None = None) -> str:
            raise RuntimeError("offline")

    retriever = RecordingRetriever(micro_corpus)
    result = ReactQueryStrategy(micro_corpus, BrokenLLM()).execute(
        request("original question"), retriever
    )
    assert [call[0] for call in retriever.calls] == ["original question"]
    assert result.trace[0].kind == "react_fallback"
    assert result.stats["fallback_searches"] == 1


def test_react_never_runs_more_than_four_planned_searches(micro_corpus) -> None:
    replies = [
        f'{{"thought":"hop","action":"search","query":"query {number}"}}'
        for number in range(1, MAX_HOPS + 1)
    ]
    replies.extend(["not json", "still not json"])
    routes = {f"query {number}": [number - 1] for number in range(1, MAX_HOPS + 1)}
    retriever = RecordingRetriever(micro_corpus, routes)
    result = ReactQueryStrategy(micro_corpus, ScriptedLLM(replies)).execute(
        request(), retriever
    )
    assert len(retriever.calls) == MAX_HOPS
    assert result.stats["hops"] == MAX_HOPS
    assert result.stats["forced_finishes"] == 1
    assert len(result.hits) == MAX_HOPS


def test_secondary_role_mask_drops_a_buggy_retrievers_leak(micro_corpus) -> None:
    leaked = replace(micro_corpus[0], allowed_roles=("c_level",))
    retriever = RecordingRetriever([leaked])
    llm = ScriptedLLM(
        ['{"thought":"look","action":"search","query":"secret"}',
         '{"thought":"done","action":"finish","evidence":[]}']
    )
    strategy = ReactQueryStrategy(micro_corpus, llm)
    result = strategy.execute(
        QueryRequest(question="secret", role="marketing", k=5), retriever
    )
    assert result.hits == []


def test_query_registry_is_the_complete_public_set() -> None:
    assert available_query_strategies() == ["direct", "hyde", "react"]
