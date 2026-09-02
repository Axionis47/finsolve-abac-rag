"""A bounded ReAct loop that plans searches and selects evidence, not answers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from finrag.llm.base import LLMBackend
from finrag.models import Chunk, Hit
from finrag.querying.base import QueryRequest, QueryResult, QueryTrace
from finrag.querying.react_prompt import (
    ACTION_FINISH,
    OBSERVATION_CHARS,
    SYSTEM_PROMPT,
    Problem,
    Step,
    build_agent_prompt,
    normalise_query,
    parse_step,
)
from finrag.retrieval.base import Retriever, visible_ids

MAX_HOPS = 4
HOP_K = 6


def new_stats() -> dict[str, int]:
    return {
        "hops": 0,
        "llm_calls": 0,
        "forced_finishes": 0,
        "parse_failures": 0,
        "unknown_actions": 0,
        "empty_searches": 0,
        "invalid_evidence": 0,
        "repeated_queries": 0,
        "call_failures": 0,
        "fallback_searches": 0,
        "observations": 0,
        "masked_out": 0,
    }


class ReactQueryStrategy:
    name = "react"

    def __init__(
        self,
        chunks: Sequence[Chunk],
        llm: LLMBackend,
        hop_k: int = HOP_K,
        max_hops: int = MAX_HOPS,
        observation_chars: int = OBSERVATION_CHARS,
    ) -> None:
        if max_hops < 1:
            raise ValueError("a ReAct query strategy needs at least one hop")
        self.chunks = list(chunks)
        self.llm = llm
        self.hop_k = hop_k
        self.max_hops = max_hops
        self.observation_chars = observation_chars
        self.last_failure: str | None = None
        self.totals = new_stats()
        self.totals["questions"] = 0

    def execute(self, request: QueryRequest, retriever: Retriever) -> QueryResult:
        allowed = visible_ids(self.chunks, request.role)
        if not allowed:
            return QueryResult([], stats=new_stats())

        stats = new_stats()
        observations: list[Hit] = []
        seen_ids: set[str] = set()
        seen_queries: set[str] = set()
        hops: list[dict[str, object]] = []
        trace: list[QueryTrace] = []
        problem: Problem | None = None
        selected: tuple[int, ...] | None = None

        while len(hops) < self.max_hops:
            prompt = build_agent_prompt(
                request.question,
                hops,
                observations,
                self.max_hops - len(hops),
                problem.message if problem else None,
                self.observation_chars,
            )
            step, trouble = self._decide(prompt, stats, seen_queries, len(observations))
            if step is None:
                stats[trouble.kind if trouble else "parse_failures"] += 1
                if problem is None:
                    problem = trouble
                    continue
                stats["forced_finishes"] += 1
                break

            problem = None
            if step.action == ACTION_FINISH:
                selected = step.evidence
                break

            numbers = self._observe(
                step.query,
                request.role,
                allowed,
                retriever,
                observations,
                seen_ids,
                stats,
            )
            seen_queries.add(normalise_query(step.query))
            hops.append({"thought": step.thought, "query": step.query, "numbers": numbers})
            trace.append(QueryTrace(kind="react", query=step.query, n_hits=len(numbers)))
        else:
            stats["forced_finishes"] += 1
            finish_problem: Problem | None = None
            for _attempt in range(2):
                prompt = build_agent_prompt(
                    request.question,
                    hops,
                    observations,
                    0,
                    finish_problem.message if finish_problem else None,
                    self.observation_chars,
                )
                step, trouble = self._decide(
                    prompt, stats, seen_queries, len(observations)
                )
                if step is not None and step.action == ACTION_FINISH:
                    selected = step.evidence
                    break
                if trouble is None:
                    trouble = Problem(
                        "unknown_actions", "all four searches are used; finish now"
                    )
                stats[trouble.kind] += 1
                finish_problem = trouble
            # Do not issue a fifth search. If the planner searched four times but
            # could not format its final selection, preserve the evidence it saw.
            if selected is None and observations:
                selected = tuple(range(1, len(observations) + 1))

        if selected is None or not observations:
            stats["fallback_searches"] += 1
            fallback: list[Hit] = []
            fallback_seen: set[str] = set()
            for hit in retriever.search(request.question, request.role, k=request.k):
                if hit.chunk_id not in allowed:
                    stats["masked_out"] += 1
                    continue
                if hit.chunk_id not in fallback_seen:
                    fallback.append(hit)
                    fallback_seen.add(hit.chunk_id)
            trace.append(QueryTrace(kind="react_fallback", query=request.question, n_hits=len(fallback)))
            result_hits = fallback
        elif not selected:
            result_hits = []
        else:
            result_hits = self._rank_selected(observations, selected, request.k, trace, stats)

        stats["hops"] = len(hops)
        stats["observations"] = len(observations)
        self._add_totals(stats)
        return QueryResult(
            hits=result_hits,
            generated_queries=[item.query for item in trace],
            trace=trace,
            stats=stats,
        )

    def _decide(
        self,
        prompt: str,
        stats: dict[str, int],
        seen_queries: set[str],
        n_observations: int,
    ) -> tuple[Step | None, Problem | None]:
        stats["llm_calls"] += 1
        try:
            reply = self.llm.generate(prompt, system=SYSTEM_PROMPT)
        except Exception as error:  # noqa: BLE001 - the strategy has a direct fallback
            stats["call_failures"] += 1
            self.last_failure = f"call failed: {type(error).__name__}: {error}"
            return None, Problem("parse_failures", self.last_failure)

        step, trouble = parse_step(reply, n_observations)
        if step is None:
            return None, trouble
        if step.action != ACTION_FINISH and normalise_query(step.query) in seen_queries:
            return None, Problem(
                "repeated_queries", f"you already searched for '{step.query}'"
            )
        return step, None

    def _observe(
        self,
        query: str,
        role: str,
        allowed: frozenset[str],
        retriever: Retriever,
        observations: list[Hit],
        seen_ids: set[str],
        stats: dict[str, int],
    ) -> list[int]:
        numbers: list[int] = []
        for hit in retriever.search(query, role, k=self.hop_k):
            if hit.chunk_id not in allowed:
                stats["masked_out"] += 1
                continue
            if hit.chunk_id in seen_ids:
                continue
            seen_ids.add(hit.chunk_id)
            observations.append(hit)
            numbers.append(len(observations))
        return numbers

    def _rank_selected(
        self,
        observations: Sequence[Hit],
        selected: Sequence[int],
        k: int,
        trace: Sequence[QueryTrace],
        stats: dict[str, int],
    ) -> list[Hit]:
        hits: list[Hit] = []
        for rank, number in enumerate(selected[:k], start=1):
            hit = observations[number - 1]
            debug = dict(hit.debug)
            debug.update(
                {
                    "base_rank": hit.rank,
                    "selected_evidence": number in selected,
                    "query_trace": [item.to_dict() for item in trace],
                    "query_stats": dict(stats),
                }
            )
            hits.append(replace(hit, rank=rank, debug=debug))
        return hits

    def _add_totals(self, stats: dict[str, int]) -> None:
        self.totals["questions"] += 1
        for key, value in stats.items():
            self.totals[key] = self.totals.get(key, 0) + value
