"""The agent as a retrieval mode: the union of what it read, ranked by what it used.

`ReactAgent` produces a transcript. The eval harness, the sweep and the API all
want a ranked list of `Hit`s, so this file turns one into the other:

  * the DEDUPED UNION of every observation across every hop is the candidate set.
    Not the last hop, not the best hop. A multi-hop question is answered from two
    places, and a result list that kept only one of them would score the mode on
    the half it happened to look at last;
  * the chunks the answer CITED come first, in citation order, then the rest in
    the order the agent found them. The agent read every observation and named
    the ones it used, which is a better ordering signal than a score, and scores
    from four different searches are not on one scale anyway;
  * THE LAST LOCK. Every hit is checked against `visible_ids` on the way out,
    even though the agent masked every observation before the model saw it. Same
    double-lock spirit as `RerankRetriever.search`: this is the layer boundary,
    and a bug between the two checks must not become a leak.

`search_with_answer` is the extra door `generate.py` uses. This mode has already
written an answer with the whole transcript in front of it; paying a second model
call to write it again from five excerpts would cost more and know less.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import replace

from finrag.models import Chunk, Hit
from finrag.react import MODE_NAME, AgentRun, ReactAgent, new_stats
from finrag.retrieval.base import visible_ids

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def cited_numbers(answer: str | None, n_observations: int) -> list[int]:
    """Observation numbers the answer cites, in citation order, each once.

    A number outside the range is dropped rather than guessed at, exactly as
    `prompting.parse_citations` drops one it cannot resolve.
    """
    if not answer:
        return []
    numbers: list[int] = []
    for match in _CITATION_PATTERN.finditer(answer):
        number = int(match.group(1))
        if number < 1 or number > n_observations:
            continue
        if number not in numbers:
            numbers.append(number)
    return numbers


def rank_observations(observations: Sequence[Hit], cited: Sequence[int]) -> list[Hit]:
    """Cited chunks first, in citation order, then the rest as they were found."""
    ordered: list[Hit] = []
    taken: set[int] = set()
    for number in cited:
        ordered.append(observations[number - 1])
        taken.add(number)
    for number in range(1, len(observations) + 1):
        if number not in taken:
            ordered.append(observations[number - 1])
    return ordered


def renumber_citations(answer: str, observations: Sequence[Hit], hits: Sequence[Hit]) -> str:
    """Rewrite `[n]` from observation numbers to the ranks the caller will see.

    Without this the answer says [7] and the response lists five hits: the
    citation convention would break at the one place a reader checks it. A number
    whose chunk did not survive the cut to k is dropped, for the same reason
    `parse_citations` drops one it cannot resolve.
    """
    rank_of: dict[str, int] = {}
    rank = 1
    for hit in hits:
        rank_of[hit.chunk_id] = rank
        rank += 1

    def swap(match: re.Match[str]) -> str:
        number = int(match.group(1))
        if number < 1 or number > len(observations):
            return ""
        found = rank_of.get(observations[number - 1].chunk_id)
        return "" if found is None else f"[{found}]"

    return _CITATION_PATTERN.sub(swap, answer)


class ReactRetriever:
    """The `react` mode. A wrapper around one `ReactAgent` run, like the reranker.

    It owns no search of its own. `totals` accumulates every run's stats across
    the process, because one question's counters say nothing about what the mode
    costs and the eval report has to state that.
    """

    name = MODE_NAME

    def __init__(self, chunks: Sequence[Chunk], agent: ReactAgent) -> None:
        self.chunks = list(chunks)
        self.agent = agent
        self.totals = new_stats()
        self.totals["questions"] = 0

    def search(self, query: str, role: str, k: int = 10) -> list[Hit]:
        """Best `k` chunks this role may see, out of everything the agent read."""
        hits, _answer = self.search_with_answer(query, role, k=k)
        return hits

    def search_with_answer(
        self, query: str, role: str, k: int = 10
    ) -> tuple[list[Hit], str | None]:
        """The ranked union of the agent's observations, and the agent's answer."""
        allowed = visible_ids(self.chunks, role)
        if not allowed:
            # A role that may see nothing gets nothing, and costs no model call.
            return [], None

        run = self.agent.run(query, role)
        self._add_totals(run.stats)

        cited = cited_numbers(run.answer, len(run.observations))
        cited_ids: set[str] = set()
        for number in cited:
            cited_ids.add(run.observations[number - 1].chunk_id)

        hits: list[Hit] = []
        rank = 1
        for hit in rank_observations(run.observations, cited):
            # THE LAST LOCK. The agent masked every observation already; this is
            # the check a bug between there and here cannot get past.
            if hit.chunk_id not in allowed:
                continue
            if rank > k:
                break
            hits.append(self._renumber(hit, rank, hit.chunk_id in cited_ids, run))
            rank += 1

        answer = run.answer
        if answer is not None:
            answer = renumber_citations(answer, run.observations, hits)
        return hits, answer

    def _renumber(self, hit: Hit, rank: int, cited: bool, run: AgentRun) -> Hit:
        """The same hit at its new rank, carrying the run's stats and hop queries.

        `Hit` is frozen, so this is a copy. `score` is left as the base
        retriever's, because the agent gave an ordering and not a number, and a
        fake score disagreeing with the order would be the worse of two lies.
        """
        debug = dict(hit.debug)
        debug["base_rank"] = hit.rank
        debug["base_retriever"] = hit.retriever
        debug["cited"] = cited
        debug["hop_queries"] = list(run.hop_queries)
        debug["agent"] = dict(run.stats)
        return replace(hit, rank=rank, retriever=self.name, debug=debug)

    def _add_totals(self, stats: dict[str, int]) -> None:
        self.totals["questions"] += 1
        for key in stats:
            self.totals[key] = self.totals.get(key, 0) + stats[key]
