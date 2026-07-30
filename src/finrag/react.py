"""The ReAct loop: the model plans the searches, the tool keeps the mask.

Every mode before this one runs a fixed pipeline: one query in, one ranked list
out. A multi-hop question ("finance names a risk, which campaign mitigates it")
needs two different searches, and no amount of fusion or reranking of ONE query
finds both halves. This mode gives the model the wheel: think, search, read what
came back, search again, then answer.

Four constraints shape it, and three of them are about keeping the agency in a
box:

  * THE MODEL NEVER LEARNS THE ROLE. Not in the system prompt, not in the
    question block, not in an observation. A model that knew the role could
    reason about permissions, and reasoning about permissions is how an agent
    talks itself into asking for something it should not have. The role travels
    only as an argument to `base.search(query, role, k)`.
  * ONE TOOL, AND IT IS A MASKED RETRIEVER. No "read document", no "list
    sections", no chunk-id lookup. Everything the model can ever see comes back
    from a registry-built retriever that masks before it ranks, so the ABAC
    guarantee here is the same one `hybrid` already has rather than a new one to
    argue about. `_observe` re-checks `visible_ids` before an excerpt reaches the
    prompt; `react_retriever.py` re-checks it again before a hit leaves.
  * THE LOOP MUST TERMINATE AND MUST BE CHEAP. At most `MAX_HOPS` searches, one
    retry per failure, then a forced finish. Every call is counted, because an
    agent whose cost you cannot state is an agent you cannot compare against a
    mode that costs one call.
  * A BROKEN MODEL MUST NOT TURN RETRIEVAL OFF. If the loop ends with nothing
    observed, one plain search of the question runs anyway
    (`fallback_searches`). That holds the floor at the base mode instead of at
    zero, and it is what lets the offline leakage sweep test this mode at all.

Three files, not one: `react_prompt.py` is what the model reads and writes,
this file is the loop, `react_retriever.py` is what the loop hands back. See
DECISIONS R7 for why the usual one-file-per-mode shape did not survive here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from finrag.llm.base import LLMBackend
from finrag.models import Chunk, Hit
from finrag.react_prompt import (
    ACTION_FINISH,
    ACTION_SEARCH,
    OBSERVATION_CHARS,
    SYSTEM_PROMPT,
    Problem,
    Step,
    build_agent_prompt,
    normalise_query,
    parse_step,
)
from finrag.retrieval.base import Retriever, visible_ids

# The mode name, and the mode its one tool searches with. `hybrid` because it is
# the best free retriever we have: the agent should pay for planning, not for a
# weaker search underneath.
MODE_NAME = "react"
REACT_BASE_MODE = "hybrid"

# At most four searches. Two is the shape of a multi-hop gold question, and four
# leaves room for one correction and one widening. More than that buys a model
# that wanders, and every hop is a billed call.
MAX_HOPS = 4

# Chunks per hop. Six keeps four hops of observations inside a prompt about the
# size of one rerank call, and the agent can always ask a narrower query.
HOP_K = 6


@dataclass
class AgentRun:
    """Everything one question produced: the answer, the evidence, the bill."""

    question: str
    answer: str | None
    observations: list[Hit] = field(default_factory=list)
    hop_queries: list[str] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)


def new_stats() -> dict[str, int]:
    """The counters one run keeps. Each is a way the loop can go wrong, or cost."""
    return {
        "hops": 0,
        "llm_calls": 0,
        "forced_finishes": 0,
        "parse_failures": 0,
        "unknown_actions": 0,
        "empty_searches": 0,
        "repeated_queries": 0,
        "call_failures": 0,
        "fallback_searches": 0,
        "observations": 0,
        "masked_out": 0,
    }


class ReactAgent:
    """Runs the think-search-observe loop for one question, and counts the cost.

    Holds no state between runs beyond `last_failure`, so two questions cannot
    contaminate each other. `run` returns everything a caller needs, which is
    what lets `ReactRetriever` stay a thin wrapper and a test read one call.
    """

    def __init__(
        self,
        chunks: Sequence[Chunk],
        base: Retriever,
        llm: LLMBackend,
        base_mode: str = REACT_BASE_MODE,
        hop_k: int = HOP_K,
        max_hops: int = MAX_HOPS,
        observation_chars: int = OBSERVATION_CHARS,
    ) -> None:
        if max_hops < 1:
            raise ValueError("an agent needs at least one hop")
        self.chunks = list(chunks)
        self.base = base
        self.llm = llm
        self.base_mode = base_mode
        self.hop_k = hop_k
        self.max_hops = max_hops
        self.observation_chars = observation_chars
        # Why the last model call failed, for the run notes. One line, no stack.
        self.last_failure: str | None = None

    def run(self, question: str, role: str) -> AgentRun:
        """Plan, search, and answer. Never raises: every failure is a counter."""
        allowed = visible_ids(self.chunks, role)
        stats = new_stats()
        observations: list[Hit] = []
        seen_ids: set[str] = set()
        seen_queries: set[str] = set()
        hops: list[dict[str, object]] = []
        problem: Problem | None = None
        answer: str | None = None

        while len(hops) < self.max_hops:
            prompt = build_agent_prompt(
                question,
                hops,
                observations,
                self.max_hops - len(hops),
                problem.message if problem else None,
                self.observation_chars,
            )
            step, trouble = self._decide(prompt, stats, seen_queries)
            if step is None:
                stats[trouble.kind if trouble else "parse_failures"] += 1
                if problem is None:
                    # One retry, with the reason quoted back. A model that wrote
                    # prose usually writes JSON when told which line was wrong.
                    problem = trouble
                    continue
                stats["forced_finishes"] += 1
                break

            problem = None
            if step.action == ACTION_FINISH:
                answer = step.argument
                break

            numbers = self._observe(step.argument, role, allowed, observations, seen_ids, stats)
            seen_queries.add(normalise_query(step.argument))
            hops.append({"thought": step.thought, "query": step.argument, "numbers": numbers})
        else:
            # The while condition, not a break: the hop budget ran out.
            answer = self._force_finish(question, hops, observations, stats)

        if not observations:
            # The model never got a usable search out. Run the plain question
            # once so this mode never returns less than its base mode would, and
            # drop whatever it said, because an answer with no evidence behind it
            # is the one thing this repo refuses to print.
            stats["fallback_searches"] += 1
            self._observe(question, role, allowed, observations, seen_ids, stats)
            answer = None

        stats["hops"] = len(hops)
        stats["observations"] = len(observations)
        queries: list[str] = []
        for hop in hops:
            queries.append(str(hop["query"]))
        return AgentRun(question, answer, observations, queries, stats)

    def _decide(
        self,
        prompt: str,
        stats: dict[str, int],
        seen_queries: set[str],
    ) -> tuple[Step | None, Problem | None]:
        """One model call, parsed, with a repeated search counted as a failure."""
        step, trouble = parse_step(self._ask(prompt, stats))
        if step is None:
            return None, trouble
        if step.action == ACTION_SEARCH and normalise_query(step.argument) in seen_queries:
            # Re-running a query returns the same chunks, which are already in
            # the prompt. That is a loop, not a hop, so it does not get one.
            return None, Problem(
                "repeated_queries",
                f"you already searched for '{step.argument}' and its results are above",
            )
        return step, None

    def _ask(self, prompt: str, stats: dict[str, int]) -> str:
        """Call the model. A transport failure is an empty reply, not an exception.

        The loop's job is to end with something. A backend that is down should
        show up as a forced finish and a `call_failures` count, not as a stack
        trace that takes a fifty-question eval down with it.
        """
        stats["llm_calls"] += 1
        try:
            return self.llm.generate(prompt, system=SYSTEM_PROMPT)
        except Exception as error:  # noqa: BLE001 - any backend failure ends in a forced finish
            stats["call_failures"] += 1
            self.last_failure = f"call failed: {type(error).__name__}: {error}"
            return ""

    def _observe(
        self,
        query: str,
        role: str,
        allowed: frozenset[str],
        observations: list[Hit],
        seen_ids: set[str],
        stats: dict[str, int],
    ) -> list[int]:
        """Search, mask, drop what was already seen, and number what is left.

        The mask here is the lock that matters most in this rung: it stands
        between a hit and the prompt, so a denied chunk cannot be read by the
        model even if the base retriever were compromised tomorrow.
        """
        numbers: list[int] = []
        for hit in self.base.search(query, role, k=self.hop_k):
            if hit.chunk_id not in allowed:
                stats["masked_out"] += 1
                continue
            if hit.chunk_id in seen_ids:
                continue
            seen_ids.add(hit.chunk_id)
            observations.append(hit)
            numbers.append(len(observations))
        return numbers

    def _force_finish(
        self,
        question: str,
        hops: Sequence[dict[str, object]],
        observations: Sequence[Hit],
        stats: dict[str, int],
    ) -> str | None:
        """Out of hops: one last call that may only answer or refuse."""
        stats["forced_finishes"] += 1
        prompt = build_agent_prompt(question, hops, observations, 0, None, self.observation_chars)
        step, trouble = parse_step(self._ask(prompt, stats))
        if step is None:
            stats[trouble.kind if trouble else "parse_failures"] += 1
            return None
        if step.action != ACTION_FINISH:
            # It asked to search after being told it could not. We do not honour
            # that, and we do not pay for another call to ask again.
            stats["unknown_actions"] += 1
            return None
        return step.argument
