"""LLM reranking: one Gemini call reorders what hybrid already found.

Hybrid gets the right passage into the top twenty far more often than it gets it
to rank 1. Fusion has no idea what the question means, only that two ranked lists
agreed. A model reading the question and twenty excerpts can tell "Q4 2024
revenue" from "Q4 2023 revenue" in a way no ranking arithmetic can.

Three constraints shape the whole file:

  * ONE call per query, not one per passage. Twenty pairwise calls would score
    better and cost twenty times as much, for a rung that has to justify itself
    against free fusion.
  * The model returns POSITIONS, never chunk ids, so it structurally cannot name
    a chunk that was not already in the masked, fused top twenty. That is the
    ABAC invariant, not politeness: a reranker that accepted ids could be talked
    into naming a denied one.
  * Any failure falls back to the hybrid order and is counted. A reranker that
    silently returns garbage is worse than no reranker, and one that raises takes
    the whole eval down over a flaky call.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace

from finrag.llm.base import LLMBackend
from finrag.models import Chunk, Hit
from finrag.retrieval.base import Retriever, visible_ids

# How many hybrid hits to show the model. Twenty passages at 400 characters is
# roughly 2,000 tokens, which is one cheap call.
RERANK_DEPTH = 20

# Excerpt length per passage. Long enough to carry the heading breadcrumb and
# the first substantive sentences, short enough that twenty of them stay cheap.
EXCERPT_CHARS = 400

# Three calls at most before we fall back. The failure this covers is a
# rate-limit blip; an outage should show up as a large fallback count rather than
# as a long wait.
RETRY_TRIES = 3

SYSTEM_PROMPT = (
    "You rank search results. You answer with a JSON array of numbers and "
    "nothing else: no prose, no code fences, no explanation."
)


def excerpt(text: str, limit: int = EXCERPT_CHARS) -> str:
    """The first `limit` characters of `text`, on one line.

    Newlines are flattened because the prompt numbers passages one per line, and
    a passage that breaks across lines would make the list ambiguous to read.
    """
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[:limit]


def build_prompt(question: str, hits: Sequence[Hit], limit: int = EXCERPT_CHARS) -> str:
    """The one prompt: the question, then the numbered excerpts, then the ask."""
    lines: list[str] = []
    lines.append(f"Question: {question}")
    lines.append("")
    lines.append("Passages:")
    number = 1
    for hit in hits:
        lines.append(f"{number}. {excerpt(hit.text, limit)}")
        number += 1
    lines.append("")
    lines.append(
        f"Order all {len(hits)} passages from most useful for answering the question "
        f"to least useful. Reply with a JSON array of the {len(hits)} numbers, most "
        "useful first. Use every number exactly once."
    )
    return "\n".join(lines)


def parse_order(reply: str, n: int) -> list[int] | None:
    """The permutation of 1..n in `reply`, or None if it is not one.

    Defensive on purpose. Models wrap JSON in code fences, prefix it with "Here
    is the order:", or return a list of strings. We take the text between the
    first `[` and the last `]` and require what comes out to be an exact
    permutation. Anything else returns None and the caller falls back.
    """
    start = reply.find("[")
    end = reply.rfind("]")
    if start == -1 or end == -1 or end < start:
        return None

    try:
        parsed = json.loads(reply[start : end + 1])
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, list) or len(parsed) != n:
        return None

    order: list[int] = []
    for item in parsed:
        # A JSON bool is an int in Python, and "3" is a number a model might
        # send, so both are handled explicitly rather than by isinstance alone.
        if isinstance(item, bool):
            return None
        if isinstance(item, int):
            order.append(item)
        elif isinstance(item, str) and item.strip().isdigit():
            order.append(int(item.strip()))
        else:
            return None

    if sorted(order) != list(range(1, n + 1)):
        return None
    return order


class LLMReranker:
    """Reorders a ranked list with one model call, and counts its failures.

    `fallbacks` and `calls` are plain counters on the instance. The eval reads
    them at the end of a run, because "the reranker helped" and "the reranker
    fell back on half the questions" are the same table row otherwise.
    """

    def __init__(
        self,
        llm: LLMBackend,
        depth: int = RERANK_DEPTH,
        excerpt_chars: int = EXCERPT_CHARS,
        tries: int = RETRY_TRIES,
    ) -> None:
        if depth < 1:
            raise ValueError("rerank depth must be at least 1")
        self.llm = llm
        self.depth = depth
        self.excerpt_chars = excerpt_chars
        self.tries = tries

        self.calls = 0
        self.fallbacks = 0
        # Why the last fallback happened, for the run notes. One line, no stack.
        self.last_failure: str | None = None

    def rerank(self, question: str, hits: Sequence[Hit]) -> list[Hit]:
        """The same hits in the model's order, or in the given order on failure.

        Never raises, never drops a hit, never adds one. The returned list is
        always a permutation of what came in, which is what makes the fallback
        safe: the worst case is the hybrid ordering we already had.
        """
        hits = list(hits)
        # One or zero passages have exactly one ordering, so a call would buy
        # nothing and would still be billed.
        if len(hits) < 2:
            return hits

        prompt = build_prompt(question, hits, self.excerpt_chars)
        order = self._ask(prompt, len(hits))
        if order is None:
            self.fallbacks += 1
            return hits

        reordered: list[Hit] = []
        for position in order:
            reordered.append(hits[position - 1])
        return reordered

    def _ask(self, prompt: str, n: int) -> list[int] | None:
        """Call the model up to `tries` times; None means every try failed."""
        for _attempt in range(self.tries):
            self.calls += 1
            try:
                reply = self.llm.generate(prompt, system=SYSTEM_PROMPT)
            except Exception as error:  # noqa: BLE001 - any backend failure falls back
                self.last_failure = f"call failed: {type(error).__name__}: {error}"
                continue

            order = parse_order(reply, n)
            if order is not None:
                return order
            # A retry is worth trying on a bad parse as well as on an error: at
            # temperature 0 the same prompt usually gives the same bad reply, but
            # a truncated or blocked response is transient and does not.
            self.last_failure = f"unparsable reply: {reply.strip()[:120]!r}"
        return None


class RerankRetriever:
    """Hybrid retrieval with an LLM reordering the top `depth` hits.

    A wrapper, not a retriever of its own: `inner` finds the candidates and
    masks them, this class only changes their order and cuts to `k`. Today
    `inner` is always the hybrid retriever, which is why the mode is named
    hybrid_rerank.
    """

    name = "hybrid_rerank"

    def __init__(
        self,
        chunks: Sequence[Chunk],
        inner: Retriever,
        reranker: LLMReranker,
    ) -> None:
        self.chunks = list(chunks)
        self.inner = inner
        self.reranker = reranker

    def search(self, query: str, role: str, k: int = 10) -> list[Hit]:
        """Best `k` chunks this role may see, in the model's preferred order."""
        # The mask comes FIRST here too, even though `inner` has already applied
        # it. Same double-lock spirit as bm25.py: the model sees passage text, so
        # a denied chunk reaching this point would be shown to it before any
        # later filter could help.
        allowed = visible_ids(self.chunks, role)
        if not allowed:
            return []

        # Ask deeper than the caller wants: the right passage is often at rank 12.
        depth = max(k, self.reranker.depth)
        candidates = self.inner.search(query, role, k=depth)

        masked: list[Hit] = []
        for hit in candidates:
            if hit.chunk_id in allowed:
                masked.append(hit)
        if not masked:
            return []

        # Read the counter either side of the call so the hits can say whether
        # the model really decided this order or the fallback did. Reporting
        # `reranked: true` on a fallback would hide exactly what we are counting.
        fallbacks_before = self.reranker.fallbacks
        ordered = self.reranker.rerank(query, masked)
        # A list of one has only one ordering and is returned without a call, so
        # no model decided it either.
        reranked = len(masked) > 1 and self.reranker.fallbacks == fallbacks_before

        hits: list[Hit] = []
        rank = 1
        for hit in ordered[:k]:
            hits.append(self._renumber(hit, rank, reranked))
            rank += 1
        return hits

    def _renumber(self, hit: Hit, rank: int, reranked: bool) -> Hit:
        """The same hit at its new rank, keeping the hybrid rank in debug.

        `score` is left as the fused RRF score rather than invented anew. The
        model gave an ordering, not a number, and a fake score that disagreed
        with the order would be the more confusing of the two lies.

        `Hit` is frozen, so this is a copy. `replace` rather than listing all
        twelve fields, because this class changes exactly three of them and
        naming the other nine would hide which ones those are.
        """
        debug = dict(hit.debug)
        debug["hybrid_rank"] = hit.rank
        debug["reranked"] = reranked
        return replace(hit, rank=rank, retriever=self.name, debug=debug)
