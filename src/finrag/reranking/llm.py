"""One-call LLM reranking over candidates from any retrieval strategy."""

from __future__ import annotations

import json
from collections.abc import Sequence

from finrag.llm.base import LLMBackend
from finrag.models import Hit

RERANK_DEPTH = 20
EXCERPT_CHARS = 400
RETRY_TRIES = 3

SYSTEM_PROMPT = (
    "You rank search results. Answer with a JSON array of numbers and nothing else."
)


def excerpt(text: str, limit: int = EXCERPT_CHARS) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit]


def build_prompt(question: str, hits: Sequence[Hit], limit: int = EXCERPT_CHARS) -> str:
    lines = [f"Question: {question}", "", "Passages:"]
    for number, hit in enumerate(hits, start=1):
        lines.append(f"{number}. {excerpt(hit.text, limit)}")
    lines.extend(
        [
            "",
            f"Order all {len(hits)} passages from most useful to least useful. "
            f"Return every number from 1 to {len(hits)} exactly once.",
        ]
    )
    return "\n".join(lines)


def parse_order(reply: str, n: int) -> list[int] | None:
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
        if isinstance(item, bool):
            return None
        if isinstance(item, int):
            order.append(item)
        elif isinstance(item, str) and item.strip().isdigit():
            order.append(int(item.strip()))
        else:
            return None
    return order if sorted(order) == list(range(1, n + 1)) else None


class LLMReranker:
    name = "llm"

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
        self.last_failure: str | None = None

    def rerank(self, question: str, hits: Sequence[Hit]) -> list[Hit]:
        candidates = list(hits)
        if len(candidates) < 2:
            return candidates
        order = self._ask(build_prompt(question, candidates, self.excerpt_chars), len(candidates))
        if order is None:
            self.fallbacks += 1
            return candidates
        return [candidates[position - 1] for position in order]

    def _ask(self, prompt: str, n: int) -> list[int] | None:
        for _attempt in range(self.tries):
            self.calls += 1
            try:
                reply = self.llm.generate(prompt, system=SYSTEM_PROMPT)
            except Exception as error:  # noqa: BLE001 - failures preserve input order
                self.last_failure = f"call failed: {type(error).__name__}: {error}"
                continue
            order = parse_order(reply, n)
            if order is not None:
                return order
            self.last_failure = f"unparsable reply: {reply.strip()[:120]!r}"
        return None
