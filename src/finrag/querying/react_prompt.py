"""Prompt and parser for the evidence-planning ReAct query strategy."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from finrag.models import Hit
from finrag.prompting import excerpt

ACTION_SEARCH = "search"
ACTION_FINISH = "finish"
OBSERVATION_CHARS = 400

SYSTEM_PROMPT = (
    "You are a retrieval planner. You may search the provided document retriever, "
    "inspect its observations, and select the evidence needed to answer the question. "
    "Never answer the question yourself. Reply with one JSON object and nothing else."
)


@dataclass(frozen=True)
class Step:
    thought: str
    action: str
    query: str = ""
    evidence: tuple[int, ...] = ()


@dataclass(frozen=True)
class Problem:
    kind: str
    message: str


def parse_step(reply: str, n_observations: int) -> tuple[Step | None, Problem | None]:
    start = reply.find("{")
    end = reply.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None, Problem("parse_failures", "there was no JSON object in your reply")
    try:
        parsed = json.loads(reply[start : end + 1])
    except json.JSONDecodeError:
        return None, Problem("parse_failures", "your reply was not valid JSON")
    if not isinstance(parsed, dict):
        return None, Problem("parse_failures", "your reply was not a JSON object")

    action = str(parsed.get("action", "")).strip().lower()
    thought = str(parsed.get("thought", "")).strip()
    if action == ACTION_SEARCH:
        query = str(parsed.get("query", "")).strip()
        if not query:
            return None, Problem("empty_searches", 'a search needs a non-empty "query"')
        return Step(thought=thought, action=action, query=query), None
    if action != ACTION_FINISH:
        return None, Problem("unknown_actions", 'action must be "search" or "finish"')

    raw_evidence = parsed.get("evidence")
    if not isinstance(raw_evidence, list):
        return None, Problem("parse_failures", 'finish needs an "evidence" JSON list')
    evidence: list[int] = []
    for item in raw_evidence:
        if isinstance(item, bool) or not isinstance(item, int):
            return None, Problem("parse_failures", "evidence entries must be integers")
        if item < 1 or item > n_observations or item in evidence:
            return None, Problem(
                "invalid_evidence", "evidence numbers must be unique existing observations"
            )
        evidence.append(item)
    return Step(thought=thought, action=action, evidence=tuple(evidence)), None


def normalise_query(query: str) -> str:
    return " ".join(query.lower().split())


def build_agent_prompt(
    question: str,
    hops: Sequence[dict[str, object]],
    observations: Sequence[Hit],
    searches_left: int,
    problem: str | None = None,
    observation_chars: int = OBSERVATION_CHARS,
) -> str:
    lines = [f"Question: {question}", ""]
    if not hops:
        lines.extend(["You have not searched yet.", ""])
    for number, hop in enumerate(hops, start=1):
        lines.extend(
            [
                f"Hop {number}",
                f"  thought: {hop['thought']}",
                f"  search: {hop['query']}",
            ]
        )
        numbers: Sequence[int] = hop["numbers"]  # type: ignore[assignment]
        if not numbers:
            lines.append("  observations: nothing new")
        for index in numbers:
            hit = observations[index - 1]
            lines.append(f"  [{index}] section: {hit.section_slug}")
            lines.append(f"      {excerpt(hit.text, observation_chars)}")
        lines.append("")

    if problem:
        lines.extend(
            [
                f"Your last reply could not be used: {problem}.",
                "This is your only retry. Reply with one JSON object.",
                "",
            ]
        )
    if searches_left:
        lines.append(f"You may run {searches_left} more search(es), or finish now.")
    else:
        lines.append("You have no searches left. Finish now.")
    lines.extend(
        [
            'Search: {"thought":"...","action":"search","query":"one focused query"}',
            'Finish: {"thought":"...","action":"finish","evidence":[1,3]}',
            "Use an empty evidence list only when no observation is relevant.",
            "Search for one missing fact at a time; do not rephrase the whole question.",
        ]
    )
    return "\n".join(lines)
