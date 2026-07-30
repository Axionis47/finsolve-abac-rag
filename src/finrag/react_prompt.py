"""What the ReAct agent reads and what it is allowed to say back.

Pure functions and two small records: text in, text out. No retriever, no
model, no policy. They live apart from `react.py` for the same reason
`prompting.py` lives apart from `generate.py`: the two things that are easy to
get wrong here, what the model is told and what counts as a usable reply, can be
read and tested without running a loop.

THE ROLE IS NOT AN ARGUMENT TO ANY FUNCTION IN THIS FILE. That is the point, not
an oversight. `build_agent_prompt` cannot leak the role into the prompt because
it is never given one, so no future edit can quietly start letting the model
reason about permissions. Masking is the tool's job, and the tool lives next
door.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from finrag.models import Hit
from finrag.prompting import REFUSAL_TOKEN, excerpt

# Excerpt length per observation, the same as the reranker's. Long enough for
# the heading breadcrumb and the first substantive sentence, short enough that
# twenty-four of them stay one cheap call.
OBSERVATION_CHARS = 400

ACTION_SEARCH = "search"
ACTION_FINISH = "finish"

SYSTEM_PROMPT = (
    "You are a research agent for FinSolve's internal assistant. You search a "
    "document corpus that has already been filtered to what this user may see, "
    "and you answer only from what your searches return. Never use outside "
    "knowledge and never guess at a number. You reply with one JSON object and "
    "nothing else: no prose around it, no code fence, no explanation."
)


@dataclass(frozen=True)
class Step:
    """One decision by the model: what it thought, and what it wants to do."""

    thought: str
    action: str
    argument: str


@dataclass(frozen=True)
class Problem:
    """Why a reply could not be used. `kind` is the stats key that counts it."""

    kind: str
    message: str


def parse_step(reply: str) -> tuple[Step | None, Problem | None]:
    """The step in `reply`, or the reason it is not one. Exactly one is None.

    Defensive in the same way `rerank.parse_order` is: models wrap JSON in code
    fences and prefix it with a sentence. We take the text between the first `{`
    and the last `}` and require an object with a known action and, for a search,
    something to search for.
    """
    start = reply.find("{")
    end = reply.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None, Problem("parse_failures", "there was no JSON object in your reply")

    try:
        parsed = json.loads(reply[start : end + 1])
    except json.JSONDecodeError:
        return None, Problem("parse_failures", "your reply was not valid JSON")
    if not isinstance(parsed, dict):
        return None, Problem("parse_failures", "your reply was JSON but not an object")

    action = str(parsed.get("action", "")).strip().lower()
    if action not in (ACTION_SEARCH, ACTION_FINISH):
        return None, Problem(
            "unknown_actions",
            f"'{action}' is not an action. Use \"search\" or \"finish\"",
        )

    argument = str(parsed.get("argument", "")).strip()
    if action == ACTION_SEARCH and not argument:
        return None, Problem("empty_searches", 'a search needs a query in "argument"')

    return Step(str(parsed.get("thought", "")).strip(), action, argument), None


def normalise_query(query: str) -> str:
    """Queries that differ only in case or spacing are the same query."""
    return " ".join(query.lower().split())


def build_agent_prompt(
    question: str,
    hops: Sequence[dict[str, object]],
    observations: Sequence[Hit],
    searches_left: int,
    problem: str | None = None,
    observation_chars: int = OBSERVATION_CHARS,
) -> str:
    """The prompt for one turn: the question, the story so far, and the ask.

    Observations are numbered once, across all hops, so a citation in the final
    answer means the same thing as a number in the middle of the transcript.
    `hops` entries are `{"thought", "query", "numbers"}`, where `numbers` indexes
    into `observations` from 1.
    """
    lines: list[str] = [f"Question: {question}", ""]

    if not hops:
        lines.append("You have not searched yet.")
        lines.append("")
    for number, hop in enumerate(hops, start=1):
        lines.append(f"Hop {number}")
        lines.append(f"  thought: {hop['thought']}")
        lines.append(f"  search: {hop['query']}")
        numbers: Sequence[int] = hop["numbers"]  # type: ignore[assignment]
        if not numbers:
            lines.append("  observations: nothing new")
        for index in numbers:
            hit = observations[index - 1]
            lines.append(f"  [{index}] section: {hit.section_slug}")
            lines.append(f"      {excerpt(hit.text, observation_chars)}")
        lines.append("")

    if problem:
        lines.append(f"Your last reply could not be used: {problem}.")
        lines.append("This is your only retry. Reply with one JSON object and nothing else.")
        lines.append("")

    if searches_left > 0:
        lines.append(f"You may run {searches_left} more search(es), or finish now.")
    else:
        lines.append("You have no searches left. Finish now, from the observations above.")
    lines.append(
        'Reply with one JSON object: {"thought": "...", "action": "...", "argument": "..."}'
    )
    lines.append("")
    lines.append("Rules:")
    if searches_left > 0:
        lines.append(
            '- action "search": "argument" is a new search query for ONE fact you still '
            "need. A question that spans two departments or two quarters needs one search "
            "for each; rewording the whole question again returns what you already have."
        )
    lines.append(
        '- action "finish": "argument" is the final answer, in a short paragraph, citing '
        "the observation numbers in square brackets, e.g. [1][3]."
    )
    lines.append(
        "- Answer with what the observations do support, and say plainly which part of "
        "the question they do not. A partial answer with its gap named is more useful "
        "than no answer."
    )
    lines.append(
        "- Use exactly "
        f"{REFUSAL_TOKEN} as the argument only when the observations say nothing "
        "relevant to the question at all."
    )
    lines.append("- Answer from the observations and nothing else.")
    return "\n".join(lines)
