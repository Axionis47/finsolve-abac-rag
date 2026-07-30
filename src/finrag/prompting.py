"""Turning hits into a prompt, and a reply back into citations.

Pure functions only: text in, text out, no retriever, no model, no policy. They
live apart from `generate.py` so the two things that are easy to get wrong, the
grounding instructions and the mapping from `[2]` back to a chunk id, can be read
and tested on their own.

The extractive fallback lives here too. It is what an answer looks like with no
model configured: sentences quoted from the top hit, never rewritten.
"""

from __future__ import annotations

import re
from typing import Any

from finrag.models import Hit
from finrag.retrieval.naive import tokenize

# How much of a chunk goes into the prompt. Long enough for a full section of
# the corpus, short enough that five of them stay a cheap call.
MAX_EXCERPT_CHARS = 1200

# How many sentences the extractive fallback quotes.
EXTRACTIVE_SENTENCES = 2

# The model writes this exact word when the context does not answer the
# question. A fixed token is easier to detect than a polite English refusal.
REFUSAL_TOKEN = "NOT_IN_CONTEXT"

SYSTEM_PROMPT = (
    "You are FinSolve's internal assistant. You answer only from the numbered "
    "excerpts you are given, which have already been filtered to what this user "
    "is allowed to see. Never use outside knowledge and never guess at a number."
)

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def build_prompt(question: str, hits: list[Hit]) -> str:
    """The grounded prompt: numbered excerpts, then the question and the rules.

    Each excerpt carries its chunk id so a human reading the raw prompt can find
    the source, and the citation rule asks for the number, not the id, because a
    long id in the middle of a sentence is what models get wrong.
    """
    lines: list[str] = ["Excerpts you may use:", ""]
    number = 1
    for hit in hits:
        lines.append(f"[{number}] chunk_id: {hit.chunk_id}")
        lines.append(f"    section: {hit.section_slug} (department: {hit.owner_dept})")
        lines.append(f"    text: {excerpt(hit.text)}")
        lines.append("")
        number += 1

    lines.append(f"Question: {question}")
    lines.append("")
    lines.append("Rules:")
    lines.append("- Answer using the excerpts above and nothing else.")
    lines.append("- Cite every claim with the excerpt numbers in square brackets, e.g. [1][3].")
    lines.append(
        f"- If the excerpts do not answer the question, reply with exactly {REFUSAL_TOKEN} "
        "and nothing else."
    )
    lines.append("- Be brief: a short paragraph, no preamble.")
    return "\n".join(lines)


def excerpt(text: str, limit: int = MAX_EXCERPT_CHARS) -> str:
    """The chunk text, cut at `limit` characters with the cut marked."""
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[:limit] + " ..."


def is_refusal(reply: str) -> bool:
    """Did the model say the excerpts do not answer this, or say nothing at all."""
    return not reply.strip() or REFUSAL_TOKEN in reply.upper()


def parse_citations(reply: str, hits: list[Hit]) -> list[str]:
    """Chunk ids for the excerpt numbers the reply actually cites, in order.

    A number outside the range of excerpts is dropped rather than guessed at: it
    is a model mistake, and inventing a source for it would be the one failure
    this whole layer exists to prevent. Repeats collapse to one citation.
    """
    cited: list[str] = []
    for match in _CITATION_PATTERN.finditer(reply):
        number = int(match.group(1))
        if number < 1 or number > len(hits):
            continue
        chunk_id = hits[number - 1].chunk_id
        if chunk_id not in cited:
            cited.append(chunk_id)
    return cited


def extractive_answer(question: str, hit: Hit) -> str:
    """The sentences of this hit that overlap the question, in their own order.

    Same tokenizer as retrieval, so "Q4" is split here the way it was indexed.
    """
    sentences = [part.strip() for part in _SENTENCE_SPLIT.split(hit.text) if part.strip()]
    if not sentences:
        return ""

    question_terms = set(tokenize(question))
    scored: list[tuple[int, int, str]] = []
    for position, sentence in enumerate(sentences):
        overlap = 0
        for term in set(tokenize(sentence)):
            if term in question_terms:
                overlap += 1
        scored.append((overlap, position, sentence))

    # Best overlap first, earlier sentence winning a tie.
    scored.sort(key=lambda item: (-item[0], item[1]))
    best_overlap = scored[0][0]
    if best_overlap == 0:
        # Nothing in the chunk shares a word with the question. Quote the opening
        # sentence rather than a sentence picked at random.
        return sentences[0]

    # A second sentence has to share more than half as many question words as the
    # best one. Without that floor, one shared word like "was" drags an unrelated
    # sentence into the quote.
    chosen: list[tuple[int, int, str]] = []
    for overlap, position, sentence in scored[:EXTRACTIVE_SENTENCES]:
        if overlap * 2 > best_overlap:
            chosen.append((overlap, position, sentence))

    # Back into document order so the quote still reads as prose.
    chosen.sort(key=lambda item: item[1])

    picked: list[str] = []
    for _overlap, _position, sentence in chosen:
        picked.append(sentence)
    return " ".join(picked)


def hit_view(hit: Hit) -> dict[str, Any]:
    """The part of a hit that is safe and useful to send to a client.

    `payload` never appears, because that is where the HR row PII lives and it is
    gated on the separate `view_row` action. `allowed_roles` never appears
    either: who else can see a chunk is nobody's business but the policy's.
    """
    return {
        "chunk_id": hit.chunk_id,
        "doc_id": hit.doc_id,
        "section_slug": hit.section_slug,
        "rank": hit.rank,
        "score": round(hit.score, 4),
        "retriever": hit.retriever,
        "owner_dept": hit.owner_dept,
        "doc_type": hit.doc_type,
        "sensitivity": hit.sensitivity,
        "text": excerpt(hit.text),
    }
