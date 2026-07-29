"""The floor: count how many query words a chunk contains, rank by that.

This is the simplest retriever that can honestly be called one. It exists for
two reasons. First, it proves the plumbing end to end (chunks in, role mask,
ranked `Hit`s out, harness scores them) before BM25 or embeddings arrive.
Second, it sets a floor: every later mode has to beat plain word overlap, or it
is not earning its complexity.

No IDF, no length normalisation, no stemming. Case folded, punctuation split
off, repeated words counted once. Ties break on chunk_id so a run is repeatable.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from finrag.models import Chunk, Hit
from finrag.retrieval.base import visible_ids

# Words are runs of letters, digits and underscores. Everything else is a
# separator, so "2.6 billion" gives the terms "2", "6" and "billion".
WORD_PATTERN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, in the order they appear."""
    return WORD_PATTERN.findall(text.lower())


def unique_terms(text: str) -> set[str]:
    """The distinct lowercase words in a piece of text."""
    return set(tokenize(text))


def overlap_score(query_terms: set[str], chunk_terms: set[str]) -> int:
    """How many distinct query words the chunk contains."""
    matched = 0
    for term in query_terms:
        if term in chunk_terms:
            matched += 1
    return matched


class NaiveRetriever:
    """Term-overlap search over chunk text, role masked before ranking."""

    name = "naive"

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = list(chunks)
        # Tokenising every chunk once keeps a full eval run cheap.
        self._terms_by_id: dict[str, set[str]] = {}
        for chunk in self.chunks:
            self._terms_by_id[chunk.chunk_id] = unique_terms(chunk.text)

    def search(self, query: str, role: str, k: int = 10) -> list[Hit]:
        """Best `k` chunks this role may see, ranked by word overlap."""
        # The mask comes FIRST, before any scoring. A denied chunk must never
        # take up a rank slot, not even one it would have lost anyway.
        allowed = visible_ids(self.chunks, role)
        query_terms = unique_terms(query)
        if not query_terms:
            return []

        scored: list[tuple[int, str, Chunk]] = []
        for chunk in self.chunks:
            if chunk.chunk_id not in allowed:
                continue
            score = overlap_score(query_terms, self._terms_by_id[chunk.chunk_id])
            if score > 0:
                scored.append((score, chunk.chunk_id, chunk))

        # Higher overlap first; chunk_id breaks ties so the order is stable.
        scored.sort(key=lambda item: (-item[0], item[1]))

        hits: list[Hit] = []
        rank = 1
        for score, _chunk_id, chunk in scored[:k]:
            hits.append(self._to_hit(chunk, rank, score, query_terms))
            rank += 1
        return hits

    def _to_hit(self, chunk: Chunk, rank: int, score: int, query_terms: set[str]) -> Hit:
        """Copy the chunk attributes the policy replay and the metrics need."""
        matched = []
        for term in sorted(query_terms):
            if term in self._terms_by_id[chunk.chunk_id]:
                matched.append(term)

        return Hit(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            section_slug=chunk.section_slug,
            rank=rank,
            score=float(score),
            retriever=self.name,
            owner_dept=chunk.owner_dept,
            doc_type=chunk.doc_type,
            sensitivity=chunk.sensitivity,
            allowed_roles=chunk.allowed_roles,
            text=chunk.text,
            debug={"matched_terms": matched, "n_query_terms": len(query_terms)},
        )
