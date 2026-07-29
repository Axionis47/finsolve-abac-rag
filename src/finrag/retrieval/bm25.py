"""BM25 lexical search over an index built once at ingest.

Where this sits: ingest writes `artifacts/chunks.jsonl` and then calls
`build_and_save_index` here to write `artifacts/bm25/`. At query time the
retriever loads that folder. Nothing is rebuilt per query, which is the main
thing the old repo got wrong.

Why BM25 beats the naive word count: it weighs a rare word far more than a common
one (IDF) and stops a long chunk from winning just by being long (length
normalisation). "InstantPay" in one chunk is strong evidence; "quarterly" in forty
is nearly none, and plain overlap cannot tell those two apart.

ONE TOKENIZER, TWO USES. Index and query both go through `tokenize` from naive.py.
They have to: bm25s looks a query word up in the vocabulary built from the indexed
words, so a word split or folded differently at query time is simply not found and
its evidence is lost with no error at all. Hence imported, never re-implemented.
THE INVARIANT: `search` computes the role's visible chunk ids before it scores
anything, and a denied chunk can never take a rank slot. See `_weight_mask`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import bm25s
import numpy as np

from finrag.models import Chunk, Hit
from finrag.retrieval.base import visible_ids
from finrag.retrieval.naive import tokenize

# Saved next to the bm25s files: the chunk ids in index row order. bm25s knows
# only rows 0..n-1, so without this list a reordered chunks.jsonl would quietly
# map scores onto the wrong chunks.
CHUNK_IDS_FILENAME = "chunk_ids.json"

# One of the bm25s files, used to tell "no index here" from "wrong folder".
PARAMS_FILENAME = "params.index.json"


def tokenize_chunks(chunks: Sequence[Chunk]) -> list[list[str]]:
    """Token lists for every chunk, in the order the chunks were given."""
    rows: list[list[str]] = []
    for chunk in chunks:
        rows.append(tokenize(chunk.text))
    return rows


def build_vocab(token_rows: Sequence[Sequence[str]]) -> dict[str, int]:
    """Every token in the corpus, numbered in alphabetical order.

    Why we number the vocabulary rather than letting bm25s do it: bm25s numbers
    tokens by walking a Python `set`, and set order for strings changes from one
    process to the next (hash randomisation). Scores come out the same either way,
    but the saved files come out byte-different on every rebuild. Sorting pins it.
    """
    tokens: set[str] = set()
    for row in token_rows:
        for token in row:
            tokens.add(token)

    vocab: dict[str, int] = {}
    number = 0
    for token in sorted(tokens):
        vocab[token] = number
        number += 1
    return vocab


def build_index(chunks: Sequence[Chunk]) -> bm25s.BM25:
    """Fit a bm25s index over the chunk texts.

    We hand bm25s the token numbers and the vocabulary rather than the words, so
    the numbering is ours and stays stable. Everything else is left at the bm25s
    defaults (lucene scoring, k1=1.5, b=0.75): tuning them before we can see
    per-tier scores would be guessing.
    """
    token_rows = tokenize_chunks(chunks)
    vocab = build_vocab(token_rows)

    numbered_rows: list[list[int]] = []
    for row in token_rows:
        numbers: list[int] = []
        for token in row:
            numbers.append(vocab[token])
        numbered_rows.append(numbers)

    index = bm25s.BM25()
    index.index((numbered_rows, vocab), show_progress=False)
    return index


def save_index(index: bm25s.BM25, chunks: Sequence[Chunk], index_dir: str | Path) -> None:
    """Write the index plus the chunk-id order into `index_dir`."""
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    index.save(str(index_dir), show_progress=False)

    chunk_ids: list[str] = []
    for chunk in chunks:
        chunk_ids.append(chunk.chunk_id)

    path = index_dir / CHUNK_IDS_FILENAME
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(chunk_ids, ensure_ascii=False))
        handle.write("\n")


def load_index(index_dir: str | Path) -> tuple[bm25s.BM25, list[str]]:
    """Read back what `save_index` wrote, or say plainly what is missing."""
    index_dir = Path(index_dir)
    if not (index_dir / PARAMS_FILENAME).exists():
        raise FileNotFoundError(f"No bm25 index at {index_dir}. Run `finrag ingest` first.")

    index = bm25s.BM25.load(str(index_dir), show_progress=False)
    with open(index_dir / CHUNK_IDS_FILENAME, encoding="utf-8") as handle:
        chunk_ids = json.load(handle)
    return index, list(chunk_ids)


def build_and_save_index(chunks: Sequence[Chunk], index_dir: str | Path) -> int:
    """Build the index for these chunks, save it, and report how many rows."""
    index = build_index(chunks)
    save_index(index, chunks, index_dir)
    return len(chunks)


class BM25Retriever:
    """BM25 search over chunk text, role masked before any ranking.

    Pass `index_dir` to use the artifact ingest wrote. Leave it out and the index
    is built in memory, which is how the tests compare the two.
    """

    name = "bm25"

    def __init__(self, chunks: Sequence[Chunk], index_dir: str | Path | None = None) -> None:
        self.chunks = list(chunks)

        # bm25s cannot index an empty corpus (average document length would be a
        # division by zero), so we skip the index and search returns nothing.
        if not self.chunks:
            self.index = None
        elif index_dir is None:
            self.index = build_index(self.chunks)
        else:
            self.index, stored_ids = load_index(index_dir)
            _check_index_matches_chunks(self.chunks, stored_ids, index_dir)

        # Kept for the debug field: which query words a chunk actually contains.
        # Same tokenizer again, so this agrees with what was indexed.
        self._terms_by_id: dict[str, set[str]] = {}
        for chunk in self.chunks:
            self._terms_by_id[chunk.chunk_id] = set(tokenize(chunk.text))

    def search(self, query: str, role: str, k: int = 10) -> list[Hit]:
        """Best `k` chunks this role may see, ranked by BM25 score."""
        if self.index is None:
            return []

        # The mask comes FIRST, before any scoring or cutting.
        allowed = visible_ids(self.chunks, role)
        query_terms = tokenize(query)
        if not query_terms:
            return []

        scores = self.index.get_scores(query_terms, weight_mask=self._weight_mask(allowed))
        distinct_terms = set(query_terms)

        scored: list[tuple[float, str, Chunk]] = []
        for row, chunk in enumerate(self.chunks):
            # Second lock on the same door, deliberately. If the mask above were
            # ever passed wrongly, this loop still refuses denied chunks.
            if chunk.chunk_id not in allowed:
                continue
            score = float(scores[row])
            if score > 0.0:
                scored.append((score, chunk.chunk_id, chunk))

        # Higher score first; chunk_id breaks ties so two runs agree.
        scored.sort(key=lambda item: (-item[0], item[1]))

        hits: list[Hit] = []
        rank = 1
        for score, _chunk_id, chunk in scored[:k]:
            hits.append(self._to_hit(chunk, rank, score, distinct_terms))
            rank += 1
        return hits

    def _weight_mask(self, allowed: frozenset[str]) -> np.ndarray:
        """1.0 for a row this role may see, 0.0 for one it may not.

        bm25s multiplies its score vector by this array, so a denied chunk leaves
        scoring already zeroed and never reaches the ranking. It is also why we
        use `get_scores` and sort ourselves instead of calling `retrieve`, which
        would do its own top-k: the mask and the tie-break stay plain Python.
        """
        mask = np.zeros(len(self.chunks), dtype="float32")
        for row, chunk in enumerate(self.chunks):
            if chunk.chunk_id in allowed:
                mask[row] = 1.0
        return mask

    def _to_hit(self, chunk: Chunk, rank: int, score: float, query_terms: set[str]) -> Hit:
        """Copy the chunk attributes the policy replay and the metrics need."""
        chunk_terms = self._terms_by_id[chunk.chunk_id]
        matched: list[str] = []
        for term in sorted(query_terms):
            if term in chunk_terms:
                matched.append(term)

        return Hit(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            section_slug=chunk.section_slug,
            rank=rank,
            score=score,
            retriever=self.name,
            owner_dept=chunk.owner_dept,
            doc_type=chunk.doc_type,
            sensitivity=chunk.sensitivity,
            allowed_roles=chunk.allowed_roles,
            text=chunk.text,
            debug={
                "bm25_score": score,
                "matched_terms": matched,
                "n_query_terms": len(query_terms),
            },
        )


def _check_index_matches_chunks(
    chunks: Sequence[Chunk],
    stored_ids: Sequence[str],
    index_dir: str | Path,
) -> None:
    """Refuse an index built from different chunks, or from a different order.

    Scores come back as a plain array of row numbers, so a mismatch would raise
    nothing: it would attach one chunk's score to another chunk, and the eval
    numbers would still look plausible. That is worse than a crash.
    """
    current_ids: list[str] = []
    for chunk in chunks:
        current_ids.append(chunk.chunk_id)

    if current_ids != list(stored_ids):
        raise ValueError(
            f"The bm25 index at {index_dir} was built from different chunks "
            f"({len(stored_ids)} rows) than the ones loaded ({len(current_ids)} rows), "
            "or from a different order. Run `finrag ingest` to rebuild both together."
        )
