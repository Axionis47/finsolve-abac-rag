"""Dense search: cosine similarity over embeddings built once at ingest.

Where this sits: ingest writes `artifacts/chunks.jsonl`, then the BM25 index, then
calls `build_and_save_matrix` here to write `artifacts/dense.npz`. At query time
the retriever loads that file and embeds only the query. Nothing is re-embedded
per query, and a query already seen comes off the embedding cache, so a repeated
eval run makes no API calls at all.

Why dense at all: BM25 scores words, so a question sharing no word with its answer
scores zero however well it means the same thing. R1's T2 paraphrase tier is at a
true 0.000 section hit rate for exactly that reason. Embeddings put "buyer
attrition in the festive stretch" near "Q4 customer churn" because the vectors are
close, not because a token matched.

THE INVARIANT: `search` computes the role's visible chunk ids before it touches a
vector, scores only those rows, and checks the mask again inside the loop. A
denied chunk never occupies a rank slot. Same shape as bm25.py, deliberately.

Two differences from BM25 worth knowing. Vectors are L2-normalised by the
embedder, so a dot product IS the cosine and no normalising happens here. And
cosine has no natural zero, so there is no "score > 0" filter: every chunk the
role may see is a candidate and `k` does all the cutting.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from finrag.models import Chunk, Hit
from finrag.retrieval.base import visible_ids
from finrag.retrieval.embed import Embedder


def build_matrix(chunks: Sequence[Chunk], embedder: Embedder) -> np.ndarray:
    """Embed every chunk's text, in the order the chunks were given.

    Row N of the result is chunk N. That correspondence is the whole contract of
    this file, which is why `save_matrix` writes the chunk ids beside it.
    """
    texts: list[str] = []
    for chunk in chunks:
        texts.append(chunk.text)
    return embedder.embed_texts(texts)


def save_matrix(
    vectors: np.ndarray,
    chunks: Sequence[Chunk],
    dense_path: str | Path,
    model: str,
) -> None:
    """Write the matrix, the chunk ids in row order, and which model made it."""
    dense_path = Path(dense_path)
    dense_path.parent.mkdir(parents=True, exist_ok=True)

    chunk_ids: list[str] = []
    for chunk in chunks:
        chunk_ids.append(chunk.chunk_id)

    np.savez(
        dense_path,
        vectors=np.array(vectors, dtype=np.float32),
        chunk_ids=np.array(chunk_ids),
        model=np.array(model),
    )


def load_matrix(dense_path: str | Path) -> tuple[np.ndarray, list[str], str]:
    """Read back what `save_matrix` wrote, or say plainly what is missing."""
    dense_path = Path(dense_path)
    if not dense_path.exists():
        raise FileNotFoundError(
            f"No dense matrix at {dense_path}. Run `finrag ingest` first."
        )

    with np.load(dense_path, allow_pickle=False) as stored:
        vectors = np.array(stored["vectors"], dtype=np.float32)
        chunk_ids: list[str] = []
        for chunk_id in stored["chunk_ids"]:
            chunk_ids.append(str(chunk_id))
        model = str(stored["model"])

    # The file's own arithmetic, checked by the one function that reads it. Every
    # other guard here compares the file against the world; this one is internal,
    # and without it a short matrix is an IndexError at query time on whichever
    # row happens to be missing, which reads as a bug in the search loop.
    if vectors.shape[0] != len(chunk_ids):
        raise ValueError(
            f"The dense matrix at {dense_path} is inconsistent: {vectors.shape[0]} "
            f"vectors against {len(chunk_ids)} chunk ids. Run `finrag ingest` to "
            "rebuild it."
        )
    return vectors, chunk_ids, model


def build_and_save_matrix(
    chunks: Sequence[Chunk],
    embedder: Embedder,
    dense_path: str | Path,
) -> int:
    """Embed these chunks, save the matrix, and report how many rows.

    The model name recorded is the embedding space, not the wrapper: a matrix and
    a query vector are only comparable when they came from the same space.
    """
    vectors = build_matrix(chunks, embedder)
    save_matrix(vectors, chunks, dense_path, model=embedder.model)
    return len(chunks)


class DenseRetriever:
    """Cosine search over chunk embeddings, role masked before any ranking.

    Pass `dense_path` to use the artifact ingest wrote; leave it out and the
    chunks are embedded in memory, which is how the tests compare the two.
    """

    name = "dense"

    def __init__(
        self,
        chunks: Sequence[Chunk],
        embedder: Embedder,
        dense_path: str | Path | None = None,
    ) -> None:
        self.chunks = list(chunks)
        self.embedder = embedder

        # The row-order guard is checked first, before the empty-corpus shortcut.
        # The other way round, an empty chunk list would accept any saved matrix
        # without a word, which is the one failure this guard exists to prevent.
        if dense_path is not None:
            self.vectors, stored_ids, stored_model = load_matrix(dense_path)
            _check_model_matches_embedder(stored_model, embedder, dense_path)
            _check_matrix_matches_chunks(self.chunks, stored_ids, dense_path)
        elif not self.chunks:
            # Nothing to embed, so there is no matrix and search returns nothing.
            self.vectors = None
        else:
            self.vectors = build_matrix(self.chunks, embedder)

    def search(self, query: str, role: str, k: int = 10) -> list[Hit]:
        """Best `k` chunks this role may see, ranked by cosine similarity."""
        if self.vectors is None:
            return []

        # The mask comes FIRST, before the query is even embedded.
        allowed = visible_ids(self.chunks, role)
        if not query.strip():
            return []

        allowed_rows: list[int] = []
        for row, chunk in enumerate(self.chunks):
            if chunk.chunk_id in allowed:
                allowed_rows.append(row)
        if not allowed_rows:
            return []

        query_vector = self.embedder.embed_texts([query])[0]

        scored: list[tuple[float, str, Chunk]] = []
        for row in allowed_rows:
            chunk = self.chunks[row]
            # Second lock on the same door, deliberately, as in bm25.py. If the
            # row list above were ever built wrongly, this still refuses.
            if chunk.chunk_id not in allowed:
                continue
            # Both sides are unit length, so the dot product is the cosine.
            score = float(np.dot(self.vectors[row], query_vector))
            scored.append((score, chunk.chunk_id, chunk))

        # Higher score first; chunk_id breaks ties so two runs agree.
        scored.sort(key=lambda item: (-item[0], item[1]))

        hits: list[Hit] = []
        rank = 1
        for score, _chunk_id, chunk in scored[:k]:
            hits.append(self._to_hit(chunk, rank, score))
            rank += 1
        return hits

    def _to_hit(self, chunk: Chunk, rank: int, score: float) -> Hit:
        """Copy the chunk attributes the policy replay and the metrics need."""
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
            debug={"cosine": score, "embedder": self.embedder.name},
        )

    def configuration(self) -> dict[str, object]:
        """Embedding space recorded beside evaluation results."""
        return {
            "name": self.name,
            "embedder": self.embedder.name,
            "embedding_model": self.embedder.model,
            "score": "cosine similarity",
        }


def _check_model_matches_embedder(
    stored_model: str,
    embedder: Embedder,
    dense_path: str | Path,
) -> None:
    """Refuse a matrix from one embedding space queried by another.

    The failure this stops is the quiet one. Vectors from two models are not
    interchangeable, but a dot product between them still returns a number: a
    Vertex-built matrix searched with the fake embedder answers every query with
    cosines around 0.05 and a ranking that is pure noise, and nothing raises.
    `embed_cache.py` handles the same hazard by starting empty, which it can do
    because a cache can be refilled. A matrix cannot, so this raises instead.
    """
    if stored_model == embedder.model:
        return
    raise ValueError(
        f"The dense matrix at {dense_path} was built with '{stored_model}' but the "
        f"query embedder is '{embedder.model}'. Cosines between two embedding "
        "spaces are meaningless. Run `finrag ingest` with the embedder you mean to "
        "search with, or set FINRAG_EMBEDDER to match the matrix."
    )


def _check_matrix_matches_chunks(
    chunks: Sequence[Chunk],
    stored_ids: Sequence[str],
    dense_path: str | Path,
) -> None:
    """Refuse a matrix built from different chunks, or from a different order.

    The npz knows only row numbers, so a mismatch would raise nothing: it would
    score one chunk with another chunk's vector, and the eval numbers would still
    look plausible. That is worse than a crash. Same guard as the BM25 index.
    """
    current_ids: list[str] = []
    for chunk in chunks:
        current_ids.append(chunk.chunk_id)

    if current_ids != list(stored_ids):
        raise ValueError(
            f"The dense matrix at {dense_path} was built from different chunks, or in "
            f"a different order ({len(stored_ids)} rows saved, {len(current_ids)} "
            "loaded). Run `finrag ingest` to rebuild both together."
        )
