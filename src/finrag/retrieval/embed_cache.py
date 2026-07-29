"""The embedding cache: pay for a text once, ever.

Why it exists. Ingest embeds every chunk and dense search embeds every query, and
both cost money and time on a real API. Keyed by sha256 of the exact text, a
second ingest over an unchanged corpus makes zero calls, and a repeated eval run
is fully offline because the 50 gold questions are already on disk.

Why the model name is stored with it. Vectors from two different models are not
interchangeable: cosine between them is meaningless but never fails, so a mixed
cache would produce plausible wrong rankings rather than an error. Loading a
cache written by another model, or at another width, therefore starts empty.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np


def text_key(text: str) -> str:
    """The cache key for one text: hex sha256 of its exact bytes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """text -> vector on disk, keyed by sha256(text).

    Stored as one npz holding a key column and a matrix, which is far smaller
    than a JSON list of floats and loads in a single read.
    """

    def __init__(self, path: str | Path, model: str, dim: int) -> None:
        self.path = Path(path)
        self.model = model
        self.dim = dim
        self._vectors: dict[str, np.ndarray] = {}
        # Saving rewrites the whole file, so we only do it when something changed.
        self._dirty = False

    @classmethod
    def load(cls, path: str | Path, model: str, dim: int) -> EmbeddingCache:
        """Read the cache, or start an empty one if it is missing or foreign."""
        cache = cls(path, model, dim)
        if not cache.path.exists():
            return cache

        with np.load(cache.path, allow_pickle=False) as stored:
            if str(stored["model"]) != model or int(stored["dim"]) != dim:
                return cache
            keys: list[str] = []
            for key in stored["keys"]:
                keys.append(str(key))
            vectors = np.array(stored["vectors"], dtype=np.float32)

        for row, key in enumerate(keys):
            cache._vectors[key] = vectors[row]
        return cache

    def get(self, text: str) -> np.ndarray | None:
        """The stored vector for this text, or None if we have never seen it."""
        return self._vectors.get(text_key(text))

    def put(self, text: str, vector: np.ndarray) -> None:
        self._vectors[text_key(text)] = np.array(vector, dtype=np.float32)
        self._dirty = True

    def save(self) -> None:
        """Write the cache out, keys sorted, so two runs produce one same file."""
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)

        keys = sorted(self._vectors)
        vectors = np.zeros((len(keys), self.dim), dtype=np.float32)
        for row, key in enumerate(keys):
            vectors[row] = self._vectors[key]

        np.savez(
            self.path,
            model=np.array(self.model),
            dim=np.array(self.dim),
            # A hex sha256 is always 64 characters, so a fixed-width string
            # column is safe and keeps the file readable by plain numpy.
            keys=np.array(keys, dtype="U64"),
            vectors=vectors,
        )
        self._dirty = False

    def __len__(self) -> int:
        return len(self._vectors)


class CachingEmbedder:
    """An embedder that only asks its backend for texts the cache lacks.

    Ingest and query time both go through one of these over the same file, which
    is what makes a repeated eval run free. Duplicate texts in one call are asked
    for once.
    """

    def __init__(self, embedder: Any, cache: EmbeddingCache) -> None:
        self.embedder = embedder
        self.cache = cache
        self.name = f"cached-{embedder.name}"
        # The cache changes who is asked, never which embedding space the vectors
        # live in, so `model` passes straight through.
        self.model = embedder.model
        self.dim = embedder.dim

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        missing: list[str] = []
        for text in texts:
            if self.cache.get(text) is None and text not in missing:
                missing.append(text)

        if missing:
            fresh = self.embedder.embed_texts(missing)
            for row, text in enumerate(missing):
                self.cache.put(text, fresh[row])
            self.cache.save()

        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            vector = self.cache.get(text)
            if vector is None:
                raise RuntimeError(f"no vector for a text we just embedded: {text[:60]!r}")
            out[row] = vector
        return out
