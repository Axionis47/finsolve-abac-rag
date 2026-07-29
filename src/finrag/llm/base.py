"""Backend contracts for text generation and embeddings.

Both protocols exist so tests and retrieval evals run offline: the fakes in
`fake.py` satisfy them without credentials or model downloads.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class LLMBackend(Protocol):
    """Single-turn text generation."""

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Return the model's text response to `prompt`."""
        ...


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Batch text embedding."""

    @property
    def dim(self) -> int:
        """Length of a single embedding vector."""
        ...

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return a (len(texts), dim) float32 array of L2-normalised vectors."""
        ...
