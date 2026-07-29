"""Text to vector: the embedding backends dense retrieval takes.

  * `Embedder`, the protocol. One method, `embed_texts`, returning float32 rows
    that are already L2-normalised, so every consumer can treat a dot product as
    a cosine and nobody has to remember to normalise.
  * `VertexEmbedder`, the real one. Vertex AI text embeddings through the
    google-genai SDK. Batched, with a short retry, because a whole ingest failing
    on one flaky call is not acceptable.
  * `FakeEmbedder`, the offline one. Deterministic vectors derived from a sha256
    of the text. No network, no model files, no randomness, so the tests and any
    offline run behave the same on every machine.

The on-disk cache that stops a re-run paying twice lives next door in
`embed_cache.py`. `build_embedder` here is what the pipeline and the CLI call:
it picks a backend from the settings and wraps it in that cache.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Protocol, runtime_checkable

import numpy as np

from finrag.retrieval.embed_cache import CachingEmbedder, EmbeddingCache

# text-embedding-005 returns 768 numbers per text. The fake uses the same width
# so a fake artifact and a real one are the same shape.
EMBED_DIM = 768

# Vertex allows far more per call than this; we stay well under, and cap the total
# characters too, because a batch of long chunks is rejected on size, not count.
MAX_BATCH_TEXTS = 100
MAX_BATCH_CHARS = 60_000

# One transient failure should not lose a whole ingest, and three tries two
# seconds apart covers a rate-limit blip without hiding a real outage.
RETRY_TRIES = 3
RETRY_SLEEP_SECONDS = 2.0


@runtime_checkable
class Embedder(Protocol):
    """Turns texts into unit-length vectors of a fixed width.

    `name` says which backend this is, `model` which embedding space its vectors
    live in. They differ once a backend is wrapped: the cache in front of Vertex
    is a different object but the same space, and it is the space that decides
    whether two saved matrices may be compared.
    """

    name: str
    model: str
    dim: int

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Return a (len(texts), dim) float32 array of L2-normalised rows."""
        ...


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale every row to length 1, leaving an all-zero row alone.

    Done once, here, so cosine similarity downstream is a plain dot product. A
    zero row cannot be scaled to length 1, so it stays zero and matches nothing.
    """
    out = np.array(matrix, dtype=np.float32)
    for row in range(out.shape[0]):
        length = float(np.linalg.norm(out[row]))
        if length > 0.0:
            out[row] = out[row] / length
    return out


def hash_vector(text: str, dim: int) -> np.ndarray:
    """A stable pseudo-random vector for this exact text.

    sha256 gives 32 bytes at a time, so we hash the text once per block of 32
    until there are `dim` numbers, and map each byte onto roughly [-1, 1).
    Nothing here holds state or reads a clock, so the same text gives the same
    vector in this process, the next one, and on another machine.
    """
    values: list[float] = []
    block = 0
    while len(values) < dim:
        digest = hashlib.sha256(f"{block}:{text}".encode("utf-8")).digest()
        for byte in digest:
            values.append((byte - 127.5) / 127.5)
        block += 1
    return np.array(values[:dim], dtype=np.float32)


class FakeEmbedder:
    """Hash vectors, offline. Records its calls so a test can count them.

    Two texts that differ at all get unrelated vectors; two identical texts get
    identical ones. That is all the structure the offline tests need: an exact
    tie is constructible, and a chunk whose text equals the query scores 1.0,
    which is the highest cosine there is.
    """

    name = "fake"
    model = "fake"

    def __init__(self, dim: int = EMBED_DIM) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        # Every batch this embedder was asked for, in order. The cache tests
        # assert on it: a cache hit must not reach the backend at all.
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            out[row] = hash_vector(text, self.dim)
        return l2_normalize(out)


class VertexEmbedder:
    """Vertex AI text embeddings through the google-genai SDK.

    The client is built on first use, not in `__init__`, so importing this module
    and constructing the object need no credentials. That matters because the
    registry imports it whether or not anyone asks for dense mode.
    """

    name = "vertex"

    def __init__(
        self,
        project: str,
        location: str,
        model: str,
        dim: int = EMBED_DIM,
        client: Any | None = None,
    ) -> None:
        if not project:
            raise ValueError("VertexEmbedder needs a project id")
        self.project = project
        self.location = location
        self.model = model
        self.dim = dim
        self._client = client

    def client(self) -> Any:
        """The google-genai client, built once and kept."""
        if self._client is None:
            from google import genai

            self._client = genai.Client(
                vertexai=True, project=self.project, location=self.location
            )
        return self._client

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        rows: list[list[float]] = []
        for batch in split_into_batches(texts):
            rows.extend(self._embed_one_batch(batch))

        # Row order is the only thing tying a vector back to its chunk, so a
        # short reply is refused rather than guessed at.
        if len(rows) != len(texts):
            raise RuntimeError(
                f"Vertex returned {len(rows)} vectors for {len(texts)} texts. "
                "Refusing to guess which text lost its vector."
            )
        return l2_normalize(np.array(rows, dtype=np.float32))

    def _embed_one_batch(self, batch: list[str]) -> list[list[float]]:
        """One API call, retried a few times on any error, then given up on."""
        last_error: Exception | None = None
        for attempt in range(RETRY_TRIES):
            try:
                response = self.client().models.embed_content(model=self.model, contents=batch)
                vectors: list[list[float]] = []
                for embedding in response.embeddings:
                    vectors.append(list(embedding.values))
                return vectors
            except Exception as error:  # noqa: BLE001 - any transport failure retries
                last_error = error
                if attempt + 1 < RETRY_TRIES:
                    time.sleep(RETRY_SLEEP_SECONDS)

        raise RuntimeError(
            f"Vertex embedding failed {RETRY_TRIES} times for a batch of "
            f"{len(batch)} texts: {last_error}"
        ) from last_error


def split_into_batches(texts: list[str]) -> list[list[str]]:
    """Group texts into API-sized batches, keeping their order.

    A single text longer than the character cap goes in a batch of its own rather
    than being cut, because a truncated chunk would be embedded as something the
    corpus does not say.
    """
    batches: list[list[str]] = []
    current: list[str] = []
    current_chars = 0

    for text in texts:
        too_many = len(current) >= MAX_BATCH_TEXTS
        too_long = bool(current) and current_chars + len(text) > MAX_BATCH_CHARS
        if too_many or too_long:
            batches.append(current)
            current = []
            current_chars = 0
        current.append(text)
        current_chars += len(text)

    if current:
        batches.append(current)
    return batches


def build_embedder(settings: Any) -> CachingEmbedder:
    """The embedder named by `settings.embedder`, wrapped in the on-disk cache.

    "fake" is the offline stand-in the tests use; anything else means Vertex. A
    missing project id is refused here, with the fix in the message, rather than
    quietly falling back to fake vectors and publishing them as real numbers.
    """
    if settings.embedder == "fake":
        inner: Embedder = FakeEmbedder(dim=EMBED_DIM)
        model_name = "fake"
    else:
        if not settings.vertex_project:
            raise RuntimeError(
                "Dense retrieval needs a Vertex project. Set VERTEX_PROJECT (or "
                "GCP_PROJECT) in .env, or set FINRAG_EMBEDDER=fake to run offline."
            )
        inner = VertexEmbedder(
            project=settings.vertex_project,
            location=settings.vertex_location,
            model=settings.vertex_embed_model,
            dim=EMBED_DIM,
        )
        model_name = settings.vertex_embed_model

    cache = EmbeddingCache.load(settings.embed_cache_path, model_name, EMBED_DIM)
    return CachingEmbedder(inner, cache)
