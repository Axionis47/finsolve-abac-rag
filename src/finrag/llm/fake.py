"""Deterministic offline stand-ins for the Vertex generation and embedding backends.

No network, no model files, no randomness. The same input always gives the same
output, in this process and the next, so tests and retrieval evals can run
anywhere.
"""

from __future__ import annotations

import hashlib

import numpy as np

FAKE_LLM_PREFIX = "FAKE_LLM"
_DEFAULT_DIM = 64


def _digest(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()


class FakeLLM:
    """Returns a canned response for exact prompt matches, else a stub.

    `responses` maps a full prompt string to its reply; anything else gets a
    stub that encodes a hash of the prompt, so tests can assert stability
    without pinning long strings. Calls are recorded for inspection.
    """

    name = "fake"

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        self.responses = dict(responses or {})
        self.calls: list[tuple[str, str | None]] = []

    def generate(self, prompt: str, system: str | None = None) -> str:
        self.calls.append((prompt, system))
        canned = self.responses.get(prompt)
        if canned is not None:
            return canned
        return f"{FAKE_LLM_PREFIX}::{_digest(prompt)}::{prompt.strip()[:80]}"


class ScriptedLLM:
    """Returns prepared replies in order, ignoring the prompt.

    `FakeLLM` keys on the exact prompt, which works when a caller builds one
    prompt it can reproduce in a test. A ReAct loop builds a different prompt
    every hop, each containing the last hop's observations, so pinning them by
    text would mean pasting the transcript into the test. This backend answers
    the first call with the first reply and so on, which is how a scripted agent
    run is written: search, search, finish.

    Once the script runs out the last reply repeats, so a loop that asks more
    times than expected keeps going rather than raising. A test that wants the
    loop to break on garbage puts the garbage in the script.
    """

    name = "scripted"

    def __init__(self, replies: list[str]) -> None:
        if not replies:
            raise ValueError("a scripted backend needs at least one reply")
        self.replies = list(replies)
        self.calls: list[tuple[str, str | None]] = []

    def generate(self, prompt: str, system: str | None = None) -> str:
        self.calls.append((prompt, system))
        index = min(len(self.calls) - 1, len(self.replies) - 1)
        return self.replies[index]


class FakeEmbeddings:
    """Hashed character-trigram vectors, L2-normalised.

    Each trigram of the lower-cased text is hashed to one of `dim` buckets with
    a sign, and the counts are normalised. Texts that share substrings end up
    with related vectors, which is enough for cosine search to behave sensibly
    in tests, while identical text always gives an identical vector. Empty text
    gives the zero vector.
    """

    name = "fake-embeddings"

    def __init__(self, dim: int = _DEFAULT_DIM) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            out[i] = self._embed_one(text)
        return out

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float64)
        padded = f" {text.lower().strip()} "
        for start in range(len(padded) - 2):
            trigram = padded[start : start + 3]
            h = int.from_bytes(
                hashlib.blake2b(trigram.encode("utf-8"), digest_size=8).digest(),
                "big",
            )
            bucket = h % self._dim
            sign = 1.0 if (h >> 32) & 1 else -1.0
            vec[bucket] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec.astype(np.float32)
