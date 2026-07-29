"""The embedding backends: determinism, normalisation, batching, backend choice.

Everything here is offline. `FakeEmbedder` is the only backend exercised end to
end; `VertexEmbedder` is exercised through a hand-written stand-in client, so the
batching and the retry are tested without a network call or a credential.

What is being pinned:

  * the fake really is deterministic, across calls and across processes, because
    every offline test and every cached run depends on that.
  * rows come back unit length, because dense.py treats a dot product as a cosine
    and would silently rank on the wrong numbers otherwise.
  * batching respects both caps and keeps order, since order is the only thing
    tying a returned vector back to its text.
  * a transient failure is retried and a permanent one is raised, not swallowed.
  * `build_embedder` refuses to pretend: no project id is an error, not a quiet
    fall back to fake vectors published as real ones.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from finrag.config import Settings
from finrag.retrieval.embed import (
    EMBED_DIM,
    MAX_BATCH_CHARS,
    MAX_BATCH_TEXTS,
    FakeEmbedder,
    VertexEmbedder,
    build_embedder,
    hash_vector,
    l2_normalize,
    split_into_batches,
)
from finrag.retrieval.embed_cache import CachingEmbedder


class StubClient:
    """The two attribute hops google-genai needs, over a canned reply.

    `fail_times` makes the first N calls raise, which is how the retry is tested
    without waiting on a real rate limit.
    """

    def __init__(self, dim: int = 4, fail_times: int = 0, drop_one: bool = False) -> None:
        self.dim = dim
        self.fail_times = fail_times
        self.drop_one = drop_one
        self.batches: list[list[str]] = []
        # google-genai is reached as `client.models.embed_content`, so the stub
        # stands in for both hops at once.
        self.models = self

    def embed_content(self, model: str, contents: list[str]) -> StubResponse:
        self.batches.append(list(contents))
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("transient")

        wanted = len(contents) - 1 if self.drop_one else len(contents)
        embeddings = []
        for position in range(wanted):
            embeddings.append(StubEmbedding([float(position + 1)] * self.dim))
        return StubResponse(embeddings)


class StubEmbedding:
    def __init__(self, values: list[float]) -> None:
        self.values = values


class StubResponse:
    def __init__(self, embeddings: list[StubEmbedding]) -> None:
        self.embeddings = embeddings


def stub_vertex(
    dim: int = 4, fail_times: int = 0, drop_one: bool = False
) -> tuple[VertexEmbedder, StubClient]:
    client = StubClient(dim=dim, fail_times=fail_times, drop_one=drop_one)
    embedder = VertexEmbedder(
        project="a-project", location="us-central1", model="m", dim=dim, client=client
    )
    return embedder, client


# --- the fake is deterministic and normalised --------------------------------


def test_the_same_text_gives_the_same_vector_every_time() -> None:
    embedder = FakeEmbedder()

    first = embedder.embed_texts(["quarterly revenue"])
    second = embedder.embed_texts(["quarterly revenue"])

    assert np.array_equal(first, second)


def test_the_same_text_gives_the_same_vector_in_a_fresh_process(repo_root: Path) -> None:
    """sha256, not `hash`, so no hash seed can move a vector between runs.

    An in-process check would pass either way and tell us nothing: Python's own
    string hashing is stable within one interpreter and randomised between them.
    """
    script = (
        "from finrag.retrieval.embed import FakeEmbedder\n"
        "vector = FakeEmbedder(dim=8).embed_texts(['quarterly revenue'])[0]\n"
        "print(','.join(f'{value:.6f}' for value in vector))\n"
    )
    printed: list[str] = []
    for seed in ["0", "1"]:
        finished = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env={"PYTHONHASHSEED": seed, "PYTHONPATH": str(repo_root / "src")},
        )
        printed.append(finished.stdout.strip())

    assert printed[0] == printed[1]


def test_different_texts_get_different_vectors() -> None:
    vectors = FakeEmbedder().embed_texts(["revenue", "headcount"])
    assert not np.array_equal(vectors[0], vectors[1])


def test_every_row_comes_back_unit_length() -> None:
    """dense.py reads a dot product as a cosine, so this is load-bearing."""
    vectors = FakeEmbedder().embed_texts(["revenue", "headcount", ""])

    for row in range(vectors.shape[0]):
        assert float(np.linalg.norm(vectors[row])) == pytest.approx(1.0, abs=1e-5)


def test_the_fake_is_the_same_width_as_the_real_model() -> None:
    """A fake artifact and a real one must be the same shape, or npz swaps break."""
    assert FakeEmbedder().dim == EMBED_DIM
    assert FakeEmbedder().embed_texts(["a"]).shape == (1, EMBED_DIM)


def test_the_fake_returns_float32_so_the_npz_stays_small() -> None:
    assert FakeEmbedder().embed_texts(["a"]).dtype == np.float32


def test_a_zero_vector_is_left_alone_rather_than_divided_by_zero() -> None:
    normalised = l2_normalize(np.zeros((1, 4), dtype=np.float32))
    assert float(np.linalg.norm(normalised[0])) == 0.0


def test_a_hash_vector_is_exactly_as_wide_as_asked_for() -> None:
    """dim is not a multiple of 32, so the last sha256 block must be trimmed."""
    assert hash_vector("revenue", 100).shape == (100,)


def test_a_zero_width_fake_is_refused() -> None:
    with pytest.raises(ValueError):
        FakeEmbedder(dim=0)


def test_the_fake_records_the_batches_it_was_asked_for() -> None:
    """The cache tests count these, so the count has to be trustworthy."""
    embedder = FakeEmbedder()

    embedder.embed_texts(["a", "b"])
    embedder.embed_texts(["c"])

    assert embedder.calls == [["a", "b"], ["c"]]


# --- batching ----------------------------------------------------------------


def test_a_short_list_goes_in_one_batch() -> None:
    assert split_into_batches(["a", "b", "c"]) == [["a", "b", "c"]]


def test_an_empty_list_makes_no_batches() -> None:
    assert split_into_batches([]) == []


def test_the_text_count_cap_splits_the_list() -> None:
    texts = [f"text {number}" for number in range(MAX_BATCH_TEXTS + 5)]

    batches = split_into_batches(texts)

    assert len(batches) == 2
    assert len(batches[0]) == MAX_BATCH_TEXTS
    assert len(batches[1]) == 5


def test_the_character_cap_splits_the_list_before_the_count_cap_does() -> None:
    """Two texts of a third of the cap each fit; the third opens a new batch."""
    long_text = "x" * (MAX_BATCH_CHARS // 3 + 1)
    batches = split_into_batches([long_text] * 3)

    assert [len(batch) for batch in batches] == [2, 1]


def test_a_single_text_over_the_character_cap_goes_alone_rather_than_being_cut() -> None:
    """Truncating would embed something the corpus does not say."""
    huge = "x" * (MAX_BATCH_CHARS + 10)
    batches = split_into_batches(["small", huge, "small"])

    assert batches == [["small"], [huge], ["small"]]


def test_batching_never_loses_or_reorders_a_text() -> None:
    texts = [f"text {number}" for number in range(250)]

    flattened: list[str] = []
    for batch in split_into_batches(texts):
        flattened.extend(batch)

    assert flattened == texts


# --- the Vertex backend, through a stand-in client ---------------------------


def test_vertex_sends_every_text_and_returns_a_row_for_each() -> None:
    embedder, client = stub_vertex()

    vectors = embedder.embed_texts(["a", "b", "c"])

    assert client.batches == [["a", "b", "c"]]
    assert vectors.shape == (3, 4)


def test_vertex_normalises_what_the_api_returns() -> None:
    """The API is not promised to return unit vectors, so we do not assume it."""
    embedder, _client = stub_vertex()

    vectors = embedder.embed_texts(["a", "b"])

    for row in range(vectors.shape[0]):
        assert float(np.linalg.norm(vectors[row])) == pytest.approx(1.0, abs=1e-5)


def test_vertex_sends_one_call_per_batch() -> None:
    embedder, client = stub_vertex()

    embedder.embed_texts([f"text {number}" for number in range(MAX_BATCH_TEXTS + 1)])

    assert [len(batch) for batch in client.batches] == [MAX_BATCH_TEXTS, 1]


def test_vertex_embeds_nothing_without_calling_the_api() -> None:
    embedder, client = stub_vertex()

    vectors = embedder.embed_texts([])

    assert vectors.shape == (0, 4)
    assert client.batches == []


def test_a_transient_failure_is_retried_and_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two failures then a success. Sleeping is patched out so the test is quick."""
    monkeypatch.setattr("finrag.retrieval.embed.time.sleep", lambda _seconds: None)
    embedder, client = stub_vertex(fail_times=2)

    vectors = embedder.embed_texts(["a"])

    assert vectors.shape == (1, 4)
    assert len(client.batches) == 3


def test_a_failure_that_never_clears_is_raised_with_the_batch_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Silently returning zeros here would poison the whole matrix."""
    monkeypatch.setattr("finrag.retrieval.embed.time.sleep", lambda _seconds: None)
    embedder, client = stub_vertex(fail_times=99)

    with pytest.raises(RuntimeError) as caught:
        embedder.embed_texts(["a", "b"])

    assert "2 texts" in str(caught.value)
    assert len(client.batches) == 3


def test_a_reply_with_fewer_vectors_than_texts_is_refused() -> None:
    """Row order is the only link back to the text, so a short reply is fatal."""
    embedder, _client = stub_vertex(drop_one=True)

    with pytest.raises(RuntimeError) as caught:
        embedder.embed_texts(["a", "b"])

    assert "1 vectors for 2 texts" in str(caught.value)


def test_a_vertex_embedder_without_a_project_is_refused() -> None:
    with pytest.raises(ValueError):
        VertexEmbedder(project="", location="us-central1", model="m")


# --- choosing a backend ------------------------------------------------------


def test_build_embedder_gives_the_fake_when_asked_for_it(
    repo_root: Path, tmp_path: Path
) -> None:
    settings = replace(
        Settings.from_env(root=repo_root), artifacts_dir=tmp_path, embedder="fake"
    )

    embedder = build_embedder(settings)

    assert isinstance(embedder, CachingEmbedder)
    assert isinstance(embedder.embedder, FakeEmbedder)
    assert embedder.dim == EMBED_DIM


def test_build_embedder_refuses_vertex_with_no_project_rather_than_faking_it(
    repo_root: Path, tmp_path: Path
) -> None:
    """Fake vectors published as real numbers would be the worst outcome here."""
    settings = replace(
        Settings.from_env(root=repo_root),
        artifacts_dir=tmp_path,
        embedder="vertex",
        vertex_project=None,
    )

    with pytest.raises(RuntimeError) as caught:
        build_embedder(settings)

    assert "VERTEX_PROJECT" in str(caught.value)
    assert "FINRAG_EMBEDDER=fake" in str(caught.value)


def test_build_embedder_names_the_cache_after_the_real_model(
    repo_root: Path, tmp_path: Path
) -> None:
    """A cache written by one model must not be read back for another."""
    settings = replace(
        Settings.from_env(root=repo_root),
        artifacts_dir=tmp_path,
        embedder="vertex",
        vertex_project="a-project",
        vertex_embed_model="text-embedding-005",
    )

    embedder = build_embedder(settings)

    assert embedder.cache.model == "text-embedding-005"
    assert embedder.cache.path == tmp_path / "embed_cache.npz"
