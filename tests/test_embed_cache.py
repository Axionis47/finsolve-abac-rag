"""The embedding cache: a hit must not reach the backend, and must not lie.

The cache is what makes a re-run free, so the test that matters most counts calls
on the backend rather than comparing vectors: a cache that returns the right
answer by asking again has not done its job.

The other load-bearing behaviour is the model guard. Vectors from two models are
not interchangeable, and cosine between them is meaningless but never raises, so a
cache read back under the wrong model name would produce plausible wrong rankings.
It has to come back empty instead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from finrag.retrieval.embed import EMBED_DIM, FakeEmbedder
from finrag.retrieval.embed_cache import CachingEmbedder, EmbeddingCache, text_key


def caching_embedder(tmp_path: Path, model: str = "fake") -> CachingEmbedder:
    """A fake backend behind a cache in a throwaway folder."""
    cache = EmbeddingCache.load(tmp_path / "embed_cache.npz", model, EMBED_DIM)
    return CachingEmbedder(FakeEmbedder(), cache)


# --- the point of the cache --------------------------------------------------


def test_a_second_call_for_the_same_text_never_reaches_the_backend(tmp_path: Path) -> None:
    """The whole reason this class exists, asserted on calls not on values."""
    embedder = caching_embedder(tmp_path)

    first = embedder.embed_texts(["quarterly revenue"])
    second = embedder.embed_texts(["quarterly revenue"])

    assert np.array_equal(first, second)
    assert embedder.embedder.calls == [["quarterly revenue"]]


def test_only_the_texts_it_has_not_seen_are_sent_on(tmp_path: Path) -> None:
    embedder = caching_embedder(tmp_path)
    embedder.embed_texts(["seen"])

    embedder.embed_texts(["seen", "new"])

    assert embedder.embedder.calls == [["seen"], ["new"]]


def test_a_text_repeated_in_one_call_is_embedded_once(tmp_path: Path) -> None:
    embedder = caching_embedder(tmp_path)

    vectors = embedder.embed_texts(["same", "same"])

    assert embedder.embedder.calls == [["same"]]
    assert np.array_equal(vectors[0], vectors[1])


def test_rows_come_back_in_the_order_they_were_asked_for(tmp_path: Path) -> None:
    """A cache that reorders would attach the wrong vector to the wrong chunk."""
    embedder = caching_embedder(tmp_path)
    embedder.embed_texts(["b"])

    vectors = embedder.embed_texts(["a", "b", "c"])

    direct = FakeEmbedder().embed_texts(["a", "b", "c"])
    assert np.allclose(vectors, direct)


def test_a_call_with_no_texts_asks_the_backend_nothing(tmp_path: Path) -> None:
    embedder = caching_embedder(tmp_path)

    vectors = embedder.embed_texts([])

    assert vectors.shape == (0, EMBED_DIM)
    assert embedder.embedder.calls == []


# --- across processes: the file, not the dictionary ---------------------------


def test_a_fresh_embedder_reads_the_file_the_first_one_wrote(tmp_path: Path) -> None:
    """This is the case that makes a repeated eval run offline."""
    first = caching_embedder(tmp_path)
    written = first.embed_texts(["quarterly revenue"])

    second = caching_embedder(tmp_path)
    read_back = second.embed_texts(["quarterly revenue"])

    assert np.array_equal(read_back, written)
    assert second.embedder.calls == []


def test_a_cache_written_by_another_model_is_not_read_back(tmp_path: Path) -> None:
    """Cosine across two embedding spaces is nonsense that never raises."""
    first = caching_embedder(tmp_path, model="fake")
    first.embed_texts(["quarterly revenue"])

    second = caching_embedder(tmp_path, model="text-embedding-005")
    second.embed_texts(["quarterly revenue"])

    assert second.embedder.calls == [["quarterly revenue"]]


def test_a_cache_saved_at_another_width_is_not_read_back(tmp_path: Path) -> None:
    path = tmp_path / "embed_cache.npz"
    narrow = EmbeddingCache(path, "fake", 8)
    narrow.put("text", np.ones(8, dtype=np.float32))
    narrow.save()

    wide = EmbeddingCache.load(path, "fake", EMBED_DIM)

    assert len(wide) == 0


def test_a_missing_file_loads_as_an_empty_cache(tmp_path: Path) -> None:
    cache = EmbeddingCache.load(tmp_path / "nothing-here.npz", "fake", EMBED_DIM)

    assert len(cache) == 0
    assert cache.get("anything") is None


def test_saving_creates_the_folder_if_it_is_not_there(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "deep" / "embed_cache.npz", "fake", 4)
    cache.put("text", np.ones(4, dtype=np.float32))

    cache.save()

    assert cache.path.exists()


def test_saving_an_unchanged_cache_writes_nothing(tmp_path: Path) -> None:
    """A save per query would rewrite the whole file for no reason."""
    cache = EmbeddingCache(tmp_path / "embed_cache.npz", "fake", 4)

    cache.save()

    assert not cache.path.exists()


def test_two_saves_of_the_same_vectors_give_identical_bytes(tmp_path: Path) -> None:
    """Keys are sorted, so the artifact does not reshuffle between runs."""
    first_path = tmp_path / "first.npz"
    second_path = tmp_path / "second.npz"

    for path in [first_path, second_path]:
        cache = EmbeddingCache(path, "fake", EMBED_DIM)
        # Inserted in opposite orders, so only sorting can make the bytes match.
        texts = ["b", "a", "c"] if path == first_path else ["c", "a", "b"]
        for text in texts:
            cache.put(text, FakeEmbedder().embed_texts([text])[0])
        cache.save()

    assert first_path.read_bytes() == second_path.read_bytes()


def test_the_cache_names_itself_a_wrapper_but_keeps_the_embedding_space(
    tmp_path: Path,
) -> None:
    """`name` says who was asked; `model` says which space the vectors live in.

    dense.npz records `model`, and two matrices are only comparable when that
    matches, so wrapping must not rename the space.
    """
    embedder = caching_embedder(tmp_path)

    assert embedder.name == "cached-fake"
    assert embedder.model == "fake"


def test_the_key_is_a_sha256_of_the_text(tmp_path: Path) -> None:
    """Stated plainly because the key is what makes a changed chunk a cache miss."""
    assert len(text_key("revenue")) == 64
    assert text_key("revenue") != text_key("revenue ")
