"""The dense mode: the role mask, the tie-break, and the saved matrix.

Everything here uses `FakeEmbedder`, so nothing touches a network. The fake gives
a text an unrelated vector unless the text is identical, which is exactly the
handle these tests need:

  * a chunk whose text IS the query scores a cosine of 1.0, the highest there is,
    so making that chunk the denied one proves the mask runs before the ranking
    rather than after it.
  * two chunks with the same text get the same vector, so an exact tie is
    constructible and the chunk_id tie-break can be pinned.

Four things are checked that a whole eval run would not tell us apart: mask before
cut, tie-break determinism, the saved matrix giving the same hits as a fresh one,
and the row-order guard refusing a reordered chunks file.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from finrag.config import Settings
from finrag.ingest.pipeline import read_chunks, run_ingest
from finrag.models import Chunk
from finrag.retrieval.dense import (
    DenseRetriever,
    build_and_save_matrix,
    build_matrix,
    load_matrix,
    save_matrix,
)
from finrag.retrieval.embed import EMBED_DIM, FakeEmbedder
from finrag.retrieval.embed_cache import CachingEmbedder, EmbeddingCache
from finrag.retrieval.registry import IndexBundle, available_retrievers, build_retriever

QUERY = "how did customer churn move over the festive quarter"


def _chunk(chunk_id: str, text: str, allowed_roles: tuple[str, ...]) -> Chunk:
    """A chunk with only the fields these tests read. Keeps the fixture readable."""
    return Chunk(
        chunk_id=chunk_id,
        doc_id=chunk_id.split("#")[0],
        section_path="Report > Section",
        section_slug=chunk_id.split("#")[1],
        text=text,
        owner_dept="finance" if chunk_id.startswith("finance/") else "marketing",
        doc_type="md",
        sensitivity="confidential" if chunk_id.startswith("finance/") else "internal",
        allowed_roles=allowed_roles,
    )


# The bait. Its text is the query word for word, so its cosine is exactly 1.0 and
# no visible chunk can beat it. Only c_level may read it.
SECRET = _chunk("finance/secret#churn", QUERY, ("c_level",))

# Two visible chunks with identical text, so identical vectors, so an exact tie.
TIED_A = _chunk("marketing/a#one", "seasonal campaign notes", ("marketing", "c_level"))
TIED_B = _chunk("marketing/b#two", "seasonal campaign notes", ("marketing", "c_level"))

# A third visible chunk, different text, so a different score.
OTHER = _chunk("marketing/c#three", "brand colours and typeface", ("marketing", "c_level"))

# The bait is listed first on purpose: source order must not save it either.
CORPUS = [SECRET, TIED_A, TIED_B, OTHER]


def ids_of(hits: list) -> list[str]:
    out: list[str] = []
    for hit in hits:
        out.append(hit.chunk_id)
    return out


def dense(chunks: list[Chunk], dense_path: Path | None = None) -> DenseRetriever:
    """A dense retriever over a fresh fake embedder. No network, no cache."""
    return DenseRetriever(chunks, FakeEmbedder(), dense_path=dense_path)


# --- the invariant: mask before cut -----------------------------------------


def test_the_best_scoring_chunk_is_hidden_when_the_role_may_not_see_it() -> None:
    """A denied chunk never occupies a rank slot, and never displaces one.

    SECRET's text is the query, so its cosine is 1.0 and nothing can outscore it.
    k is 10, wider than the whole visible pool of three, so no cut-off is doing
    the hiding. If the mask ran after scoring, SECRET would be at rank 1.
    """
    hits = dense(CORPUS).search(QUERY, "marketing", k=10)

    assert SECRET.chunk_id not in ids_of(hits)
    assert len(hits) == 3
    # Ranks stay 1..3 with no gap, so the denied chunk did not silently consume
    # a slot on the way past.
    assert [hit.rank for hit in hits] == [1, 2, 3]


def test_the_denied_chunk_really_was_the_top_scorer() -> None:
    """The control: the bait was findable, and the role is what stopped it."""
    hits = dense(CORPUS).search(QUERY, "c_level", k=10)

    assert hits[0].chunk_id == SECRET.chunk_id
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert hits[0].score > hits[1].score


def test_the_mask_holds_at_every_k_including_one_that_is_too_small() -> None:
    """A narrower cut must not be what protects the denied chunk."""
    for k in [1, 2, 3, 10, 50]:
        hits = dense(CORPUS).search(QUERY, "marketing", k=k)
        assert SECRET.chunk_id not in ids_of(hits)


def test_a_role_with_nothing_visible_gets_nothing() -> None:
    retriever = dense(CORPUS)
    assert retriever.search(QUERY, "engineering", k=10) == []
    assert retriever.search(QUERY, "not_a_role", k=10) == []


def test_every_hit_is_one_the_role_may_read() -> None:
    for hit in dense(CORPUS).search(QUERY, "marketing", k=10):
        assert "marketing" in hit.allowed_roles


# --- ranking behaviour -------------------------------------------------------


def test_ties_break_on_chunk_id() -> None:
    """Identical text means an identical vector, so the id has to decide."""
    hits = dense(CORPUS).search(QUERY, "marketing", k=10)

    tied = []
    for hit in hits:
        if hit.chunk_id in {TIED_A.chunk_id, TIED_B.chunk_id}:
            tied.append(hit)

    assert ids_of(tied) == [TIED_A.chunk_id, TIED_B.chunk_id]
    assert tied[0].score == tied[1].score


def test_ties_break_the_same_way_whatever_order_the_chunks_arrived_in() -> None:
    """Feeding the corpus backwards must not change the answer."""
    forwards = dense(CORPUS).search(QUERY, "marketing", k=10)
    backwards = dense(list(reversed(CORPUS))).search(QUERY, "marketing", k=10)

    assert ids_of(forwards) == ids_of(backwards)


def test_the_same_query_twice_gives_identical_hits() -> None:
    """Field for field, not just the ids: the eval must be repeatable."""
    retriever = dense(CORPUS)

    first = retriever.search(QUERY, "marketing", k=10)
    second = retriever.search(QUERY, "marketing", k=10)

    assert [hit.to_dict() for hit in first] == [hit.to_dict() for hit in second]


def test_k_cuts_the_list_and_ranks_still_start_at_one() -> None:
    hits = dense(CORPUS).search(QUERY, "marketing", k=2)

    assert len(hits) == 2
    assert [hit.rank for hit in hits] == [1, 2]


def test_an_empty_query_returns_nothing() -> None:
    """No question means no evidence, so we return nothing rather than everything."""
    retriever = dense(CORPUS)

    assert retriever.search("", "c_level", k=10) == []
    assert retriever.search("   ", "c_level", k=10) == []


def test_every_visible_chunk_is_a_candidate_however_unrelated() -> None:
    """Cosine has no natural zero, so unlike BM25 there is no score cut-off.

    OTHER shares no word with the query. Under BM25 it scores exactly 0 and is
    dropped; here it is simply last. k does all the cutting.
    """
    hits = dense(CORPUS).search(QUERY, "marketing", k=10)

    assert OTHER.chunk_id in ids_of(hits)


def test_an_empty_corpus_returns_nothing() -> None:
    retriever = dense([])

    assert retriever.vectors is None
    assert retriever.search(QUERY, "c_level", k=10) == []


def test_a_hit_carries_what_the_policy_replay_and_the_metrics_need() -> None:
    hit = dense(CORPUS).search(QUERY, "c_level", k=1)[0]

    assert hit.chunk_id == SECRET.chunk_id
    assert hit.doc_id == "finance/secret"
    assert hit.section_slug == "churn"
    assert hit.retriever == "dense"
    assert hit.owner_dept == "finance"
    assert hit.doc_type == "md"
    assert hit.sensitivity == "confidential"
    assert hit.allowed_roles == ("c_level",)
    assert hit.debug["cosine"] == hit.score
    assert hit.debug["embedder"] == "fake"


# --- the query goes through the embedder once, and through the cache ---------


def test_only_the_query_is_embedded_at_search_time() -> None:
    """Chunk vectors come off the matrix; re-embedding them per query is the
    mistake the old repo made with its index."""
    embedder = FakeEmbedder()
    retriever = DenseRetriever(CORPUS, embedder)
    embedder.calls.clear()

    retriever.search(QUERY, "marketing", k=10)

    assert embedder.calls == [[QUERY]]


# --- the saved artifact ------------------------------------------------------


def test_a_loaded_matrix_gives_exactly_the_same_hits_as_a_fresh_one(tmp_path: Path) -> None:
    """The round trip is the promise ingest makes to every later rung."""
    dense_path = tmp_path / "dense.npz"
    build_and_save_matrix(CORPUS, FakeEmbedder(), dense_path)

    in_memory = dense(CORPUS).search(QUERY, "marketing", k=10)
    from_disk = dense(CORPUS, dense_path=dense_path).search(QUERY, "marketing", k=10)

    assert [hit.to_dict() for hit in from_disk] == [hit.to_dict() for hit in in_memory]


def test_the_saved_chunk_ids_are_the_matrix_row_order(tmp_path: Path) -> None:
    dense_path = tmp_path / "dense.npz"
    build_and_save_matrix(CORPUS, FakeEmbedder(), dense_path)

    _vectors, stored_ids, model = load_matrix(dense_path)

    assert stored_ids == [chunk.chunk_id for chunk in CORPUS]
    assert model == "fake"


def test_the_matrix_records_the_embedding_space_not_the_wrapper(tmp_path: Path) -> None:
    """Through the cache the vectors are still Vertex vectors, and must say so."""
    dense_path = tmp_path / "dense.npz"
    cache = EmbeddingCache.load(tmp_path / "embed_cache.npz", "fake", EMBED_DIM)

    build_and_save_matrix(CORPUS, CachingEmbedder(FakeEmbedder(), cache), dense_path)

    _vectors, _ids, model = load_matrix(dense_path)
    assert model == "fake"


def test_the_saved_matrix_is_one_row_per_chunk_at_the_model_width(tmp_path: Path) -> None:
    dense_path = tmp_path / "dense.npz"

    n_rows = build_and_save_matrix(CORPUS, FakeEmbedder(), dense_path)
    vectors, _ids, _model = load_matrix(dense_path)

    assert n_rows == len(CORPUS)
    assert vectors.shape == (len(CORPUS), EMBED_DIM)
    assert vectors.dtype == np.float32


def test_a_matrix_built_from_other_chunks_is_refused(tmp_path: Path) -> None:
    """A row-number mismatch would score one chunk with another chunk's vector.

    That is worse than a crash: the eval numbers would still look plausible. So
    the chunk-id list is checked on load and a mismatch is refused outright.
    """
    dense_path = tmp_path / "dense.npz"
    build_and_save_matrix(CORPUS, FakeEmbedder(), dense_path)

    with pytest.raises(ValueError) as caught:
        dense(list(reversed(CORPUS)), dense_path=dense_path)
    assert "rebuild" in str(caught.value)

    with pytest.raises(ValueError):
        dense(CORPUS[:2], dense_path=dense_path)


def test_an_empty_chunk_list_does_not_wave_a_saved_matrix_through(tmp_path: Path) -> None:
    """The empty-corpus shortcut must not sit in front of the row-order guard.

    Ordered the other way, `DenseRetriever([], dense_path=stale)` would accept any
    matrix at all without a word, which is exactly what the guard is for.
    """
    dense_path = tmp_path / "dense.npz"
    build_and_save_matrix(CORPUS, FakeEmbedder(), dense_path)

    with pytest.raises(ValueError):
        dense([], dense_path=dense_path)


def test_a_missing_matrix_says_to_run_ingest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as caught:
        dense(CORPUS, dense_path=tmp_path / "nothing-here.npz")
    assert "finrag ingest" in str(caught.value)


def test_save_matrix_creates_the_folder_if_it_is_not_there(tmp_path: Path) -> None:
    dense_path = tmp_path / "deep" / "dense.npz"

    save_matrix(build_matrix(CORPUS, FakeEmbedder()), CORPUS, dense_path, model="fake")

    assert dense_path.exists()


# --- the registry -----------------------------------------------------------


def test_available_retrievers_still_lists_dense() -> None:
    """The full list moved to test_rerank.py, which owns the newest mode.

    Pinning it here as well means every new rung has to edit this file for a
    reason that has nothing to do with dense retrieval. Same reasoning R1 wrote
    into test_bm25.py when R2 broke its version of this assertion.
    """
    assert "dense" in available_retrievers()


def test_build_retriever_builds_the_dense_mode_in_memory() -> None:
    bundle = IndexBundle(chunks=CORPUS, embedder=FakeEmbedder())

    retriever = build_retriever("dense", bundle)

    assert isinstance(retriever, DenseRetriever)
    assert retriever.name == "dense"
    assert retriever.search(QUERY, "c_level", k=1)[0].chunk_id == SECRET.chunk_id


def test_build_retriever_uses_the_matrix_file_when_the_bundle_names_one(
    tmp_path: Path,
) -> None:
    dense_path = tmp_path / "dense.npz"
    build_and_save_matrix(CORPUS, FakeEmbedder(), dense_path)

    bundle = IndexBundle(chunks=CORPUS, dense_path=dense_path, embedder=FakeEmbedder())
    retriever = build_retriever("dense", bundle)

    assert retriever.search(QUERY, "marketing", k=10)[0].chunk_id == TIED_A.chunk_id


def test_dense_mode_without_an_embedder_says_how_to_build_one() -> None:
    """Falling back to a default embedder here could open a Vertex client by
    surprise, in a process that only asked for a retriever."""
    with pytest.raises(ValueError) as caught:
        build_retriever("dense", IndexBundle(chunks=CORPUS))

    assert "build_embedder" in str(caught.value)


# --- ingest writes the matrix for real --------------------------------------


def test_ingest_writes_a_matrix_that_matches_the_chunks_file(
    repo_root: Path, tmp_path: Path
) -> None:
    """One real ingest over the real corpus, offline, into a throwaway folder."""
    settings = replace(
        Settings.from_env(root=repo_root), artifacts_dir=tmp_path, embedder="fake"
    )

    stats = run_ingest(settings)
    chunks = read_chunks(settings.chunks_path)

    assert stats.n_embedded == stats.n_chunks == len(chunks)
    assert stats.dense_path == settings.dense_path

    _vectors, stored_ids, _model = load_matrix(settings.dense_path)
    assert stored_ids == [chunk.chunk_id for chunk in chunks]

    # And the artifact is usable as written: every hit is one this role may see.
    retriever = DenseRetriever(chunks, FakeEmbedder(), dense_path=settings.dense_path)
    hits = retriever.search("what are the recovery objectives", "engineering", k=10)

    assert hits, "the real matrix returned nothing for a plain question"
    for hit in hits:
        assert "engineering" in hit.allowed_roles


def test_a_second_ingest_embeds_nothing_new(repo_root: Path, tmp_path: Path) -> None:
    """The cache is what makes a re-run cost nothing on the real backend."""
    settings = replace(
        Settings.from_env(root=repo_root), artifacts_dir=tmp_path, embedder="fake"
    )
    run_ingest(settings)

    counting = FakeEmbedder()
    cache = EmbeddingCache.load(settings.embed_cache_path, "fake", EMBED_DIM)
    run_ingest(settings, embedder=CachingEmbedder(counting, cache))

    assert counting.calls == []


# The two guards a hand-forged or stale npz has to trip. Both were found by the
# R2-R4 verifier: each failure was silent, and a silent wrong ranking is worse
# than a crash because it still produces a table.


class _OtherSpaceEmbedder(FakeEmbedder):
    """The same fake vectors under a different model name."""

    name = "fake-other"
    model = "text-embedding-005"


def test_a_matrix_from_another_embedding_space_is_refused(tmp_path: Path) -> None:
    # The quiet failure: a Vertex-built matrix queried with the fake embedder
    # returns cosines around 0.05 and a ranking that is noise, and nothing raises.
    dense_path = tmp_path / "dense.npz"
    build_and_save_matrix(CORPUS, FakeEmbedder(), dense_path)

    with pytest.raises(ValueError) as caught:
        DenseRetriever(CORPUS, _OtherSpaceEmbedder(), dense_path=dense_path)

    message = str(caught.value)
    assert "fake" in message
    assert "text-embedding-005" in message


def test_the_matching_embedder_is_accepted(tmp_path: Path) -> None:
    # The other half: the guard must not refuse the ordinary case.
    dense_path = tmp_path / "dense.npz"
    build_and_save_matrix(CORPUS, FakeEmbedder(), dense_path)

    retriever = DenseRetriever(CORPUS, FakeEmbedder(), dense_path=dense_path)
    assert retriever.search("revenue", "c_level", k=1)


def test_a_matrix_with_fewer_rows_than_chunk_ids_is_refused(tmp_path: Path) -> None:
    # Hand-forged, the shape a half-written file would have. Without this guard
    # the failure is an IndexError inside the scoring loop at query time.
    dense_path = tmp_path / "dense.npz"
    vectors = build_matrix(CORPUS, FakeEmbedder())
    np.savez(
        dense_path,
        vectors=vectors[:-1],
        chunk_ids=np.array([chunk.chunk_id for chunk in CORPUS]),
        model=np.array("fake"),
    )

    with pytest.raises(ValueError, match="inconsistent"):
        load_matrix(dense_path)
