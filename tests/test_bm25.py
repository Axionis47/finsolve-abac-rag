"""The BM25 mode: the role mask, the ranking, and the saved index.

Four things are checked here that a whole eval run would not tell us apart:

  * the mask runs BEFORE ranking. The proof is direct, the same shape as the one
    in test_naive.py: the highest scoring chunk in the corpus is one this role may
    not see, and `k` is wider than the entire visible pool, so a post-filter would
    still have had room to return it. It must not appear at any rank.
  * IDF earns its keep. A word in one chunk outranks a word in most chunks. The
    same query is run through the naive retriever, which scores those chunks level
    and puts the rare-word chunk LAST. That difference is the whole reason this
    rung exists.
  * the saved index is the same index. Ingest writes it once; if a loaded index
    gave even slightly different hits, the eval numbers would not describe what
    the API serves.
  * ties break on chunk_id and two runs agree, or two runs of the same eval would
    post different numbers for no reason.

The fixture below is tiny and hand-checked. Scores are compared as exact floats
where two chunks are genuinely identical in length and matched term, which under
BM25 they are.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from finrag.config import Settings
from finrag.ingest.pipeline import read_chunks, run_ingest
from finrag.models import Chunk
from finrag.retrieval import naive
from finrag.retrieval.bm25 import (
    CHUNK_IDS_FILENAME,
    PARAMS_FILENAME,
    BM25Retriever,
    build_and_save_index,
    build_index,
    build_vocab,
    save_index,
    tokenize,
)
from finrag.retrieval.naive import NaiveRetriever
from finrag.retrieval.registry import IndexBundle, available_retrievers, build_retriever

# The bait: the top scoring chunk for the query, readable only by c_level. It
# holds both query words and repeats the rare one.
SECRET = Chunk(
    chunk_id="finance/secret#margins",
    doc_id="finance/secret",
    section_path="Secret > Margins",
    section_slug="margins",
    text="zeppelin notes zeppelin airship dossier margins",
    owner_dept="finance",
    doc_type="md",
    sensitivity="confidential",
    allowed_roles=("c_level",),
)


def _marketing_chunk(chunk_id: str, section_slug: str, text: str) -> Chunk:
    """A chunk marketing and c_level may read. Keeps the fixture readable."""
    return Chunk(
        chunk_id=chunk_id,
        doc_id=chunk_id.split("#")[0],
        section_path="Market Report > Section",
        section_slug=section_slug,
        text=text,
        owner_dept="marketing",
        doc_type="md",
        sensitivity="internal",
        allowed_roles=("marketing", "c_level"),
    )


# "notes" is the common word: it sits in three visible chunks and in SECRET.
# All three are two words long, so BM25 scores them exactly level and the tie has
# to be broken by chunk_id.
COMMON_A = _marketing_chunk("marketing/a#one", "one", "campaign notes")
COMMON_B = _marketing_chunk("marketing/b#two", "two", "budget notes")
COMMON_D = _marketing_chunk("marketing/d#three", "three", "creative notes")

# "zeppelin" is the rare word: one visible chunk, and this id sorts LAST, so the
# naive tie-break puts it bottom of the list. BM25 puts it top.
RARE_Z = _marketing_chunk("marketing/z#airship", "airship", "the zeppelin flew")

# Shares no word with the query, so it never earns a slot.
UNRELATED = _marketing_chunk("marketing/u#logo", "logo", "brand colours and typeface")

# The bait is listed first on purpose: source order must not save it either.
CORPUS = [SECRET, COMMON_A, COMMON_B, COMMON_D, UNRELATED, RARE_Z]

QUERY = "zeppelin notes"

# The four chunks marketing may see that contain a query word, best first.
EXPECTED_FOR_MARKETING = [
    "marketing/z#airship",
    "marketing/a#one",
    "marketing/b#two",
    "marketing/d#three",
]


def ids_of(hits: list) -> list[str]:
    out: list[str] = []
    for hit in hits:
        out.append(hit.chunk_id)
    return out


# --- the tokenizer is shared, on purpose ------------------------------------


def test_the_index_and_the_query_use_the_very_same_tokenizer() -> None:
    """Not a copy of it: the same function object.

    bm25s looks each query word up in the vocabulary built at index time. A word
    split or folded differently at query time is simply not found, and its
    evidence vanishes with no error at all. Importing beats re-implementing.
    """
    assert tokenize is naive.tokenize


# --- the invariant: mask before rank ----------------------------------------


def test_the_best_scoring_chunk_is_hidden_when_the_role_may_not_see_it() -> None:
    """A denied chunk never occupies a rank slot, and never displaces one.

    SECRET outscores everything for this query. k is 10, wider than the whole
    visible pool of four, so nothing is being squeezed out by the cut-off. If the
    mask ran after scoring, SECRET would be sitting at rank 1 right now.
    """
    retriever = BM25Retriever(CORPUS)

    hits = retriever.search(QUERY, "marketing", k=10)

    assert SECRET.chunk_id not in ids_of(hits)
    assert ids_of(hits) == EXPECTED_FOR_MARKETING
    # Ranks stay 1..4 with no gap, so the denied chunk did not silently consume
    # a slot on the way past.
    assert [hit.rank for hit in hits] == [1, 2, 3, 4]


def test_the_same_chunk_is_reachable_by_the_role_that_may_read_it() -> None:
    """The control: the bait really was findable, and the role is what stopped it."""
    hits = BM25Retriever(CORPUS).search(QUERY, "c_level", k=10)

    assert hits[0].chunk_id == SECRET.chunk_id
    assert hits[0].score > hits[1].score


def test_a_role_with_nothing_visible_gets_nothing() -> None:
    retriever = BM25Retriever(CORPUS)
    assert retriever.search(QUERY, "engineering", k=10) == []
    assert retriever.search(QUERY, "not_a_role", k=10) == []


# --- what BM25 buys over plain word overlap ---------------------------------


def test_a_rare_word_outranks_a_common_one_where_naive_cannot_tell() -> None:
    """IDF sanity, stated as the comparison that motivates this rung.

    "zeppelin" is in one chunk, "notes" in four. BM25 ranks the rare-word chunk
    first. The naive retriever counts one matched word for all four chunks, scores
    them level, and its chunk_id tie-break drops the rare-word chunk to the bottom.
    """
    bm25_hits = BM25Retriever(CORPUS).search(QUERY, "marketing", k=10)
    naive_hits = NaiveRetriever(CORPUS).search(QUERY, "marketing", k=10)

    assert bm25_hits[0].chunk_id == RARE_Z.chunk_id
    assert bm25_hits[0].score > bm25_hits[1].score

    assert ids_of(naive_hits)[-1] == RARE_Z.chunk_id
    for hit in naive_hits:
        assert hit.score == 1.0


def test_ties_break_on_chunk_id() -> None:
    """The three common-word chunks score identically, so the id decides."""
    hits = BM25Retriever(CORPUS).search(QUERY, "marketing", k=10)
    tied = hits[1:]

    assert ids_of(tied) == ["marketing/a#one", "marketing/b#two", "marketing/d#three"]
    assert tied[0].score == tied[1].score == tied[2].score


def test_ties_break_the_same_way_whatever_order_the_chunks_arrived_in() -> None:
    """Feeding the corpus backwards must not change the answer."""
    forwards = BM25Retriever(CORPUS).search(QUERY, "marketing", k=10)
    backwards = BM25Retriever(list(reversed(CORPUS))).search(QUERY, "marketing", k=10)
    assert ids_of(forwards) == ids_of(backwards)


# --- boring but load-bearing behaviour --------------------------------------


def test_the_same_query_twice_gives_identical_hits() -> None:
    """Field for field, not just the ids: the eval must be repeatable."""
    retriever = BM25Retriever(CORPUS)

    first = retriever.search(QUERY, "marketing", k=10)
    second = retriever.search(QUERY, "marketing", k=10)

    assert [hit.to_dict() for hit in first] == [hit.to_dict() for hit in second]


def test_k_cuts_the_list_and_ranks_still_start_at_one() -> None:
    hits = BM25Retriever(CORPUS).search(QUERY, "marketing", k=2)

    assert len(hits) == 2
    assert [hit.rank for hit in hits] == [1, 2]
    assert ids_of(hits) == EXPECTED_FOR_MARKETING[:2]


def test_an_empty_query_returns_nothing() -> None:
    """No words means no evidence, so we return nothing rather than everything."""
    retriever = BM25Retriever(CORPUS)
    assert retriever.search("", "c_level", k=10) == []
    assert retriever.search("   ", "c_level", k=10) == []
    # Punctuation alone tokenises to no words at all.
    assert retriever.search("?!,.", "c_level", k=10) == []


def test_a_query_of_words_the_corpus_never_used_returns_nothing() -> None:
    """Unknown words are not in the vocabulary, so every score is zero."""
    assert BM25Retriever(CORPUS).search("quixotic bellwether", "c_level", k=10) == []


def test_a_chunk_sharing_no_word_with_the_query_never_appears() -> None:
    hits = BM25Retriever(CORPUS).search(QUERY, "marketing", k=10)
    assert UNRELATED.chunk_id not in ids_of(hits)


def test_an_empty_corpus_returns_nothing() -> None:
    """bm25s cannot index nothing, so the retriever skips the index entirely."""
    retriever = BM25Retriever([])
    assert retriever.index is None
    assert retriever.search(QUERY, "c_level", k=10) == []


def test_a_hit_carries_what_the_policy_replay_and_the_metrics_need() -> None:
    hit = BM25Retriever(CORPUS).search(QUERY, "marketing", k=1)[0]

    assert hit.doc_id == "marketing/z"
    assert hit.section_slug == "airship"
    assert hit.retriever == "bm25"
    assert hit.owner_dept == "marketing"
    assert hit.doc_type == "md"
    assert hit.sensitivity == "internal"
    assert hit.allowed_roles == ("marketing", "c_level")
    assert hit.debug["bm25_score"] == hit.score
    assert hit.debug["matched_terms"] == ["zeppelin"]
    assert hit.debug["n_query_terms"] == 2


# --- the saved artifact ------------------------------------------------------


def test_a_loaded_index_gives_exactly_the_same_hits_as_a_fresh_one(tmp_path: Path) -> None:
    """The round trip is the promise ingest makes to every later rung."""
    index_dir = tmp_path / "bm25"
    build_and_save_index(CORPUS, index_dir)

    in_memory = BM25Retriever(CORPUS).search(QUERY, "marketing", k=10)
    from_disk = BM25Retriever(CORPUS, index_dir=index_dir).search(QUERY, "marketing", k=10)

    assert [hit.to_dict() for hit in from_disk] == [hit.to_dict() for hit in in_memory]


def test_saving_the_index_twice_gives_identical_bytes(tmp_path: Path) -> None:
    """Nothing in the index folder is a timestamp, so rebuilds diff cleanly.

    bm25s does write its own version number into params.index.json. That is fixed
    by the installed library, not by the run, so it does not make two builds
    differ; upgrading bm25s is meant to change the artifact.
    """
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    build_and_save_index(CORPUS, first_dir)
    build_and_save_index(CORPUS, second_dir)

    names = sorted(path.name for path in first_dir.iterdir())
    assert PARAMS_FILENAME in names
    assert CHUNK_IDS_FILENAME in names

    for name in names:
        assert (first_dir / name).read_bytes() == (second_dir / name).read_bytes(), name


def test_the_vocabulary_is_numbered_alphabetically() -> None:
    """The mechanism behind the test below: our numbering, not a set's order."""
    vocab = build_vocab([["notes", "campaign"], ["zeppelin", "notes"]])

    assert vocab == {"campaign": 0, "notes": 1, "zeppelin": 2}


def test_two_separate_processes_save_the_same_bytes(repo_root: Path, tmp_path: Path) -> None:
    """The reason we number the vocabulary ourselves, proved the only honest way.

    Left to bm25s, token numbers come out of a Python set, and set order for
    strings depends on the hash seed the interpreter started with. Two builds in
    ONE process share a seed, agree, and would hide the problem completely. So
    this runs two fresh interpreters with different seeds over the real corpus,
    whose vocabulary is big enough that two orderings could not coincide by luck.
    """
    settings = replace(Settings.from_env(root=repo_root), artifacts_dir=tmp_path)
    run_ingest(settings)

    first_dir = tmp_path / "seed0"
    second_dir = tmp_path / "seed1"
    _build_index_in_a_fresh_process(repo_root, settings.chunks_path, first_dir, "0")
    _build_index_in_a_fresh_process(repo_root, settings.chunks_path, second_dir, "1")

    vocab = json.loads((first_dir / "vocab.index.json").read_text(encoding="utf-8"))
    assert len(vocab) > 1000, "corpus vocabulary too small for this test to mean anything"

    for path in sorted(first_dir.iterdir()):
        assert path.read_bytes() == (second_dir / path.name).read_bytes(), path.name


def _build_index_in_a_fresh_process(
    repo_root: Path,
    chunks_path: Path,
    index_dir: Path,
    hash_seed: str,
) -> None:
    """Build an index from a chunks.jsonl in a new interpreter. Offline, no model."""
    script = (
        "import sys\n"
        "from finrag.ingest.pipeline import read_chunks\n"
        "from finrag.retrieval.bm25 import build_and_save_index\n"
        "build_and_save_index(read_chunks(sys.argv[1]), sys.argv[2])\n"
    )
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = hash_seed
    # The child needs to find `finrag` even when the editable install is unhappy,
    # which is the same reason conftest.py patches sys.path.
    environment["PYTHONPATH"] = str(repo_root / "src")

    subprocess.run(
        [sys.executable, "-c", script, str(chunks_path), str(index_dir)],
        check=True,
        env=environment,
    )


def test_the_saved_chunk_ids_are_the_index_row_order(tmp_path: Path) -> None:
    index_dir = tmp_path / "bm25"
    build_and_save_index(CORPUS, index_dir)

    stored = json.loads((index_dir / CHUNK_IDS_FILENAME).read_text(encoding="utf-8"))
    assert stored == [chunk.chunk_id for chunk in CORPUS]


def test_an_index_built_from_other_chunks_is_refused(tmp_path: Path) -> None:
    """A row-number mismatch would attach one chunk's score to another chunk.

    That is worse than a crash: the eval numbers would still look plausible. So
    the chunk-id list is checked on load and a mismatch is refused outright.
    """
    index_dir = tmp_path / "bm25"
    build_and_save_index(CORPUS, index_dir)

    with pytest.raises(ValueError) as caught:
        BM25Retriever(list(reversed(CORPUS)), index_dir=index_dir)
    assert "rebuild" in str(caught.value)

    with pytest.raises(ValueError):
        BM25Retriever(CORPUS[:3], index_dir=index_dir)


def test_an_empty_chunk_list_does_not_excuse_a_mismatched_index(tmp_path: Path) -> None:
    # The order of the two guards in `__init__`. With the empty-corpus shortcut
    # first, this construction accepted a stale index without a word.
    index_dir = tmp_path / "bm25"
    build_and_save_index(CORPUS, index_dir)

    with pytest.raises(ValueError):
        BM25Retriever([], index_dir=index_dir)


def test_a_missing_index_says_to_run_ingest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as caught:
        BM25Retriever(CORPUS, index_dir=tmp_path / "nothing-here")
    assert "finrag ingest" in str(caught.value)


def test_a_half_written_index_folder_names_the_file_that_is_missing(tmp_path: Path) -> None:
    # params present, chunk ids gone: what an interrupted ingest leaves behind.
    index_dir = tmp_path / "bm25"
    build_and_save_index(CORPUS, index_dir)
    (index_dir / CHUNK_IDS_FILENAME).unlink()

    with pytest.raises(FileNotFoundError) as caught:
        BM25Retriever(CORPUS, index_dir=index_dir)

    message = str(caught.value)
    assert CHUNK_IDS_FILENAME in message
    assert "finrag ingest" in message


def test_save_index_creates_the_folder_if_it_is_not_there(tmp_path: Path) -> None:
    index_dir = tmp_path / "deep" / "bm25"
    save_index(build_index(CORPUS), CORPUS, index_dir)
    assert (index_dir / PARAMS_FILENAME).exists()


# --- the registry -----------------------------------------------------------


def test_available_retrievers_still_lists_bm25() -> None:
    """The full list is pinned in test_dense.py, which owns the newest mode."""
    assert "bm25" in available_retrievers()


def test_build_retriever_builds_the_bm25_mode_in_memory() -> None:
    retriever = build_retriever("bm25", IndexBundle(chunks=CORPUS))

    assert isinstance(retriever, BM25Retriever)
    assert retriever.name == "bm25"
    assert retriever.search(QUERY, "marketing", k=10)[0].chunk_id == RARE_Z.chunk_id


def test_build_retriever_uses_the_index_folder_when_the_bundle_names_one(
    tmp_path: Path,
) -> None:
    index_dir = tmp_path / "bm25"
    build_and_save_index(CORPUS, index_dir)

    retriever = build_retriever("bm25", IndexBundle(chunks=CORPUS, bm25_dir=index_dir))

    assert ids_of(retriever.search(QUERY, "marketing", k=10)) == EXPECTED_FOR_MARKETING


# --- ingest writes the index for real ---------------------------------------


def test_ingest_writes_an_index_that_matches_the_chunks_file(
    repo_root: Path, tmp_path: Path
) -> None:
    """One real ingest over the real corpus, into a throwaway artifacts folder."""
    settings = replace(Settings.from_env(root=repo_root), artifacts_dir=tmp_path)

    stats = run_ingest(settings)
    chunks = read_chunks(settings.chunks_path)

    assert stats.n_indexed == stats.n_chunks == len(chunks)
    assert stats.bm25_dir == settings.bm25_dir

    stored = json.loads((settings.bm25_dir / CHUNK_IDS_FILENAME).read_text(encoding="utf-8"))
    assert stored == [chunk.chunk_id for chunk in chunks]

    # And the artifact is usable as written: a lexical question finds its section
    # and every hit is one this role may see.
    retriever = BM25Retriever(chunks, index_dir=settings.bm25_dir)
    hits = retriever.search("RTO and RPO recovery objectives", "engineering", k=10)

    assert hits, "the real index returned nothing for a plain lexical question"
    for hit in hits:
        assert "engineering" in hit.allowed_roles
