"""The ingest pipeline run end to end against the real corpus.

Every test here runs one real ingest into a temporary artifacts folder, so what
is checked is the artifact the rest of the system will load: the counts, the
role stamp on every chunk, the JSON round trip, and the promise that a second
run produces the same bytes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from finrag.config import Settings
from finrag.ingest.pipeline import IngestStats, read_chunks, run_ingest
from finrag.models import ROLES, Chunk, ResourceAttrs
from finrag.policy.engine import PolicyEngine

ALL_ROLES = frozenset(ROLES)

# The corpus is ten files. Chunk counts move a little when the chunker is
# retuned, so the band is wide enough to allow that and tight enough to catch a
# folder that stopped being read at all.
MIN_CHUNKS = 200
MAX_CHUNKS = 320

# These two are exact: one chunk per employee row, and the five fixed summaries.
EXPECTED_CSV_ROWS = 100
EXPECTED_AGGREGATES = 5


@dataclass(frozen=True)
class IngestRun:
    """One completed ingest: the stats it returned and the chunks it wrote."""

    stats: IngestStats
    chunks: list[Chunk]
    path: Path


@pytest.fixture(scope="module")
def ingest_run(repo_root: Path, tmp_path_factory: pytest.TempPathFactory) -> IngestRun:
    """Run ingest once for the whole module, into a throwaway artifacts folder."""
    artifacts_dir = tmp_path_factory.mktemp("artifacts")
    settings = replace(Settings.from_env(root=repo_root), artifacts_dir=artifacts_dir)

    stats = run_ingest(settings)
    return IngestRun(
        stats=stats,
        chunks=read_chunks(settings.chunks_path),
        path=settings.chunks_path,
    )


@pytest.fixture(scope="module")
def engine(policy_path: Path) -> PolicyEngine:
    return PolicyEngine.from_yaml(policy_path)


def chunks_of_type(run: IngestRun, doc_type: str) -> list[Chunk]:
    selected = []
    for chunk in run.chunks:
        if chunk.doc_type == doc_type:
            selected.append(chunk)
    return selected


def test_chunk_count_is_in_a_sane_band(ingest_run: IngestRun) -> None:
    assert MIN_CHUNKS <= ingest_run.stats.n_chunks <= MAX_CHUNKS
    assert ingest_run.stats.n_chunks == len(ingest_run.chunks)


def test_hr_csv_produces_one_chunk_per_row(ingest_run: IngestRun) -> None:
    assert ingest_run.stats.by_doc_type["csv_row"] == EXPECTED_CSV_ROWS


def test_hr_aggregates_are_the_five_fixed_summaries(ingest_run: IngestRun) -> None:
    assert ingest_run.stats.by_doc_type["aggregate"] == EXPECTED_AGGREGATES


def test_every_department_folder_was_read(ingest_run: IngestRun) -> None:
    """A missing folder is the failure this catches; the corpus has five."""
    expected = {"engineering", "finance", "general", "hr", "marketing"}
    assert set(ingest_run.stats.by_owner_dept) == expected


def test_every_chunk_carries_at_least_one_role(ingest_run: IngestRun) -> None:
    """An empty mask would make a chunk invisible to everyone, silently."""
    for chunk in ingest_run.chunks:
        assert chunk.allowed_roles, f"{chunk.chunk_id} has no allowed_roles"


def test_allowed_roles_match_the_policy_engine(
    ingest_run: IngestRun, engine: PolicyEngine
) -> None:
    """The stamp is a cache of decisions, so re-deciding must give the same set."""
    for chunk in ingest_run.chunks:
        resource = ResourceAttrs(chunk.owner_dept, chunk.doc_type, chunk.sensitivity)
        assert frozenset(chunk.allowed_roles) == engine.allowed_roles(resource)


def test_hr_row_chunks_are_exactly_hr_and_c_level(ingest_run: IngestRun) -> None:
    """HR owns its rows and c_level sees everything. Nobody else, ever.

    Exact equality, not a bound: HR must be able to find its own rows, and no
    other department may. The PII itself is not in the searchable text, so
    finding a row is safe; opening its payload still needs a `view_row` permit.
    """
    rows = chunks_of_type(ingest_run, "csv_row")
    assert len(rows) == EXPECTED_CSV_ROWS

    for chunk in rows:
        assert frozenset(chunk.allowed_roles) == frozenset({"hr", "c_level"}), chunk.chunk_id


def test_hr_aggregate_chunks_are_hr_and_c_level(ingest_run: IngestRun) -> None:
    aggregates = chunks_of_type(ingest_run, "aggregate")
    assert len(aggregates) == EXPECTED_AGGREGATES

    for chunk in aggregates:
        assert frozenset(chunk.allowed_roles) == frozenset({"hr", "c_level"})


def test_general_chunks_are_visible_to_every_role(ingest_run: IngestRun) -> None:
    general = []
    for chunk in ingest_run.chunks:
        if chunk.owner_dept == "general":
            general.append(chunk)

    assert general, "the general folder produced no chunks"
    for chunk in general:
        assert chunk.sensitivity == "general"
        assert frozenset(chunk.allowed_roles) == ALL_ROLES


def test_department_markdown_is_owner_plus_c_level(ingest_run: IngestRun) -> None:
    """The ordinary case: a department's own docs, and nobody else's."""
    for chunk in ingest_run.chunks:
        if chunk.doc_type == "md" and chunk.owner_dept != "general":
            expected = frozenset({chunk.owner_dept, "c_level"})
            assert frozenset(chunk.allowed_roles) == expected, chunk.chunk_id


def test_chunk_ids_are_unique_and_sorted(ingest_run: IngestRun) -> None:
    """Sorted ids are what makes two runs byte-identical."""
    ids = []
    for chunk in ingest_run.chunks:
        ids.append(chunk.chunk_id)

    assert len(set(ids)) == len(ids)
    assert ids == sorted(ids)


def test_jsonl_round_trips_through_chunk_from_json(ingest_run: IngestRun) -> None:
    """Read every written line back and compare the object, field for field."""
    lines = ingest_run.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(ingest_run.chunks)

    for line, chunk in zip(lines, ingest_run.chunks):
        assert Chunk.from_json(line) == chunk


def test_reindexing_twice_gives_identical_bytes(
    repo_root: Path, tmp_path: Path
) -> None:
    """Ingest is reproducible: same corpus in, same file out, no timestamps."""
    settings = replace(Settings.from_env(root=repo_root), artifacts_dir=tmp_path)

    run_ingest(settings)
    first = settings.chunks_path.read_bytes()

    run_ingest(settings)
    second = settings.chunks_path.read_bytes()

    assert first == second
