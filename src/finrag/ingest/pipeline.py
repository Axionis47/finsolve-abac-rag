"""The ingest run: corpus folders in, one chunks.jsonl artifact out.

This is the last step of ingest and the only one that touches the whole corpus.
It walks `data/corpus`, treats each folder name as the owning department, hands
markdown files to the chunker and the HR CSV to the row loader and the
aggregate builder, then asks the policy engine which roles may retrieve each
chunk and stamps the answer onto it. Finally it sorts every chunk by chunk_id
and writes one JSON object per line.

Two properties matter downstream and are both bought by that sort plus a fixed
tenure reference date: running ingest twice over an unchanged corpus produces a
byte-identical file, and `allowed_roles` is always a cache of real policy
decisions rather than a hand-written second permission table.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from finrag.config import Settings
from finrag.ingest.chunker import chunk_markdown
from finrag.ingest.hr_aggregates import REFERENCE_DATE, build_aggregates
from finrag.ingest.hr_csv import load_hr_rows
from finrag.models import ROLES, Chunk, ResourceAttrs
from finrag.policy.engine import PolicyEngine

# The folder whose documents everyone may read.
GENERAL_DEPT = "general"

# Sensitivity we stamp on markdown. The general folder is company-wide reading;
# every other folder is departmental, which the policy treats as internal.
GENERAL_SENSITIVITY = "general"
DEPARTMENT_SENSITIVITY = "internal"

# The one CSV in the corpus. Any other CSV would need its own PII review, so we
# match the name rather than the extension.
HR_CSV_FILENAME = "hr_data.csv"

# Tenure is measured against a pinned date, never the wall clock, so a rebuild
# next month does not change a single byte of the artifact. The date itself
# lives in hr_aggregates.py, next to the code that does the subtraction, so
# there is only ever one value to change.


@dataclass(frozen=True)
class IngestStats:
    """What one ingest run produced. Returned to the CLI for printing."""

    n_chunks: int
    by_doc_type: dict[str, int]
    by_owner_dept: dict[str, int]
    chunks_path: Path


def run_ingest(settings: Settings) -> IngestStats:
    """Build every chunk, stamp its roles, write the artifact, report counts."""
    engine = PolicyEngine.from_yaml(settings.policy_path)

    chunks = collect_chunks(settings.corpus_dir)
    chunks = stamp_allowed_roles(chunks, engine)
    chunks.sort(key=lambda chunk: chunk.chunk_id)

    write_chunks(settings.chunks_path, chunks)
    return _summarise(chunks, settings.chunks_path)


def collect_chunks(corpus_dir: str | Path) -> list[Chunk]:
    """Read every department folder and return its chunks, roles not yet set."""
    corpus_dir = Path(corpus_dir)
    chunks: list[Chunk] = []
    for entry in sorted(corpus_dir.iterdir()):
        if entry.is_dir() and not entry.name.startswith("."):
            chunks.extend(collect_department(entry, entry.name))
    return chunks


def collect_department(dept_dir: Path, owner_dept: str) -> list[Chunk]:
    """Read one folder. Markdown goes to the chunker, the HR CSV to its loader."""
    chunks: list[Chunk] = []
    for path in sorted(dept_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix == ".md":
            chunks.extend(_chunk_one_markdown(path, owner_dept))
        elif path.name == HR_CSV_FILENAME:
            chunks.extend(_chunk_hr_csv(path))
    return chunks


def _chunk_one_markdown(path: Path, owner_dept: str) -> list[Chunk]:
    """Chunk a single markdown file under its folder's department."""
    doc_id = f"{owner_dept}/{path.stem}"
    sensitivity = sensitivity_for(owner_dept)
    return chunk_markdown(path, doc_id, owner_dept, sensitivity)


def _chunk_hr_csv(path: Path) -> list[Chunk]:
    """One restricted chunk per employee row, plus the five summary chunks.

    The row chunks and the aggregates set their own owner department and
    sensitivity, because the CSV is the one source where those differ from the
    folder default.
    """
    rows = load_hr_rows(path)
    aggregates = build_aggregates(rows, reference_date=REFERENCE_DATE)
    return rows + aggregates


def sensitivity_for(owner_dept: str) -> str:
    """General folder means general reading; everything else is internal."""
    if owner_dept == GENERAL_DEPT:
        return GENERAL_SENSITIVITY
    return DEPARTMENT_SENSITIVITY


def stamp_allowed_roles(chunks: list[Chunk], engine: PolicyEngine) -> list[Chunk]:
    """Ask the policy engine, per chunk, which roles may retrieve it.

    The same three attributes drive the answer for most chunks, so decisions are
    cached by (owner_dept, doc_type, sensitivity). That is only a speed-up: the
    engine still decides every distinct resource shape.
    """
    decided: dict[tuple[str, str, str], tuple[str, ...]] = {}
    stamped: list[Chunk] = []

    for chunk in chunks:
        key = (chunk.owner_dept, chunk.doc_type, chunk.sensitivity)
        if key not in decided:
            resource = ResourceAttrs(chunk.owner_dept, chunk.doc_type, chunk.sensitivity)
            decided[key] = _in_canonical_order(engine.allowed_roles(resource))
        stamped.append(replace(chunk, allowed_roles=decided[key]))

    return stamped


def _in_canonical_order(roles: frozenset[str]) -> tuple[str, ...]:
    """Order roles as ROLES declares them, so the artifact never reshuffles."""
    ordered: list[str] = []
    for role in ROLES:
        if role in roles:
            ordered.append(role)
    return tuple(ordered)


def write_chunks(path: str | Path, chunks: list[Chunk]) -> None:
    """Write one compact JSON object per line, creating the folder if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for chunk in chunks:
            handle.write(chunk.to_json())
            handle.write("\n")


def read_chunks(path: str | Path) -> list[Chunk]:
    """Read a chunks.jsonl back into Chunk objects, skipping blank lines."""
    chunks: list[Chunk] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                chunks.append(Chunk.from_json(line))
    return chunks


def _summarise(chunks: list[Chunk], chunks_path: Path) -> IngestStats:
    doc_types: list[str] = []
    owner_depts: list[str] = []
    for chunk in chunks:
        doc_types.append(chunk.doc_type)
        owner_depts.append(chunk.owner_dept)

    return IngestStats(
        n_chunks=len(chunks),
        by_doc_type=_count_values(doc_types),
        by_owner_dept=_count_values(owner_depts),
        chunks_path=Path(chunks_path),
    )


def _count_values(values: list[str]) -> dict[str, int]:
    """Count how often each value appears, keyed in sorted order for printing."""
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))
