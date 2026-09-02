"""The ingest run: corpus folders in, chunks.jsonl plus both indexes out.

This is the last step of ingest and the only one that touches the whole corpus.
It walks `data/corpus`, treats each folder name as the owning department, hands
markdown files to the chunker and the HR CSV to the row loader and the
aggregate builder, then asks the policy engine which roles may retrieve each
chunk and stamps the answer onto it. Finally it sorts every chunk by chunk_id,
writes one JSON object per line, and builds both search indexes over those same
chunks in that same order.

Both indexes are built here, once, and not per query. The BM25 index lives in
`artifacts/bm25/` and the tokenizer that builds it is the one the query side
imports, so the two can never drift apart. See retrieval/bm25.py for why that
matters. The dense matrix lives in `artifacts/dense.npz`, and its vectors go
through the embedding cache, so a second ingest over an unchanged corpus makes no
API calls at all.

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
from finrag.ingest.registry import ChunkerFn, build_chunker
from finrag.models import ROLES, Chunk, ResourceAttrs
from finrag.policy.engine import PolicyEngine
from finrag.retrieval.bm25 import build_and_save_index
from finrag.retrieval.dense import build_and_save_matrix
from finrag.retrieval.embed import Embedder, build_embedder

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
    bm25_dir: Path
    n_indexed: int
    dense_path: Path
    n_embedded: int
    embedder_name: str
    chunker: str


def run_ingest(settings: Settings, embedder: Embedder | None = None) -> IngestStats:
    """Build every chunk, stamp its roles, write all three artifacts, report counts.

    `embedder` is injectable so a caller can hand in the offline fake; left out,
    it is chosen by `settings.embedder`, which is "vertex" unless told otherwise.
    """
    engine = PolicyEngine.from_yaml(settings.policy_path)
    chunker_fn = build_chunker(settings.chunker)

    chunks = collect_chunks(settings.corpus_dir, chunker_fn=chunker_fn)
    chunks = stamp_allowed_roles(chunks, engine)
    chunks.sort(key=lambda chunk: chunk.chunk_id)

    # An empty corpus is a mis-set path, not a valid run. Say so here rather than
    # letting the index build fail later on a division by zero.
    if not chunks:
        raise ValueError(f"No chunks found under {settings.corpus_dir}. Nothing to index.")

    write_chunks(settings.chunks_path, chunks)
    # Same list, same order for both indexes, so row N is always line N.
    n_indexed = build_and_save_index(chunks, settings.bm25_dir)

    if embedder is None:
        embedder = build_embedder(settings)
    n_embedded = build_and_save_matrix(chunks, embedder, settings.dense_path)

    return _summarise(chunks, settings, n_indexed, n_embedded, embedder.name)


def collect_chunks(corpus_dir: str | Path, chunker_fn: ChunkerFn = chunk_markdown) -> list[Chunk]:
    """Read every department folder and return its chunks, roles not yet set."""
    corpus_dir = Path(corpus_dir)
    chunks: list[Chunk] = []
    for entry in sorted(corpus_dir.iterdir()):
        if entry.is_dir() and not entry.name.startswith("."):
            chunks.extend(collect_department(entry, entry.name, chunker_fn=chunker_fn))
    return chunks


def collect_department(
    dept_dir: Path, owner_dept: str, chunker_fn: ChunkerFn = chunk_markdown
) -> list[Chunk]:
    """Read one folder. Markdown goes to the chunker, the HR CSV to its loader."""
    chunks: list[Chunk] = []
    for path in sorted(dept_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix == ".md":
            chunks.extend(_chunk_one_markdown(path, owner_dept, chunker_fn=chunker_fn))
        elif path.name == HR_CSV_FILENAME:
            chunks.extend(_chunk_hr_csv(path))
    return chunks


def _chunk_one_markdown(
    path: Path, owner_dept: str, chunker_fn: ChunkerFn = chunk_markdown
) -> list[Chunk]:
    """Chunk a single markdown file under its folder's department."""
    doc_id = f"{owner_dept}/{path.stem}"
    sensitivity = sensitivity_for(owner_dept)
    return chunker_fn(path, doc_id, owner_dept, sensitivity)


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


def _summarise(
    chunks: list[Chunk],
    settings: Settings,
    n_indexed: int,
    n_embedded: int,
    embedder_name: str,
) -> IngestStats:
    doc_types: list[str] = []
    owner_depts: list[str] = []
    for chunk in chunks:
        doc_types.append(chunk.doc_type)
        owner_depts.append(chunk.owner_dept)

    return IngestStats(
        n_chunks=len(chunks),
        by_doc_type=_count_values(doc_types),
        by_owner_dept=_count_values(owner_depts),
        chunks_path=Path(settings.chunks_path),
        bm25_dir=Path(settings.bm25_dir),
        n_indexed=n_indexed,
        dense_path=Path(settings.dense_path),
        n_embedded=n_embedded,
        embedder_name=embedder_name,
        chunker=settings.chunker,
    )


def _count_values(values: list[str]) -> dict[str, int]:
    """Count how often each value appears, keyed in sorted order for printing."""
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))
