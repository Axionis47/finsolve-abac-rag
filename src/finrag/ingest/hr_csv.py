"""Turn the HR CSV into one Chunk per employee row, with PII kept out of search.

This sits in the ingest stage, next to the markdown chunker. The searchable
`text` is built from four harmless fields only: name, role, department and
location. Everything else in the row, including salary, email, date of birth,
joining date and ratings, goes into `payload`, which the API serves only when
the policy engine permits `view_row`. `allowed_roles` is left empty here; the
pipeline fills it by evaluating the policy engine.

Because the searchable text carries no PII, the policy lets HR and c_level
retrieve these rows like any other chunk. The payload is the part that stays
locked behind `view_row`.
"""

from __future__ import annotations

import csv
from pathlib import Path

from finrag.models import Chunk

DOC_ID = "hr/hr_data"
SECTION_PATH = "HR Data > Employee Records"
OWNER_DEPT = "hr"
SENSITIVITY = "restricted"

# The only row fields allowed anywhere near the search index.
SEARCHABLE_FIELDS: tuple[str, ...] = (
    "full_name",
    "role",
    "department",
    "location",
)

# Everything else on the row. Listed for documentation and for the leakage
# tests, which assert none of these values ever reach `text`.
#
# date_of_joining sits here rather than in SEARCHABLE_FIELDS: a joining date
# pins a named person to a date and nobody searches by it. The tenure
# aggregates are unaffected, because they read the CSV rows directly and never
# go through the searchable text.
PII_FIELDS: tuple[str, ...] = (
    "email",
    "date_of_birth",
    "date_of_joining",
    "manager_id",
    "salary",
    "leave_balance",
    "leaves_taken",
    "attendance_pct",
    "performance_rating",
    "last_review_date",
)

# Label printed for each searchable field, in the order they appear in `text`.
_TEXT_LABELS: tuple[tuple[str, str], ...] = (
    ("full_name", "Employee"),
    ("role", "Role"),
    ("department", "Department"),
    ("location", "Location"),
)


def build_searchable_text(row: dict[str, str]) -> str:
    """Render the four non-PII fields as short labelled lines.

    A field that is missing or blank is skipped rather than printed empty, so a
    partial row still produces sensible text.
    """
    lines: list[str] = []
    for field_name, label in _TEXT_LABELS:
        value = (row.get(field_name) or "").strip()
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def row_to_chunk(row: dict[str, str]) -> Chunk:
    """Build one Chunk from one CSV row.

    The chunk id is a pure function of the employee id, so re-running ingest
    never moves a gold label.
    """
    employee_id = (row.get("employee_id") or "").strip()
    section_slug = f"row-{employee_id}"
    return Chunk(
        chunk_id=f"{DOC_ID}#{section_slug}",
        doc_id=DOC_ID,
        section_path=SECTION_PATH,
        section_slug=section_slug,
        text=build_searchable_text(row),
        owner_dept=OWNER_DEPT,
        doc_type="csv_row",
        sensitivity=SENSITIVITY,
        allowed_roles=(),
        quarter=None,
        payload=dict(row),
    )


def load_hr_rows(csv_path: str | Path) -> list[Chunk]:
    """Read the HR CSV and return one restricted Chunk per row.

    Rows keep the order they have in the file, so two runs over the same file
    produce the same list.
    """
    path = Path(csv_path)
    chunks: list[Chunk] = []
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            chunks.append(row_to_chunk(row))
    return chunks
