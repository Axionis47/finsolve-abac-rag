"""Precomputed HR summary chunks: headcount, salary, location, tenure, roles.

Also part of ingest. Individual HR rows are restricted, so a question like
"which department pays the most on average" can only be answered from a
summary. These five chunks hold those summaries. They are `internal` rather
than `restricted` because they describe groups, never a named person, and they
carry no payload. `allowed_roles` is left empty; the pipeline fills it by
evaluating the policy engine.

Tenure is measured against a reference date passed in by the caller, never
`datetime.now`, so a rebuild months later produces byte-identical chunks.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Iterable

from finrag.models import Chunk

DOC_ID = "hr/aggregates"
OWNER_DEPT = "hr"
SENSITIVITY = "internal"

# Fixed clock for tenure. Callers may override it, but the default is pinned so
# the artifact does not drift with the calendar.
REFERENCE_DATE = date(2026, 7, 29)

DAYS_PER_YEAR = 365.25

SECTION_SLUGS: tuple[str, ...] = (
    "headcount-by-department",
    "salary-by-department",
    "location-distribution",
    "tenure-by-department",
    "role-distribution",
)


def build_aggregates(
    rows: Iterable[Any],
    reference_date: date = REFERENCE_DATE,
) -> list[Chunk]:
    """Build the five HR aggregate chunks, always in the same order.

    `rows` may be the raw CSV dicts or the row Chunks from `load_hr_rows`,
    whose payload holds the same dicts.
    """
    row_dicts = _as_row_dicts(rows)
    return [
        _make_chunk(
            "headcount-by-department",
            "HR Aggregates > Headcount by Department",
            _headcount_text(row_dicts),
        ),
        _make_chunk(
            "salary-by-department",
            "HR Aggregates > Salary by Department",
            _salary_text(row_dicts),
        ),
        _make_chunk(
            "location-distribution",
            "HR Aggregates > Location Distribution",
            _location_text(row_dicts),
        ),
        _make_chunk(
            "tenure-by-department",
            "HR Aggregates > Tenure by Department",
            _tenure_text(row_dicts, reference_date),
        ),
        _make_chunk(
            "role-distribution",
            "HR Aggregates > Role Distribution",
            _role_text(row_dicts),
        ),
    ]


def _as_row_dicts(rows: Iterable[Any]) -> list[dict[str, str]]:
    """Accept either raw CSV rows or row Chunks and return plain dicts."""
    result: list[dict[str, str]] = []
    for row in rows:
        if isinstance(row, Chunk):
            result.append(dict(row.payload))
        else:
            result.append(dict(row))
    return result


def _make_chunk(section_slug: str, section_path: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"{DOC_ID}#{section_slug}",
        doc_id=DOC_ID,
        section_path=section_path,
        section_slug=section_slug,
        text=text,
        owner_dept=OWNER_DEPT,
        doc_type="aggregate",
        sensitivity=SENSITIVITY,
        allowed_roles=(),
        quarter=None,
        payload={},
    )


def _count_by(rows: list[dict[str, str]], field_name: str) -> dict[str, int]:
    """Count rows per value of one field. Blank values become "Unknown"."""
    counts: dict[str, int] = {}
    for row in rows:
        value = (row.get(field_name) or "").strip() or "Unknown"
        counts[value] = counts.get(value, 0) + 1
    return counts


def _numbers_by(
    rows: list[dict[str, str]],
    key_field: str,
    value_field: str,
) -> dict[str, list[float]]:
    """Group one numeric field by another field, skipping unparsable values."""
    grouped: dict[str, list[float]] = {}
    for row in rows:
        key = (row.get(key_field) or "").strip() or "Unknown"
        raw = (row.get(value_field) or "").strip()
        try:
            number = float(raw)
        except ValueError:
            continue
        grouped.setdefault(key, []).append(number)
    return grouped


def _headcount_text(rows: list[dict[str, str]]) -> str:
    counts = _count_by(rows, "department")
    lines = ["Employee headcount by department."]
    for dept in sorted(counts):
        lines.append(f"{dept}: {counts[dept]} employees")
    lines.append(f"Total employees: {len(rows)}")
    return "\n".join(lines)


def _salary_text(rows: list[dict[str, str]]) -> str:
    salaries = _numbers_by(rows, "department", "salary")
    lines = ["Average, lowest and highest salary by department."]
    for dept in sorted(salaries):
        values = salaries[dept]
        average = sum(values) / len(values)
        lines.append(
            f"{dept}: average salary {average:,.0f}, "
            f"lowest {min(values):,.0f}, highest {max(values):,.0f}, "
            f"across {len(values)} employees."
        )
    return "\n".join(lines)


def _location_text(rows: list[dict[str, str]]) -> str:
    counts = _count_by(rows, "location")
    lines = ["Employee distribution by office location."]
    for location in sorted(counts):
        lines.append(f"{location}: {counts[location]} employees")
    return "\n".join(lines)


def _tenure_text(rows: list[dict[str, str]], reference_date: date) -> str:
    """Average years since joining, per department, measured at reference_date."""
    tenures: dict[str, list[float]] = {}
    for row in rows:
        dept = (row.get("department") or "").strip() or "Unknown"
        joined = _parse_date(row.get("date_of_joining"))
        if joined is None:
            continue
        years = (reference_date - joined).days / DAYS_PER_YEAR
        tenures.setdefault(dept, []).append(years)

    lines = [f"Average employee tenure by department, as of {reference_date.isoformat()}."]
    for dept in sorted(tenures):
        values = tenures[dept]
        average = sum(values) / len(values)
        lines.append(f"{dept}: {average:.1f} years average tenure across {len(values)} employees.")
    return "\n".join(lines)


def _role_text(rows: list[dict[str, str]]) -> str:
    """Roles listed most common first, ties broken by name so order is stable."""
    counts = _count_by(rows, "role")
    ordered = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    lines = ["Employee distribution by job role."]
    for role_name, count in ordered:
        lines.append(f"{role_name}: {count} employees")
    return "\n".join(lines)


def _parse_date(raw: str | None) -> date | None:
    """Parse an ISO date, returning None instead of raising on bad input."""
    text = (raw or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None
