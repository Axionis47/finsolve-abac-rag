"""HR row ingest: shape of each chunk, and proof that no PII reaches the text."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from finrag.ingest.hr_csv import PII_FIELDS, SEARCHABLE_FIELDS, load_hr_rows

# Fields the searchable text is allowed to contain. Written out here rather
# than imported alone, so dropping a field from the module has to be a
# deliberate edit in two places.
ALLOWED_FIELDS = ("full_name", "role", "department", "location")


@pytest.fixture(scope="module")
def hr_csv_path(repo_root: Path) -> Path:
    return repo_root / "data" / "corpus" / "hr" / "hr_data.csv"


@pytest.fixture(scope="module")
def raw_rows(hr_csv_path: Path) -> list[dict[str, str]]:
    """Read the CSV here, independently of the code under test."""
    with open(hr_csv_path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def chunks(hr_csv_path: Path):
    return load_hr_rows(hr_csv_path)


def test_one_chunk_per_row(chunks, raw_rows):
    assert len(chunks) == len(raw_rows) == 100


def test_chunk_identity_and_metadata(chunks, raw_rows):
    first = chunks[0]
    assert first.chunk_id == f"hr/hr_data#row-{raw_rows[0]['employee_id']}"
    assert first.doc_id == "hr/hr_data"
    assert first.section_slug == f"row-{raw_rows[0]['employee_id']}"
    for chunk in chunks:
        assert chunk.doc_type == "csv_row"
        assert chunk.sensitivity == "restricted"
        assert chunk.owner_dept == "hr"
        assert chunk.allowed_roles == ()


def test_chunk_ids_are_unique(chunks):
    ids = [chunk.chunk_id for chunk in chunks]
    assert len(set(ids)) == len(ids)


def test_searchable_fields_are_the_four_allowed_ones():
    """The module's own list and this test's list must agree."""
    assert SEARCHABLE_FIELDS == ALLOWED_FIELDS


def test_text_format_matches_the_four_allowed_fields(chunks, raw_rows):
    row = raw_rows[0]
    expected = (
        f"Employee: {row['full_name']}\n"
        f"Role: {row['role']}\n"
        f"Department: {row['department']}\n"
        f"Location: {row['location']}"
    )
    assert chunks[0].text == expected


def test_every_text_is_exactly_the_allowed_fields(chunks, raw_rows):
    for chunk, row in zip(chunks, raw_rows):
        lines = chunk.text.split("\n")
        assert len(lines) == len(ALLOWED_FIELDS)
        for line, field_name in zip(lines, ALLOWED_FIELDS):
            assert line.endswith(row[field_name])


def test_payload_holds_the_whole_row(chunks, raw_rows):
    assert chunks[0].payload == raw_rows[0]
    for field_name in PII_FIELDS:
        assert field_name in chunks[0].payload


def test_no_pii_token_in_any_searchable_text(chunks, raw_rows):
    all_text = "\n".join(chunk.text for chunk in chunks)

    # Email domain, one raw salary, and the row-1 rating field label.
    assert "@fintechco.com" not in all_text
    assert "1332478" not in all_text
    assert "Rating" not in all_text

    for row in raw_rows:
        assert row["email"] not in all_text
        assert row["date_of_birth"] not in all_text
        assert row["salary"] not in all_text
        assert row["salary"].split(".")[0] not in all_text
        assert row["last_review_date"] not in all_text
        assert row["manager_id"] not in all_text
        # Joining dates left the searchable text; the tenure aggregates read
        # them straight off the CSV instead.
        assert row["date_of_joining"] not in all_text


def test_no_rating_or_attendance_digits_leak(chunks, raw_rows):
    """Ratings and attendance are single or short numbers, so check per row."""
    for chunk, row in zip(chunks, raw_rows):
        assert row["attendance_pct"] not in chunk.text
        assert f"performance_rating: {row['performance_rating']}" not in chunk.text
        assert row["leave_balance"] not in chunk.text


def test_load_is_deterministic(hr_csv_path: Path):
    first = load_hr_rows(hr_csv_path)
    second = load_hr_rows(hr_csv_path)
    assert [chunk.to_json() for chunk in first] == [chunk.to_json() for chunk in second]
