"""HR aggregates: exact slugs, values recomputed by hand here, determinism."""

from __future__ import annotations

import csv
from collections import Counter
from datetime import date
from pathlib import Path

import pytest

from finrag.ingest.hr_aggregates import REFERENCE_DATE, build_aggregates
from finrag.ingest.hr_csv import load_hr_rows

EXPECTED_SLUGS = [
    "headcount-by-department",
    "salary-by-department",
    "location-distribution",
    "tenure-by-department",
    "role-distribution",
]


@pytest.fixture(scope="module")
def hr_csv_path(repo_root: Path) -> Path:
    return repo_root / "data" / "corpus" / "hr" / "hr_data.csv"


@pytest.fixture(scope="module")
def raw_rows(hr_csv_path: Path) -> list[dict[str, str]]:
    """Read the CSV here, independently of the code under test."""
    with open(hr_csv_path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def aggregates(raw_rows):
    return build_aggregates(raw_rows)


def by_slug(aggregates, slug: str):
    for chunk in aggregates:
        if chunk.section_slug == slug:
            return chunk
    raise AssertionError(f"no aggregate with slug {slug}")


def test_five_aggregates_with_exact_slugs(aggregates):
    assert len(aggregates) == 5
    assert [chunk.section_slug for chunk in aggregates] == EXPECTED_SLUGS


def test_aggregate_metadata(aggregates):
    for chunk in aggregates:
        assert chunk.doc_id == "hr/aggregates"
        assert chunk.chunk_id == f"hr/aggregates#{chunk.section_slug}"
        assert chunk.doc_type == "aggregate"
        assert chunk.sensitivity == "internal"
        assert chunk.owner_dept == "hr"
        assert chunk.allowed_roles == ()
        assert chunk.payload == {}


def test_headcount_total_and_finance_count(aggregates, raw_rows):
    counts = Counter(row["department"] for row in raw_rows)
    text = by_slug(aggregates, "headcount-by-department").text

    assert len(raw_rows) == 100
    assert "Total employees: 100" in text
    assert f"Finance: {counts['Finance']} employees" in text
    for department, count in counts.items():
        assert f"{department}: {count} employees" in text


def test_average_salary_for_two_departments(aggregates, raw_rows):
    text = by_slug(aggregates, "salary-by-department").text

    for department in ("Finance", "Technology"):
        salaries = [
            float(row["salary"]) for row in raw_rows if row["department"] == department
        ]
        average = sum(salaries) / len(salaries)
        expected = (
            f"{department}: average salary {average:,.0f}, "
            f"lowest {min(salaries):,.0f}, highest {max(salaries):,.0f}, "
            f"across {len(salaries)} employees."
        )
        assert expected in text


def test_top_location_count(aggregates, raw_rows):
    counts = Counter(row["location"] for row in raw_rows)
    top_location, top_count = counts.most_common(1)[0]
    text = by_slug(aggregates, "location-distribution").text

    assert f"{top_location}: {top_count} employees" in text
    assert sum(counts.values()) == 100


def test_role_distribution_is_ordered_by_count(aggregates, raw_rows):
    counts = Counter(row["role"] for row in raw_rows)
    text = by_slug(aggregates, "role-distribution").text
    lines = text.split("\n")[1:]

    assert len(lines) == len(counts)
    seen = [int(line.rsplit(": ", 1)[1].split(" ")[0]) for line in lines]
    assert seen == sorted(seen, reverse=True)
    top_role, top_count = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[0]
    assert lines[0] == f"{top_role}: {top_count} employees"


def test_tenure_uses_the_fixed_reference_date(aggregates, raw_rows):
    assert REFERENCE_DATE == date(2026, 7, 29)
    text = by_slug(aggregates, "tenure-by-department").text
    assert "2026-07-29" in text

    joins = [
        date.fromisoformat(row["date_of_joining"])
        for row in raw_rows
        if row["department"] == "Finance"
    ]
    years = [(REFERENCE_DATE - joined).days / 365.25 for joined in joins]
    average = sum(years) / len(years)
    expected = f"Finance: {average:.1f} years average tenure across {len(joins)} employees."
    assert expected in text


def test_tenure_moves_with_a_different_reference_date(raw_rows):
    later = build_aggregates(raw_rows, reference_date=date(2027, 7, 29))
    now = by_slug(later, "tenure-by-department").text
    assert "2027-07-29" in now
    assert now != by_slug(build_aggregates(raw_rows), "tenure-by-department").text


def test_no_individual_pii_in_aggregate_text(aggregates, raw_rows):
    all_text = "\n".join(chunk.text for chunk in aggregates)

    assert "@fintechco.com" not in all_text
    assert "1332478" not in all_text
    for row in raw_rows:
        assert row["full_name"] not in all_text
        assert row["email"] not in all_text
        assert row["date_of_birth"] not in all_text
        assert row["employee_id"] not in all_text


def test_accepts_row_chunks_as_well_as_dicts(hr_csv_path: Path, aggregates):
    from_chunks = build_aggregates(load_hr_rows(hr_csv_path))
    assert [chunk.to_json() for chunk in from_chunks] == [
        chunk.to_json() for chunk in aggregates
    ]


def test_build_is_deterministic(raw_rows):
    first = build_aggregates(raw_rows)
    second = build_aggregates(raw_rows)
    assert [chunk.to_json() for chunk in first] == [chunk.to_json() for chunk in second]
