"""
Edge case tests for HR CSV service.

Tests probe:
- Missing/corrupted CSV files
- Offset/limit boundary conditions
- Aggregation with degenerate data
- Masking with unusual field names
"""
import os
import pytest

from app.hr.csv_service import load_hr_rows, compute_safe_aggregates, mask_row


class TestLoadRowsEdgeCases:
    """load_hr_rows boundary conditions."""

    def test_missing_csv_raises(self):
        """Missing CSV file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_hr_rows(csv_path="/nonexistent/path/hr_data.csv")

    def test_offset_beyond_file(self, tmp_path):
        """Offset > total rows returns empty list."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,email\nAlice,alice@co.com\nBob,bob@co.com\n")
        rows = load_hr_rows(csv_path=str(csv_file), offset=999)
        assert rows == []

    def test_limit_zero(self, tmp_path):
        """limit=0 returns empty list."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,email\nAlice,alice@co.com\n")
        rows = load_hr_rows(csv_path=str(csv_file), limit=0)
        assert rows == []

    def test_offset_zero_limit_one(self, tmp_path):
        """Standard pagination: first row only."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,email\nAlice,alice@co.com\nBob,bob@co.com\n")
        rows = load_hr_rows(csv_path=str(csv_file), offset=0, limit=1)
        assert len(rows) == 1
        assert rows[0]["name"] == "Alice"

    def test_empty_csv_headers_only(self, tmp_path):
        """CSV with headers but no data rows returns empty."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,email,department,salary,performance_rating\n")
        rows = load_hr_rows(csv_path=str(csv_file))
        assert rows == []


class TestAggregateEdgeCases:
    """compute_safe_aggregates boundary conditions."""

    def test_empty_csv(self, tmp_path):
        """Headers-only CSV should return zero counts."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("name,email,department,salary,performance_rating\n")
        agg = compute_safe_aggregates(csv_path=str(csv_file))
        assert agg["total_employees"] == 0
        assert agg["count_by_department"] == {}
        assert agg["average_salary_by_department"] == {}

    def test_non_numeric_salary(self, tmp_path):
        """Non-numeric salary should be silently skipped."""
        csv_file = tmp_path / "bad_salary.csv"
        csv_file.write_text(
            "name,email,department,salary,performance_rating\n"
            "Alice,a@co.com,eng,not_a_number,4.0\n"
            "Bob,b@co.com,eng,100000,4.5\n"
        )
        agg = compute_safe_aggregates(csv_path=str(csv_file))
        assert agg["total_employees"] == 2
        # Only Bob's salary should be counted
        assert agg["average_salary_by_department"]["eng"] == 100000.0

    def test_missing_columns(self, tmp_path):
        """CSV missing salary and rating columns should not crash."""
        csv_file = tmp_path / "minimal.csv"
        csv_file.write_text("name,department\nAlice,eng\nBob,marketing\n")
        agg = compute_safe_aggregates(csv_path=str(csv_file))
        assert agg["total_employees"] == 2
        assert agg["count_by_department"] == {"eng": 1, "marketing": 1}
        # No salary/rating data
        assert agg["average_salary_by_department"] == {}
        assert agg["average_performance_rating_by_department"] == {}

    def test_missing_department_uses_unknown(self, tmp_path):
        """Rows without department should be grouped as 'Unknown'."""
        csv_file = tmp_path / "no_dept.csv"
        csv_file.write_text("name,salary\nAlice,100000\n")
        agg = compute_safe_aggregates(csv_path=str(csv_file))
        assert "Unknown" in agg["count_by_department"]


class TestMaskRowEdgeCases:
    """mask_row boundary conditions."""

    def test_empty_dict(self):
        """Empty row dict should return empty dict."""
        result = mask_row({})
        assert result == {}

    def test_no_pii_fields(self):
        """Row with no PII fields should be returned unchanged."""
        row = {"department": "eng", "salary": "100000"}
        result = mask_row(row)
        assert result == row

    def test_custom_field_not_masked(self):
        """Fields not in the PII list should not be masked."""
        row = {"custom_id": "12345", "department": "eng"}
        result = mask_row(row)
        assert result["custom_id"] == "12345"

    def test_mixed_case_field_masked(self):
        """PII field matching is case-insensitive (k.lower())."""
        row = {"EMAIL": "alice@co.com", "Name": "Alice", "department": "eng"}
        result = mask_row(row)
        assert result["EMAIL"] == "[REDACTED]"
        assert result["Name"] == "[REDACTED]"
        assert result["department"] == "eng"

    def test_original_row_not_mutated(self):
        """mask_row should return a copy, not mutate the original."""
        row = {"name": "Alice", "email": "alice@co.com"}
        result = mask_row(row)
        assert result["name"] == "[REDACTED]"
        assert row["name"] == "Alice"  # Original unchanged

    def test_full_name_masked(self):
        """full_name field (used by HR CSV) must be redacted."""
        row = {"full_name": "Aadhya Patel", "department": "eng", "salary": "100000"}
        result = mask_row(row)
        assert result["full_name"] == "[REDACTED]"
        assert result["department"] == "eng"

    def test_all_pii_fields_masked(self):
        """All known PII fields should be redacted."""
        row = {
            "name": "Alice",
            "full_name": "Alice Smith",
            "first_name": "Alice",
            "last_name": "Smith",
            "email": "a@co.com",
            "phone": "555-1234",
            "address": "123 Main St",
            "ssn": "123-45-6789",
            "dob": "1990-01-01",
            "employee_id": "E001",
            "department": "eng",
            "salary": "100000",
        }
        result = mask_row(row)
        for field in ["name", "full_name", "first_name", "last_name", "email", "phone",
                       "address", "ssn", "dob", "employee_id"]:
            assert result[field] == "[REDACTED]", f"{field} should be redacted"
        # Non-PII preserved
        assert result["department"] == "eng"
        assert result["salary"] == "100000"
