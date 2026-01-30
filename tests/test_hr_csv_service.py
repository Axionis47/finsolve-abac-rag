"""
Tests for HR CSV service (loading, aggregates, masking).
"""
import os
import pytest
import tempfile
from pathlib import Path

from app.hr.csv_service import load_hr_rows, compute_safe_aggregates, mask_row


@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample HR CSV file for testing."""
    csv_content = """name,email,department,salary,performance_rating
John Doe,john@example.com,engineering,100000,4.5
Jane Smith,jane@example.com,marketing,95000,4.2
Bob Wilson,bob@example.com,engineering,110000,4.8
Alice Brown,alice@example.com,hr,85000,4.0
"""
    csv_path = tmp_path / "hr_test.csv"
    csv_path.write_text(csv_content)
    return str(csv_path)


@pytest.fixture
def empty_csv(tmp_path):
    """Create an empty HR CSV file (header only)."""
    csv_content = "name,email,department,salary,performance_rating\n"
    csv_path = tmp_path / "empty_hr.csv"
    csv_path.write_text(csv_content)
    return str(csv_path)


class TestLoadHrRows:
    """Tests for load_hr_rows function."""

    def test_loads_all_rows(self, sample_csv):
        """Loads all rows from CSV."""
        rows = load_hr_rows(csv_path=sample_csv, limit=100)
        assert len(rows) == 4

    def test_respects_limit(self, sample_csv):
        """Respects the limit parameter."""
        rows = load_hr_rows(csv_path=sample_csv, limit=2)
        assert len(rows) == 2
        assert rows[0]["name"] == "John Doe"
        assert rows[1]["name"] == "Jane Smith"

    def test_respects_offset(self, sample_csv):
        """Respects the offset parameter."""
        rows = load_hr_rows(csv_path=sample_csv, limit=100, offset=2)
        assert len(rows) == 2
        assert rows[0]["name"] == "Bob Wilson"
        assert rows[1]["name"] == "Alice Brown"

    def test_limit_and_offset_together(self, sample_csv):
        """Limit and offset work together correctly."""
        rows = load_hr_rows(csv_path=sample_csv, limit=1, offset=1)
        assert len(rows) == 1
        assert rows[0]["name"] == "Jane Smith"

    def test_returns_dict_rows(self, sample_csv):
        """Returns rows as dictionaries with all fields."""
        rows = load_hr_rows(csv_path=sample_csv, limit=1)
        row = rows[0]
        assert isinstance(row, dict)
        assert "name" in row
        assert "email" in row
        assert "department" in row
        assert "salary" in row
        assert "performance_rating" in row

    def test_empty_csv_returns_empty_list(self, empty_csv):
        """Returns empty list for CSV with only header."""
        rows = load_hr_rows(csv_path=empty_csv)
        assert rows == []

    def test_offset_beyond_rows(self, sample_csv):
        """Offset beyond row count returns empty list."""
        rows = load_hr_rows(csv_path=sample_csv, offset=100)
        assert rows == []

    def test_raises_on_missing_file(self, tmp_path):
        """Raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_hr_rows(csv_path=str(tmp_path / "nonexistent.csv"))


class TestComputeSafeAggregates:
    """Tests for compute_safe_aggregates function."""

    def test_counts_total_employees(self, sample_csv):
        """Counts total employees correctly."""
        agg = compute_safe_aggregates(csv_path=sample_csv)
        assert agg["total_employees"] == 4

    def test_counts_by_department(self, sample_csv):
        """Counts employees by department."""
        agg = compute_safe_aggregates(csv_path=sample_csv)
        dept_counts = agg["count_by_department"]
        assert dept_counts["engineering"] == 2
        assert dept_counts["marketing"] == 1
        assert dept_counts["hr"] == 1

    def test_computes_average_salary_by_department(self, sample_csv):
        """Computes average salary per department."""
        agg = compute_safe_aggregates(csv_path=sample_csv)
        avg_salary = agg["average_salary_by_department"]
        # engineering: (100000 + 110000) / 2 = 105000
        assert avg_salary["engineering"] == 105000.0
        # marketing: 95000 / 1 = 95000
        assert avg_salary["marketing"] == 95000.0
        # hr: 85000 / 1 = 85000
        assert avg_salary["hr"] == 85000.0

    def test_computes_average_rating_by_department(self, sample_csv):
        """Computes average performance rating per department."""
        agg = compute_safe_aggregates(csv_path=sample_csv)
        avg_rating = agg["average_performance_rating_by_department"]
        # engineering: (4.5 + 4.8) / 2 = 4.65
        assert avg_rating["engineering"] == pytest.approx(4.65, rel=0.01)
        assert avg_rating["marketing"] == 4.2
        assert avg_rating["hr"] == 4.0

    def test_empty_csv_returns_zeros(self, empty_csv):
        """Returns zeros for empty CSV."""
        agg = compute_safe_aggregates(csv_path=empty_csv)
        assert agg["total_employees"] == 0
        assert agg["count_by_department"] == {}
        assert agg["average_salary_by_department"] == {}

    def test_handles_missing_salary(self, tmp_path):
        """Handles rows with missing salary values."""
        csv_content = """name,email,department,salary,performance_rating
John,john@test.com,eng,,4.5
Jane,jane@test.com,eng,50000,4.0
"""
        csv_path = tmp_path / "partial.csv"
        csv_path.write_text(csv_content)

        agg = compute_safe_aggregates(csv_path=str(csv_path))
        # Only Jane has valid salary; John's empty salary is skipped
        assert agg["average_salary_by_department"]["eng"] == 50000.0

    def test_handles_invalid_salary(self, tmp_path):
        """Handles rows with non-numeric salary."""
        csv_content = """name,email,department,salary,performance_rating
John,john@test.com,eng,not_a_number,4.5
Jane,jane@test.com,eng,50000,4.0
"""
        csv_path = tmp_path / "invalid.csv"
        csv_path.write_text(csv_content)

        # Should not raise, just skip invalid row
        agg = compute_safe_aggregates(csv_path=str(csv_path))
        assert "eng" in agg["average_salary_by_department"]


class TestMaskRow:
    """Tests for mask_row function."""

    def test_masks_name_field(self):
        """Masks the name field."""
        row = {"name": "John Doe", "department": "eng"}
        masked = mask_row(row)
        assert masked["name"] == "[REDACTED]"
        assert masked["department"] == "eng"

    def test_masks_email_field(self):
        """Masks the email field."""
        row = {"email": "john@example.com", "role": "engineer"}
        masked = mask_row(row)
        assert masked["email"] == "[REDACTED]"
        assert masked["role"] == "engineer"

    def test_masks_ssn_field(self):
        """Masks the SSN field."""
        row = {"ssn": "123-45-6789", "dept": "hr"}
        masked = mask_row(row)
        assert masked["ssn"] == "[REDACTED]"

    def test_masks_multiple_pii_fields(self):
        """Masks all PII fields in a row."""
        row = {
            "name": "John",
            "email": "john@test.com",
            "phone": "555-1234",
            "address": "123 Main St",
            "department": "eng",
            "salary": "100000",
        }
        masked = mask_row(row)
        assert masked["name"] == "[REDACTED]"
        assert masked["email"] == "[REDACTED]"
        assert masked["phone"] == "[REDACTED]"
        assert masked["address"] == "[REDACTED]"
        # Non-PII fields preserved
        assert masked["department"] == "eng"
        assert masked["salary"] == "100000"

    def test_case_insensitive_masking(self):
        """Masking is case-insensitive."""
        row = {"NAME": "John", "Email": "test@test.com", "PHONE": "555"}
        masked = mask_row(row)
        assert masked["NAME"] == "[REDACTED]"
        assert masked["Email"] == "[REDACTED]"
        assert masked["PHONE"] == "[REDACTED]"

    def test_does_not_modify_original(self):
        """Masking does not modify the original row."""
        row = {"name": "John", "dept": "eng"}
        masked = mask_row(row)
        assert row["name"] == "John"  # Original unchanged
        assert masked["name"] == "[REDACTED]"

    def test_masks_common_pii_variants(self):
        """Masks common PII field name variants."""
        row = {
            "first_name": "John",
            "last_name": "Doe",
            "dob": "1990-01-01",
            "date_of_birth": "1990-01-01",
            "national_id": "ABC123",
            "employee_id": "EMP001",
        }
        masked = mask_row(row)
        for key in row:
            assert masked[key] == "[REDACTED]"

    def test_empty_row(self):
        """Handles empty row."""
        row = {}
        masked = mask_row(row)
        assert masked == {}

    def test_strict_level(self):
        """Strict level masks all standard PII."""
        row = {"name": "John", "salary": "100000"}
        masked = mask_row(row, level="strict")
        assert masked["name"] == "[REDACTED]"
        assert masked["salary"] == "100000"  # Salary not considered PII in mask_row
