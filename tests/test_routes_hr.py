"""
Tests for HR routes (employee data, aggregates, masked rows).
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from conftest import basic_auth, TEST_USERS


@pytest.fixture
def client():
    """Create test client."""
    from app.main import app
    return TestClient(app)


@pytest.fixture
def hr_auth():
    """Auth header for HR user."""
    return basic_auth("natasha", "hrpass123")


@pytest.fixture
def c_level_auth():
    """Auth header for c_level user."""
    return basic_auth("clark", "chief")


@pytest.fixture
def marketing_auth():
    """Auth header for marketing user (no HR access)."""
    return basic_auth("bruce", "securepass")


@pytest.fixture
def mock_hr_rows():
    """Mock HR data rows."""
    return [
        {"name": "John Doe", "email": "john@example.com", "department": "engineering", "salary": "100000", "performance_rating": "4.5"},
        {"name": "Jane Smith", "email": "jane@example.com", "department": "marketing", "salary": "95000", "performance_rating": "4.2"},
    ]


class TestHrRows:
    """Tests for /hr/rows endpoint."""

    def test_hr_rows_requires_auth(self, client):
        """HR rows endpoint requires authentication."""
        resp = client.get("/hr/rows")
        assert resp.status_code == 401

    def test_hr_rows_denied_for_marketing(self, client, marketing_auth):
        """Marketing role cannot access HR rows."""
        resp = client.get("/hr/rows", headers=marketing_auth)
        assert resp.status_code == 403

    def test_hr_rows_allowed_for_hr(self, client, hr_auth, monkeypatch, mock_hr_rows):
        """HR role can access HR rows."""
        monkeypatch.setattr("app.routes.hr.load_hr_rows", lambda **k: mock_hr_rows)

        resp = client.get("/hr/rows", headers=hr_auth)
        assert resp.status_code == 200
        data = resp.json()
        assert "rows" in data
        assert "count" in data
        assert "correlation_id" in data
        assert data["count"] == 2

    def test_hr_rows_allowed_for_c_level(self, client, c_level_auth, monkeypatch, mock_hr_rows):
        """C-level can access HR rows."""
        monkeypatch.setattr("app.routes.hr.load_hr_rows", lambda **k: mock_hr_rows)

        resp = client.get("/hr/rows", headers=c_level_auth)
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2

    def test_hr_rows_pagination(self, client, hr_auth, monkeypatch):
        """HR rows supports limit and offset."""
        def mock_load(limit=10, offset=0, **k):
            return [{"id": i} for i in range(offset, min(offset + limit, 10))]

        monkeypatch.setattr("app.routes.hr.load_hr_rows", mock_load)

        resp = client.get("/hr/rows?limit=5&offset=2", headers=hr_auth)
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 5

    def test_hr_rows_has_correlation_id(self, client, hr_auth, monkeypatch, mock_hr_rows):
        """HR rows response includes correlation ID."""
        monkeypatch.setattr("app.routes.hr.load_hr_rows", lambda **k: mock_hr_rows)

        resp = client.get("/hr/rows", headers=hr_auth)
        assert resp.status_code == 200
        assert "X-Correlation-ID" in resp.headers


class TestHrAggregate:
    """Tests for /hr/aggregate endpoint."""

    def test_hr_aggregate_requires_auth(self, client):
        """HR aggregate endpoint requires authentication."""
        resp = client.get("/hr/aggregate")
        assert resp.status_code == 401

    def test_hr_aggregate_denied_for_non_hr_when_disabled(self, client, marketing_auth, monkeypatch):
        """HR aggregate denied for non-HR users when mode is disabled.

        Note: HR and C-Level users always have access regardless of mode.
        The mode flag only controls access for non-HR/non-C-Level users.
        """
        monkeypatch.setenv("HR_AGGREGATE_MODE", "disabled")
        # Need to reset settings singleton
        from app.utils.config import reset_settings
        reset_settings()

        resp = client.get("/hr/aggregate", headers=marketing_auth)
        # Should be 403 since aggregate mode is disabled for non-HR
        assert resp.status_code == 403

    def test_hr_aggregate_has_correlation_id_on_error(self, client, marketing_auth):
        """HR aggregate includes correlation ID even on error."""
        resp = client.get("/hr/aggregate", headers=marketing_auth)
        assert resp.status_code == 403
        assert "X-Correlation-ID" in resp.headers


class TestHrRowsMasked:
    """Tests for /hr/rows_masked endpoint."""

    def test_hr_rows_masked_requires_auth(self, client):
        """HR masked rows endpoint requires authentication."""
        resp = client.get("/hr/rows_masked")
        assert resp.status_code == 401

    def test_hr_rows_masked_denied_for_non_hr_when_disabled(self, client, marketing_auth, monkeypatch):
        """HR masked rows denied for non-HR users when mode is disabled.

        Note: HR and C-Level users always have access regardless of mode.
        The mode flag only controls access for non-HR/non-C-Level users.
        """
        monkeypatch.setenv("HR_MASKED_ROWS_MODE", "disabled")
        from app.utils.config import reset_settings
        reset_settings()

        resp = client.get("/hr/rows_masked", headers=marketing_auth)
        assert resp.status_code == 403

    def test_hr_rows_masked_includes_decision(self, client, marketing_auth, monkeypatch):
        """HR masked rows response includes policy decision on denial.

        When mode is disabled, non-HR users should receive a 403 with decision info.
        """
        monkeypatch.setenv("HR_MASKED_ROWS_MODE", "disabled")
        from app.utils.config import reset_settings
        reset_settings()

        resp = client.get("/hr/rows_masked", headers=marketing_auth)
        assert resp.status_code == 403
        data = resp.json()
        assert "detail" in data
        detail = data["detail"]
        assert detail == "Access denied"
