"""
Cross-role security tests.

Verifies that role-based access boundaries are enforced across
all HR and admin endpoints for every role in the system.
"""
import base64
import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.conftest import basic_auth, TEST_USERS

client = TestClient(app)

# Roles that should NOT have HR row access
NON_HR_ROLES = [
    ("tony", "password123", "engineering"),
    ("bruce", "securepass", "marketing"),
    ("sam", "financepass", "finance"),
    ("peter", "pete123", "engineering"),
    ("sid", "sidpass123", "marketing"),
]

# Roles that SHOULD have HR row access
HR_ROLES = [
    ("natasha", "hrpass123", "hr"),
    ("clark", "chief", "c_level"),
]

# Non-admin roles (not c_level)
NON_ADMIN_ROLES = [
    ("tony", "password123", "engineering"),
    ("bruce", "securepass", "marketing"),
    ("sam", "financepass", "finance"),
    ("natasha", "hrpass123", "hr"),
    ("peter", "pete123", "engineering"),
    ("sid", "sidpass123", "marketing"),
]


class TestHRRowAccessControl:
    """Every non-HR/non-c_level role must be denied HR rows."""

    @pytest.mark.parametrize("username,password,role", NON_HR_ROLES)
    def test_non_hr_denied_hr_rows(self, username, password, role):
        """Role '{role}' ({username}) must be denied /hr/rows."""
        resp = client.get("/hr/rows", headers=basic_auth(username, password))
        assert resp.status_code == 403, (
            f"{role} user '{username}' got {resp.status_code}, expected 403"
        )

    @pytest.mark.parametrize("username,password,role", HR_ROLES)
    def test_hr_and_c_level_can_access_hr_rows(self, username, password, role):
        """Role '{role}' ({username}) must be allowed /hr/rows."""
        resp = client.get("/hr/rows", headers=basic_auth(username, password))
        assert resp.status_code == 200, (
            f"{role} user '{username}' got {resp.status_code}, expected 200"
        )


class TestHRAggregateAccessControl:
    """Non-HR roles denied aggregate when flag is disabled (default)."""

    @pytest.mark.parametrize("username,password,role", NON_HR_ROLES)
    def test_non_hr_denied_aggregate_by_default(self, username, password, role):
        """Role '{role}' must be denied /hr/aggregate when flag is disabled."""
        resp = client.get("/hr/aggregate", headers=basic_auth(username, password))
        assert resp.status_code == 403, (
            f"{role} user '{username}' got {resp.status_code}, expected 403"
        )

    def test_hr_can_access_aggregate(self):
        """HR role should always access aggregates."""
        resp = client.get("/hr/aggregate", headers=basic_auth("natasha", "hrpass123"))
        assert resp.status_code == 200

    def test_c_level_can_access_aggregate(self):
        """C-level role should always access aggregates."""
        resp = client.get("/hr/aggregate", headers=basic_auth("clark", "chief"))
        assert resp.status_code == 200


class TestAdminAccessControl:
    """Admin endpoints restricted to c_level only."""

    @pytest.mark.parametrize("username,password,role", NON_ADMIN_ROLES)
    def test_non_admin_denied_reindex(self, username, password, role):
        """Role '{role}' must be denied /admin/reindex."""
        resp = client.post("/admin/reindex", headers=basic_auth(username, password))
        assert resp.status_code == 403, (
            f"{role} user '{username}' got {resp.status_code}, expected 403"
        )

    @pytest.mark.parametrize("username,password,role", NON_ADMIN_ROLES)
    def test_non_admin_denied_reindex_dense(self, username, password, role):
        """Role '{role}' must be denied /admin/reindex_dense."""
        resp = client.post("/admin/reindex_dense", headers=basic_auth(username, password))
        assert resp.status_code == 403, (
            f"{role} user '{username}' got {resp.status_code}, expected 403"
        )


class TestCrossRoleDeptBoundary:
    """Cross-department access boundaries via PDP."""

    def test_engineering_cannot_view_hr_rows(self):
        resp = client.get("/hr/rows", headers=basic_auth("tony", "password123"))
        assert resp.status_code == 403

    def test_finance_cannot_view_hr_rows(self):
        resp = client.get("/hr/rows", headers=basic_auth("sam", "financepass"))
        assert resp.status_code == 403

    def test_marketing_cannot_view_hr_rows(self):
        resp = client.get("/hr/rows", headers=basic_auth("bruce", "securepass"))
        assert resp.status_code == 403

    def test_c_level_can_view_all(self):
        """C-level should access HR rows."""
        resp = client.get("/hr/rows", headers=basic_auth("clark", "chief"))
        assert resp.status_code == 200
