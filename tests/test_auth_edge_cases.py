"""
Edge case tests for authentication and authorization.

Tests probe boundary conditions in the auth flow:
- Empty/invalid credentials
- Case sensitivity
- Injection attempts
- Resource exhaustion
"""
import base64

import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.conftest import basic_auth

client = TestClient(app)


class TestAuthEdgeCases:
    """Authentication boundary tests."""

    def test_empty_username(self):
        """Empty username returns 401."""
        resp = client.get("/health", headers=basic_auth("", "password123"))
        # /health may not require auth; use an auth-required endpoint
        resp = client.get("/hr/rows", headers=basic_auth("", "password123"))
        assert resp.status_code == 401

    def test_empty_password(self):
        """Empty password returns 401."""
        resp = client.get("/hr/rows", headers=basic_auth("tony", ""))
        assert resp.status_code == 401

    def test_case_insensitive_username_upper(self):
        """Uppercase username should authenticate (deps.py lowercases)."""
        resp = client.get("/hr/rows", headers=basic_auth("TONY", "password123"))
        # Tony is engineering, HR rows should be 403 but auth should pass (not 401)
        assert resp.status_code in (200, 403), f"Expected 200 or 403, got {resp.status_code}"

    def test_case_insensitive_username_mixed(self):
        """Mixed-case username should authenticate."""
        resp = client.get("/hr/rows", headers=basic_auth("Tony", "password123"))
        assert resp.status_code in (200, 403), f"Expected 200 or 403, got {resp.status_code}"

    def test_nonexistent_username(self):
        """Unknown username returns 401."""
        resp = client.get("/hr/rows", headers=basic_auth("nonexistent_user_xyz", "pass"))
        assert resp.status_code == 401

    def test_correct_user_wrong_password(self):
        """Valid user with wrong password returns 401."""
        resp = client.get("/hr/rows", headers=basic_auth("tony", "wrongpassword"))
        assert resp.status_code == 401

    def test_no_auth_header(self):
        """Missing Authorization header returns 401."""
        resp = client.get("/hr/rows")
        assert resp.status_code == 401

    def test_malformed_auth_header(self):
        """Garbage Authorization header returns 401."""
        resp = client.get("/hr/rows", headers={"Authorization": "Basic not-valid-base64!!!"})
        assert resp.status_code in (401, 400)

    def test_sql_injection_in_username(self):
        """SQL injection payload in username returns 401, no crash."""
        resp = client.get("/hr/rows", headers=basic_auth("' OR 1=1 --", "pass"))
        assert resp.status_code == 401

    def test_xss_in_username(self):
        """XSS payload in username returns 401, no crash."""
        resp = client.get("/hr/rows", headers=basic_auth("<script>alert(1)</script>", "pass"))
        assert resp.status_code == 401

    def test_very_long_username(self):
        """10KB username returns 401, no crash or hang."""
        long_user = "a" * 10_000
        resp = client.get("/hr/rows", headers=basic_auth(long_user, "pass"))
        assert resp.status_code == 401

    def test_unicode_username(self):
        """Unicode username returns 401, no crash."""
        resp = client.get("/hr/rows", headers=basic_auth("tøny", "password123"))
        assert resp.status_code == 401

    def test_401_response_has_no_internal_details(self):
        """401 response should not leak user database info."""
        resp = client.get("/hr/rows", headers=basic_auth("tony", "wrongpass"))
        assert resp.status_code == 401
        body = resp.json()
        detail = body.get("detail", "")
        # Should be generic, not "user exists but password wrong"
        assert "password" not in detail.lower()
        assert "tony" not in detail.lower()
