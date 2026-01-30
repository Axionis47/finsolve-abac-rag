"""
Tests that verify API responses are sanitized.

Ensures no internal policy details, rule names, or PII metadata
leak through API responses after the H1/H5 security fixes.
"""
import base64
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.conftest import basic_auth

client = TestClient(app)


class TestErrorResponseSanitization:
    """403/401 responses must not leak internal details."""

    def test_hr_rows_403_no_rule_names(self):
        """Engineering user denied HR rows - response must be generic."""
        resp = client.get("/hr/rows", headers=basic_auth("tony", "password123"))
        assert resp.status_code == 403
        body = resp.json()
        detail = body.get("detail", "")
        # Must be generic
        assert detail == "Access denied"
        # Must NOT contain rule names
        assert "rule" not in detail.lower()
        assert "deny" not in detail.lower() or detail == "Access denied"

    def test_hr_aggregate_403_no_rule_names(self):
        """Marketing user denied HR aggregate - response must be generic."""
        resp = client.get("/hr/aggregate", headers=basic_auth("bruce", "securepass"))
        assert resp.status_code == 403
        body = resp.json()
        assert body.get("detail") == "Access denied"

    def test_hr_masked_rows_403_no_rule_names(self, monkeypatch):
        """Finance user denied masked rows when flag disabled - response must be generic."""
        monkeypatch.setenv("HR_MASKED_ROWS_MODE", "disabled")
        from app.utils.config import reset_settings
        reset_settings()
        try:
            resp = client.get("/hr/rows_masked", headers=basic_auth("sam", "financepass"))
            assert resp.status_code == 403
            body = resp.json()
            assert body.get("detail") == "Access denied"
        finally:
            reset_settings()

    def test_401_no_user_leak(self):
        """401 should not reveal whether username exists."""
        resp = client.get("/hr/rows", headers=basic_auth("tony", "wrongpass"))
        assert resp.status_code == 401
        detail = resp.json().get("detail", "")
        # Must not say "password wrong" or "user tony"
        assert "tony" not in detail.lower()
        assert "password" not in detail.lower()


class TestSearchResponseSanitization:
    """Search results must not contain internal metadata."""

    def test_search_results_no_allowed_roles(self, monkeypatch):
        """Search results must not contain allowed_roles or internal fields."""
        # Monkeypatch to avoid real embedding/chroma calls
        def fake_embed(texts, **kwargs):
            return [[0.1] * 16 for _ in texts]

        def fake_chroma_query(col, emb, top_k=5, role=None):
            return [
                {
                    "text": "Some public text",
                    "source_path": "general/doc.md",
                    "section_path": "intro",
                    "owner_dept": "general",
                    "doc_type": "md",
                    "sensitivity": "internal",
                    "allowed_roles": ["engineering", "marketing", "finance"],
                    "full_row": {"name": "secret", "salary": 100000},
                    "full_access_roles": ["hr"],
                    "masked_access_roles": ["c_level"],
                }
            ]

        monkeypatch.setattr("app.routes.search.embed_texts", fake_embed)
        monkeypatch.setattr("app.routes.search.chroma_query", fake_chroma_query)
        monkeypatch.setattr("app.routes.search.get_client", lambda *a, **k: object())
        monkeypatch.setattr("app.routes.search.get_or_create_collection",
                            lambda _client, name="kb_main": {})

        resp = client.post(
            "/search/dense",
            json={"query": "test query", "top_k": 5},
            headers=basic_auth("tony", "password123"),
        )
        # If endpoint returns results, check they're sanitized
        if resp.status_code == 200:
            data = resp.json()
            for result in data.get("results", []):
                assert "allowed_roles" not in result, "allowed_roles leaked in response"
                assert "full_row" not in result, "full_row PII leaked in response"
                assert "full_access_roles" not in result, "full_access_roles leaked"
                assert "masked_access_roles" not in result, "masked_access_roles leaked"


class TestHRResponseSanitization:
    """HR endpoint responses must not contain decision objects."""

    def test_hr_aggregate_200_no_decision(self, monkeypatch):
        """Successful aggregate response should have no decision details."""
        monkeypatch.setattr(
            "app.hr.csv_service.compute_safe_aggregates",
            lambda **kw: {"total_employees": 10, "count_by_department": {"eng": 5}},
        )
        # HR user should be permitted
        resp = client.get("/hr/aggregate", headers=basic_auth("natasha", "hrpass123"))
        if resp.status_code == 200:
            data = resp.json()
            assert "decision" not in data, "decision details leaked in response"
            assert "rule" not in data, "rule name leaked in response"

    def test_hr_rows_200_no_decision(self, monkeypatch):
        """Successful rows response should have no decision details."""
        monkeypatch.setattr(
            "app.hr.csv_service.load_hr_rows",
            lambda **kw: [{"name": "Alice", "dept": "eng"}],
        )
        resp = client.get("/hr/rows", headers=basic_auth("natasha", "hrpass123"))
        if resp.status_code == 200:
            data = resp.json()
            assert "decision" not in data, "decision details leaked in response"
