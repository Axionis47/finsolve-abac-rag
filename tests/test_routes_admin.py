"""
Tests for admin routes (status, reindex, cache management).
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from conftest import basic_auth, TEST_USERS


@pytest.fixture
def client():
    """Create test client."""
    from app.main import app
    return TestClient(app)


@pytest.fixture
def c_level_auth():
    """Auth header for c_level user."""
    return basic_auth("clark", "chief")


@pytest.fixture
def non_admin_auth():
    """Auth header for non-admin user."""
    return basic_auth("bruce", "securepass")


class TestAdminStatus:
    """Tests for /admin/status endpoint."""

    def test_status_requires_auth(self, client):
        """Status endpoint requires authentication."""
        resp = client.get("/admin/status")
        assert resp.status_code == 401

    def test_status_requires_c_level(self, client, non_admin_auth):
        """Status endpoint requires c_level role."""
        resp = client.get("/admin/status", headers=non_admin_auth)
        assert resp.status_code == 403
        assert "c_level" in resp.json().get("detail", "").lower()

    def test_status_returns_counts(self, client, c_level_auth, monkeypatch):
        """Status endpoint returns sparse and dense counts."""
        # Mock external dependencies
        monkeypatch.setattr(
            "app.routes.admin.get_model_metadata",
            lambda *a, **k: (200, {"id": "text-embedding-3-small"})
        )

        resp = client.get("/admin/status", headers=c_level_auth)
        assert resp.status_code == 200
        data = resp.json()

        # Check required fields
        assert "sparse_count" in data
        assert "dense_count" in data
        assert "persist_dir" in data
        assert "openai" in data
        assert "cache" in data
        assert "rag_cache" in data
        assert "correlation_id" in data

    def test_status_includes_correlation_id_header(self, client, c_level_auth, monkeypatch):
        """Status response includes X-Correlation-ID header."""
        monkeypatch.setattr(
            "app.routes.admin.get_model_metadata",
            lambda *a, **k: (200, {"id": "test"})
        )

        resp = client.get("/admin/status", headers=c_level_auth)
        assert resp.status_code == 200
        assert "X-Correlation-ID" in resp.headers


class TestAdminReindex:
    """Tests for /admin/reindex endpoint."""

    def test_reindex_requires_auth(self, client):
        """Reindex endpoint requires authentication."""
        resp = client.post("/admin/reindex")
        assert resp.status_code == 401

    def test_reindex_requires_c_level(self, client, non_admin_auth):
        """Reindex endpoint requires c_level role."""
        resp = client.post("/admin/reindex", headers=non_admin_auth)
        assert resp.status_code == 403

    def test_reindex_rebuilds_index(self, client, c_level_auth, monkeypatch):
        """Reindex rebuilds the sparse index."""
        # Mock build_index to avoid filesystem access
        mock_index = [{"id": 1}, {"id": 2}]
        monkeypatch.setattr("app.routes.admin.build_index", lambda **k: mock_index)

        resp = client.post("/admin/reindex", headers=c_level_auth)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["count"] == 2


class TestAdminReindexDense:
    """Tests for /admin/reindex_dense endpoint."""

    def test_reindex_dense_requires_auth(self, client):
        """Reindex dense endpoint requires authentication."""
        resp = client.post("/admin/reindex_dense")
        assert resp.status_code == 401

    def test_reindex_dense_requires_c_level(self, client, non_admin_auth):
        """Reindex dense endpoint requires c_level role."""
        resp = client.post("/admin/reindex_dense", headers=non_admin_auth)
        assert resp.status_code == 403

    def test_reindex_dense_rebuilds_chroma(self, client, c_level_auth, tmp_path, monkeypatch):
        """Reindex dense rebuilds Chroma index."""
        # Mock index_corpus_to_chroma
        monkeypatch.setattr(
            "app.routes.admin.index_corpus_to_chroma",
            lambda **k: 5
        )

        resp = client.post(
            "/admin/reindex_dense",
            json={"base_dir": str(tmp_path), "persist_dir": str(tmp_path / ".chroma")},
            headers=c_level_auth
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["count"] == 5


class TestAdminCacheClear:
    """Tests for /admin/cache/clear endpoint."""

    def test_cache_clear_requires_auth(self, client):
        """Cache clear endpoint requires authentication."""
        resp = client.post("/admin/cache/clear")
        assert resp.status_code == 401

    def test_cache_clear_requires_c_level(self, client, non_admin_auth):
        """Cache clear endpoint requires c_level role."""
        resp = client.post("/admin/cache/clear", headers=non_admin_auth)
        assert resp.status_code == 403

    def test_cache_clear_clears_cache(self, client, c_level_auth):
        """Cache clear actually clears the cache."""
        resp = client.post("/admin/cache/clear", headers=c_level_auth)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "cleared" in data
        assert "correlation_id" in data


class TestAdminRagCacheClear:
    """Tests for /admin/rag_cache/clear endpoint."""

    def test_rag_cache_clear_requires_auth(self, client):
        """RAG cache clear endpoint requires authentication."""
        resp = client.post("/admin/rag_cache/clear")
        assert resp.status_code == 401

    def test_rag_cache_clear_requires_c_level(self, client, non_admin_auth):
        """RAG cache clear endpoint requires c_level role."""
        resp = client.post("/admin/rag_cache/clear", headers=non_admin_auth)
        assert resp.status_code == 403

    def test_rag_cache_clear_clears_cache(self, client, c_level_auth):
        """RAG cache clear actually clears the cache."""
        resp = client.post("/admin/rag_cache/clear", headers=c_level_auth)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "cleared" in data


class TestAdminRagCacheStats:
    """Tests for /admin/rag_cache/stats endpoint."""

    def test_rag_cache_stats_requires_auth(self, client):
        """RAG cache stats endpoint requires authentication."""
        resp = client.get("/admin/rag_cache/stats")
        assert resp.status_code == 401

    def test_rag_cache_stats_requires_c_level(self, client, non_admin_auth):
        """RAG cache stats endpoint requires c_level role."""
        resp = client.get("/admin/rag_cache/stats", headers=non_admin_auth)
        assert resp.status_code == 403

    def test_rag_cache_stats_returns_stats(self, client, c_level_auth):
        """RAG cache stats returns cache statistics."""
        resp = client.get("/admin/rag_cache/stats", headers=c_level_auth)
        assert resp.status_code == 200
        data = resp.json()
        assert "stats" in data
        assert "correlation_id" in data
