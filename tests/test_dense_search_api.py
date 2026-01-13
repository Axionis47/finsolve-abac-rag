from fastapi.testclient import TestClient
from pathlib import Path
import base64
import pytest

from app.main import app

client = TestClient(app)


def basic_auth(user: str, password: str):
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def stub_embed_texts(texts, use_cache=True, **kwargs):
    # Deterministic tiny embeddings without network
    out = []
    for t in texts:
        s = sum(ord(c) for c in t) % 97
        vec = [float((s + i * 3) % 13) for i in range(16)]
        out.append(vec)
    return out


def setup_temp_corpus(tmp_path):
    base = tmp_path / "resources" / "data"
    (base / "general").mkdir(parents=True)
    (base / "finance").mkdir(parents=True)

    (base / "general" / "gen.md").write_text("""
# General
Company growth and culture are important.
""".strip(), encoding="utf-8")

    (base / "finance" / "fin.md").write_text("""
# Finance
Budget planning for next quarter.
""".strip(), encoding="utf-8")
    return base


# Skip these tests if chromadb is not installed
pytest_plugins = []
try:
    import chromadb
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False


@pytest.mark.skipif(not HAS_CHROMADB, reason="chromadb not installed")
def test_dense_reindex_and_search_with_authz(tmp_path, monkeypatch):
    # Monkeypatch embeddings in indexer and routes
    import app.ingest.dense_indexer as dense_indexer
    import app.routes.chat as chat_mod
    import app.routes.search as search_mod
    import app.services.reranker_openai as reranker_mod

    monkeypatch.setattr(dense_indexer, "embed_texts", stub_embed_texts)
    monkeypatch.setattr(chat_mod, "embed_texts", stub_embed_texts)
    monkeypatch.setattr(search_mod, "embed_texts", stub_embed_texts)
    monkeypatch.setattr(reranker_mod, "embed_texts", stub_embed_texts)

    base_dir = str(setup_temp_corpus(tmp_path))
    chroma_dir = str(tmp_path / ".chroma")

    # Reindex dense as c_level
    resp = client.post(
        "/admin/reindex_dense",
        json={"base_dir": base_dir, "persist_dir": chroma_dir},
        headers=basic_auth("Clark", "chief"),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["count"] > 0

    # Marketing can find only general, not finance
    r1 = client.post(
        "/search/dense",
        json={"query": "growth", "top_k": 5, "persist_dir": chroma_dir},
        headers=basic_auth("Bruce", "securepass"),
    )
    assert r1.status_code == 200, r1.text
    data1 = r1.json()
    assert data1["count"] >= 1
    assert all(it["owner_dept"] == "general" for it in data1["results"])  # only general for marketing

    r2 = client.post(
        "/search/dense",
        json={"query": "budget", "top_k": 5, "persist_dir": chroma_dir},
        headers=basic_auth("Bruce", "securepass"),
    )
    assert r2.status_code == 200
    # Marketing should not see finance-owned docs
    assert all(it["owner_dept"] != "finance" for it in r2.json().get("results", []))

    # Finance user can see finance content
    r3 = client.post(
        "/search/dense",
        json={"query": "budget", "top_k": 5, "persist_dir": chroma_dir},
        headers=basic_auth("Sam", "financepass"),
    )
    assert r3.status_code == 200
    assert any(it["owner_dept"] == "finance" for it in r3.json()["results"])

