from fastapi.testclient import TestClient
import base64
from pathlib import Path
import pytest

from app.main import app

client = TestClient(app)


def basic_auth(user: str, password: str):
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def stub_embed_texts(texts, model=None, api_key=None):
    # Deterministic tiny embeddings without network
    out = []
    for t in texts:
        s = sum(ord(c) for c in t) % 97
        vec = [float((s + i * 3) % 13) for i in range(16)]
        out.append(vec)
    return out


def setup_temp_corpus(tmp_path: Path):
    base = tmp_path / "resources" / "data"
    (base / "general").mkdir(parents=True)

    (base / "general" / "gen1.md").write_text(
        """
# General Doc 1
Company growth and culture are important.
""".strip(),
        encoding="utf-8",
    )
    (base / "general" / "gen2.md").write_text(
        """
# General Doc 2
Quarterly planning and budgets.
""".strip(),
        encoding="utf-8",
    )
    (base / "general" / "gen3.md").write_text(
        """
# General Doc 3
Hiring, onboarding, and HR policies.
""".strip(),
        encoding="utf-8",
    )
    return base


# Skip these tests if chromadb is not installed
try:
    import chromadb
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False


@pytest.mark.skipif(not HAS_CHROMADB, reason="chromadb not installed")
def test_openai_embedding_reranker_path(tmp_path, monkeypatch):
    # Monkeypatch embeddings in indexer and routes
    import app.ingest.dense_indexer as dense_indexer
    import app.routes.chat as chat_mod
    import app.routes.search as search_mod
    import app.services.reranker_openai as rr_openai

    monkeypatch.setattr(dense_indexer, "embed_texts", stub_embed_texts)
    monkeypatch.setattr(chat_mod, "embed_texts", stub_embed_texts)
    monkeypatch.setattr(search_mod, "embed_texts", stub_embed_texts)
    monkeypatch.setattr(rr_openai, "embed_texts", stub_embed_texts)

    base_dir = str(setup_temp_corpus(tmp_path))
    chroma_dir = str(tmp_path / ".chroma")

    # Reindex dense as c_level
    resp = client.post(
        "/admin/reindex_dense",
        json={"base_dir": base_dir, "persist_dir": chroma_dir},
        headers=basic_auth("Clark", "chief"),
    )
    assert resp.status_code == 200, resp.text

    # Baseline dense results (no rerank)
    r0 = client.post(
        "/search/dense",
        json={"query": "general", "top_k": 3, "persist_dir": chroma_dir},
        headers=basic_auth("Bruce", "securepass"),
    )
    assert r0.status_code == 200, r0.text
    base_results = r0.json()["results"]
    assert len(base_results) >= 2

    # Monkeypatch the openai reranker to reverse the order - must patch in routes module too
    def reverse_rerank(query, items, top_k=None, **kwargs):
        ordered = list(reversed(items))
        return ordered[:top_k] if top_k else ordered

    monkeypatch.setattr(rr_openai, "openai_embedding_rerank", reverse_rerank)
    # Also patch in routes.search and routes.chat where it's imported
    import app.routes.search as search_mod
    import app.routes.chat as chat_mod
    monkeypatch.setattr(search_mod, "openai_embedding_rerank", reverse_rerank)
    monkeypatch.setattr(chat_mod, "openai_embedding_rerank", reverse_rerank)

    # With rerank=OpenAI -> order should change
    r1 = client.post(
        "/search/dense",
        json={"query": "general", "top_k": 3, "persist_dir": chroma_dir, "rerank": True, "reranker": "openai"},
        headers=basic_auth("Bruce", "securepass"),
    )
    assert r1.status_code == 200, r1.text
    reranked = r1.json()
    assert reranked["count"] >= 2
    # First result should differ when reversed
    assert reranked["results"][0]["text"] != base_results[0]["text"]
    assert "metrics" in reranked and reranked["metrics"].get("rerank_ms", 0) >= 0.0

