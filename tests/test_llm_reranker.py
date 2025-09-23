from fastapi.testclient import TestClient
import base64
from pathlib import Path

from app.main import app

client = TestClient(app)


def basic_auth(user: str, password: str):
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def stub_embed_texts(texts, model=None, api_key=None):
    # Deterministic embeddings without calling APIs
    out = []
    for t in texts:
        s = sum(ord(c) for c in t) % 97
        vec = [float((s + i * 7) % 17) for i in range(12)]
        out.append(vec)
    return out


def setup_corpus(tmp_path: Path):
    base = tmp_path / "resources" / "data"
    (base / "general").mkdir(parents=True)
    (base / "general" / "a.md").write_text("# A\nAlpha beta gamma.", encoding="utf-8")
    (base / "general" / "b.md").write_text("# B\nDelta epsilon zeta.", encoding="utf-8")
    (base / "general" / "c.md").write_text("# C\nEta theta iota.", encoding="utf-8")
    return base


def test_openai_llm_reranker_path(tmp_path, monkeypatch):
    # Monkeypatch embeddings to avoid external calls
    import app.ingest.dense_indexer as dense_indexer
    import app.main as main_mod
    import app.services.reranker_openai as rr_openai

    monkeypatch.setattr(dense_indexer, "embed_texts", stub_embed_texts)
    monkeypatch.setattr(main_mod, "embed_texts", stub_embed_texts)

    base_dir = str(setup_corpus(tmp_path))
    chroma_dir = str(tmp_path / ".chroma")

    # Reindex as c_level
    resp = client.post(
        "/admin/reindex_dense",
        json={"base_dir": base_dir, "persist_dir": chroma_dir},
        headers=basic_auth("Clark", "chief"),
    )
    assert resp.status_code == 200, resp.text

    # Baseline order
    r0 = client.post(
        "/search/hybrid",
        json={"query": "alpha", "top_k": 3, "persist_dir": chroma_dir, "base_dir": base_dir},
        headers=basic_auth("Bruce", "securepass"),
    )
    assert r0.status_code == 200, r0.text
    base_results = r0.json()["results"]
    assert len(base_results) >= 2

    # Monkeypatch the llm reranker to reverse order
    def reverse_llm(query, items, top_k=None, **kwargs):
        ordered = list(reversed(items))
        return ordered[:top_k] if top_k else ordered

    monkeypatch.setattr(rr_openai, "openai_llm_rerank", reverse_llm)

    r1 = client.post(
        "/search/hybrid",
        json={
            "query": "alpha",
            "top_k": 3,
            "persist_dir": chroma_dir,
            "base_dir": base_dir,
            "rerank": True,
            "reranker": "openai_llm",
        },
        headers=basic_auth("Bruce", "securepass"),
    )
    assert r1.status_code == 200, r1.text
    reranked = r1.json()
    assert reranked["count"] >= 2
    assert reranked["results"][0]["text"] != base_results[0]["text"]
    assert "correlation_id" in reranked
    assert reranked["metrics"]["rerank_ms"] >= 0.0

