from fastapi.testclient import TestClient
import base64
import json

from app.main import app


client = TestClient(app)


def _basic_auth(username: str, password: str) -> dict:
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def test_e2e_chat_marketing_simple(monkeypatch):
    # Monkeypatch OpenAI/Chroma-dependent pieces to avoid network and native deps
    def fake_embed_texts(texts, model=None, api_key=None):
        return [[0.01] * 16 for _ in texts]

    def fake_generate_answer(query, snippets, model: str, api_key: str):
        if not snippets:
            return f"NO_CONTEXT for: {query}\n\nCitations: []"
        src = f"{snippets[0].get('source_path','')}#{snippets[0].get('section_path','')}".rstrip('#')
        return f"ANSWER for: {query}. See [1].\n\nCitations: [1] {src}"

    def fake_chroma_query(collection, q_emb, top_k=5, role=None):
        return []

    # Apply monkeypatches against app.main namespace
    monkeypatch.setattr("app.main.embed_texts", fake_embed_texts)
    monkeypatch.setattr("app.main.chroma_query", fake_chroma_query)
    monkeypatch.setattr("app.main.generate_answer", fake_generate_answer)
    # Avoid importing/using real chromadb client in CI
    monkeypatch.setattr("app.main.get_client", lambda *a, **k: object())
    monkeypatch.setattr("app.main.get_or_create_collection", lambda _client, name="kb_main": {})

    payload = {
        "message": "growth strategy",
        "top_k": 3,
        "rerank": False,
    }
    res = client.post("/chat", headers=_basic_auth("Bruce", "securepass"), json=payload)
    assert res.status_code == 200, res.text
    data = res.json()
    assert "answer" in data and "metrics" in data and "citations" in data
    assert "Correlation-ID" in res.headers.get("X-Correlation-ID", "") or res.headers.get("X-Correlation-ID")
    # Our fake generator embeds this marker
    assert "ANSWER for: growth strategy" in data["answer"]


def test_e2e_chat_marketing_hr_query(monkeypatch):
    def fake_embed_texts(texts, model=None, api_key=None):
        return [[0.02] * 16 for _ in texts]

    def fake_generate_answer(query, snippets, model: str, api_key: str):
        if not snippets:
            return f"NO_CONTEXT for: {query}\n\nCitations: []"
        src = f"{snippets[0].get('source_path','')}#{snippets[0].get('section_path','')}".rstrip('#')
        return f"ANSWER for: {query}. See [1].\n\nCitations: [1] {src}"

    def fake_chroma_query(collection, q_emb, top_k=5, role=None):
        return []

    monkeypatch.setattr("app.main.embed_texts", fake_embed_texts)
    monkeypatch.setattr("app.main.chroma_query", fake_chroma_query)
    monkeypatch.setattr("app.main.generate_answer", fake_generate_answer)
    # Avoid real chromadb client
    monkeypatch.setattr("app.main.get_client", lambda *a, **k: object())
    monkeypatch.setattr("app.main.get_or_create_collection", lambda _client, name="kb_main": {})

    payload = {
        "message": "employee salary details",
        "top_k": 3,
        "rerank": False,
    }
    res = client.post("/chat", headers=_basic_auth("Bruce", "securepass"), json=payload)
    assert res.status_code == 200, res.text
    data = res.json()
    # Expect marketing user to NOT see HR-only sources in citations
    cits = data.get("citations", [])
    assert isinstance(cits, list)
    assert all("/hr/" not in (c.get("source_path") or "") for c in cits)
    # But answer can still come from general docs
    assert "ANSWER for: employee salary details" in data["answer"]


def test_admin_status_and_forbidden(monkeypatch):
    # Avoid real chroma and OpenAI calls in CI
    monkeypatch.setattr("app.main.get_client", lambda *a, **k: object())
    monkeypatch.setattr("app.main.get_or_create_collection", lambda _client, name="kb_main": type("C", (), {"count": lambda self: 0})())

    def fake_get_model_metadata(api_key: str, model_id: str):
        return (200, {"id": model_id})

    monkeypatch.setattr("app.main.get_model_metadata", fake_get_model_metadata)

    res = client.get("/admin/status", headers=_basic_auth("Clark", "chief"))
    assert res.status_code == 200, res.text
    data = res.json()
    for k in ["sparse_count", "dense_count", "persist_dir", "openai", "correlation_id"]:
        assert k in data

    # Non-c_level cannot reindex
    res2 = client.post("/admin/reindex", headers=_basic_auth("Bruce", "securepass"))
    assert res2.status_code == 403

