import base64
from pathlib import Path
from fastapi.testclient import TestClient
import pytest

from app.main import app

client = TestClient(app)


def basic_auth(user: str, password: str):
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def stub_embed_texts(texts, use_cache=True, **kwargs):
    """Deterministic embeddings without network calls."""
    out = []
    for t in texts:
        s = sum(ord(c) for c in t) % 97
        vec = [float((s + i * 3) % 13) for i in range(16)]
        out.append(vec)
    return out


def setup_temp_corpus(tmp_path: Path):
    base = tmp_path / "resources" / "data"
    (base / "general").mkdir(parents=True)
    (base / "general" / "gen.md").write_text(
        "# General\nCompany growth and strategy overview.",
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
def test_chat_endpoint_with_monkeypatched_generator(tmp_path, monkeypatch):
    # Monkeypatch embeddings to avoid network calls
    import app.ingest.dense_indexer as dense_indexer
    import app.routes.chat as chat_mod
    import app.routes.search as search_mod
    import app.services.reranker_openai as reranker_mod
    import app.services.generation as gen

    monkeypatch.setattr(dense_indexer, "embed_texts", stub_embed_texts)
    monkeypatch.setattr(chat_mod, "embed_texts", stub_embed_texts)
    monkeypatch.setattr(search_mod, "embed_texts", stub_embed_texts)
    monkeypatch.setattr(reranker_mod, "embed_texts", stub_embed_texts)

    def fake_generate_answer(query, snippets, model: str, api_key: str):
        if not snippets:
            return "I don't know based on the available context."
        first = snippets[0]
        src = f"{first.get('source_path','')}#{first.get('section_path','')}".strip('#')
        return f"ANSWER for: {query}. See [1].\n\nCitations: [1] {src}"

    monkeypatch.setattr(gen, "generate_answer", fake_generate_answer)

    # Setup temp corpus and index
    base_dir = str(setup_temp_corpus(tmp_path))
    chroma_dir = str(tmp_path / ".chroma")

    # Reindex dense as c_level first
    resp = client.post(
        "/admin/reindex_dense",
        json={"base_dir": base_dir, "persist_dir": chroma_dir},
        headers=basic_auth("Clark", "chief"),
    )
    assert resp.status_code == 200, resp.text

    # Now test chat
    r = client.post(
        "/chat",
        json={"message": "growth strategy", "top_k": 3, "persist_dir": chroma_dir, "base_dir": base_dir},
        headers=basic_auth("Bruce", "securepass"),
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert "answer" in data and isinstance(data["answer"], str)
    assert "citations" in data and isinstance(data["citations"], list)
    assert "metrics" in data and "llm_ms" in data["metrics"]
    assert "X-Correlation-ID" in r.headers

