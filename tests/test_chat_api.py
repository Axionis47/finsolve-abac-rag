import base64
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

def basic_auth(user: str, password: str):
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def test_chat_endpoint_with_monkeypatched_generator(monkeypatch):
    # Monkeypatch generation to be deterministic and offline
    import app.services.generation as gen

    def fake_generate_answer(query, snippets, model: str, api_key: str):
        if not snippets:
            return "I don't know based on the available context."
        first = snippets[0]
        src = f"{first.get('source_path','')}#{first.get('section_path','')}".strip('#')
        return f"ANSWER for: {query}. See [1].\n\nCitations: [1] {src}"

    monkeypatch.setattr(gen, "generate_answer", fake_generate_answer)

    # Use hybrid retrieval defaults; rely on previously set up index/test data
    r = client.post(
        "/chat",
        json={"message": "growth strategy", "top_k": 3, "persist_dir": ".chroma"},
        headers=basic_auth("Bruce", "securepass"),
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert "answer" in data and isinstance(data["answer"], str)
    assert "citations" in data and isinstance(data["citations"], list)
    assert "metrics" in data and "llm_ms" in data["metrics"]
    assert "X-Correlation-ID" in r.headers

