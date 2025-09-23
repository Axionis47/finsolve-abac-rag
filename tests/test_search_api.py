from fastapi.testclient import TestClient
from pathlib import Path

from app.main import app
from app.ingest.runner import build_index

client = TestClient(app)


def basic_auth(user: str, password: str):
    import base64
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def setup_temp_index(tmp_path):
    base = tmp_path / "resources" / "data"
    (base / "general").mkdir(parents=True)
    (base / "finance").mkdir(parents=True)

    (base / "general" / "gen.md").write_text("""
# General
We focus on growth and culture.
""".strip(), encoding="utf-8")

    (base / "finance" / "fin.md").write_text("""
# Finance
Budget planning and allocations are key.
""".strip(), encoding="utf-8")

    app.state.index = build_index(base_dir=str(base))


def test_search_endpoint_respects_permissions(tmp_path):
    setup_temp_index(tmp_path)

    # Marketing user can search and see general docs but not finance
    resp = client.post("/search", json={"query": "growth", "top_k": 5}, headers=basic_auth("Bruce", "securepass"))
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] >= 1
    assert all(r["owner_dept"] == "general" for r in data["results"])  # only general in results

    # Marketing searching for finance content should return 0
    resp2 = client.post("/search", json={"query": "budget", "top_k": 5}, headers=basic_auth("Bruce", "securepass"))
    assert resp2.status_code == 200
    assert resp2.json()["count"] == 0

    # Finance user can see finance content
    resp3 = client.post("/search", json={"query": "budget", "top_k": 5}, headers=basic_auth("Sam", "financepass"))
    assert resp3.status_code == 200
    data3 = resp3.json()
    assert data3["count"] >= 1
    assert any(r["owner_dept"] == "finance" for r in data3["results"])
