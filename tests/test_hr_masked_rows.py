from fastapi.testclient import TestClient
import base64
import os

from app.main import app

client = TestClient(app)


def basic_auth(user: str, password: str):
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def test_masked_rows_denied_when_disabled(monkeypatch):
    import app.main as main_mod
    # Ensure flag disabled
    main_mod.SETTINGS["HR_MASKED_ROWS_MODE"] = "disabled"

    r = client.get("/hr/rows_masked", headers=basic_auth("Bruce", "securepass"))
    assert r.status_code == 403
    assert "X-Correlation-ID" in r.headers


def test_masked_rows_permitted_when_enabled(monkeypatch):
    import app.main as main_mod
    # Enable flag
    main_mod.SETTINGS["HR_MASKED_ROWS_MODE"] = "enabled"

    r = client.get("/hr/rows_masked", headers=basic_auth("Bruce", "securepass"))
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["count"] >= 1
    assert "rows" in payload and isinstance(payload["rows"], list)
    assert "X-Correlation-ID" in r.headers

    # HR should also be permitted regardless of flag
    r2 = client.get("/hr/rows_masked", headers=basic_auth("Natasha", "hrpass123"))
    assert r2.status_code == 200, r2.text

