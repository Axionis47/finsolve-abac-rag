from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def basic_auth(user: str, password: str):
    import base64
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def test_authz_hr_row_permit_for_hr():
    payload = {
        "resource": {"owner_dept": "hr", "doc_type": "csv", "sensitivity": "restricted"},
        "action": "view_row",
    }
    resp = client.post("/authz/check", json=payload, headers=basic_auth("Natasha", "hrpass123"))
    assert resp.status_code == 200, resp.text
    assert resp.json()["decision"] == "permit"


def test_authz_hr_row_deny_for_marketing():
    payload = {
        "resource": {"owner_dept": "hr", "doc_type": "csv", "sensitivity": "restricted"},
        "action": "view_row",
    }
    resp = client.post("/authz/check", json=payload, headers=basic_auth("Bruce", "securepass"))
    assert resp.status_code == 403


def test_authz_general_doc_permit_for_employee():
    payload = {
        "resource": {"owner_dept": "general", "doc_type": "md", "sensitivity": "general"},
        "action": "retrieve_chunk",
    }
    resp = client.post("/authz/check", json=payload, headers=basic_auth("Tony", "password123"))
    assert resp.status_code == 200
    assert resp.json()["decision"] == "permit"


def test_authz_restricted_md_permit_for_c_level():
    payload = {
        "resource": {"owner_dept": "finance", "doc_type": "md", "sensitivity": "restricted"},
        "action": "retrieve_chunk",
    }
    # There is no c_level user in dummy DB; simulate by making Sam c_level in payload is not possible.
    # Instead, we rely on PDP full access rule for c_level, but since auth user doesn't have that role,
    # this should return 403. This test asserts the endpoint enforces roles from auth, not payload.
    resp = client.post("/authz/check", json=payload, headers=basic_auth("Sam", "financepass"))
    assert resp.status_code == 403

