from app.ingest.discovery import discover_resources
from app.ingest.allow_roles import compute_allowed_roles


def test_discovery_finds_general_and_departmental_files():
    items = discover_resources()
    assert isinstance(items, list)
    # Should discover at least one general markdown doc
    general = [i for i in items if i["owner_dept"] == "general" and i["doc_type"] == "md"]
    assert len(general) >= 1
    # Should discover at least one departmental markdown doc
    dept = [i for i in items if i["owner_dept"] in {"engineering", "finance", "marketing", "hr"} and i["doc_type"] in {"md", "csv"}]
    assert len(dept) >= 1


def test_allowed_roles_computation_basic_cases():
    # General docs visible to all
    md = {"owner_dept": "general", "doc_type": "md", "sensitivity": "general"}
    assert set(compute_allowed_roles(md)) >= {"employee", "engineering", "finance", "marketing", "hr", "c_level"}

    # Departmental md internal
    md2 = {"owner_dept": "finance", "doc_type": "md", "sensitivity": "internal"}
    assert set(compute_allowed_roles(md2)) == {"finance", "c_level"}

    # HR rows
    md3 = {"owner_dept": "hr", "doc_type": "csv", "sensitivity": "restricted"}
    assert set(compute_allowed_roles(md3)) == {"hr", "c_level"}

