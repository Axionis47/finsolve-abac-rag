from app.ingest.discovery import discover_resources, infer_doc_type, default_sensitivity
from app.ingest.allow_roles import compute_allowed_roles
from pathlib import Path


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


# ============================================================================
# Additional tests for complete coverage
# ============================================================================


def test_infer_doc_type_md():
    """Test markdown file detection."""
    assert infer_doc_type(Path("/some/path/doc.md")) == "md"
    assert infer_doc_type(Path("/some/path/doc.MD")) == "md"  # case insensitive


def test_infer_doc_type_csv():
    """Test CSV file detection."""
    assert infer_doc_type(Path("/some/path/data.csv")) == "csv"
    assert infer_doc_type(Path("/some/path/data.CSV")) == "csv"


def test_infer_doc_type_unknown():
    """Test unknown file types."""
    assert infer_doc_type(Path("/some/path/file.txt")) == "unknown"
    assert infer_doc_type(Path("/some/path/file.pdf")) == "unknown"
    assert infer_doc_type(Path("/some/path/file")) == "unknown"


def test_default_sensitivity_general():
    """General dept docs have 'general' sensitivity."""
    assert default_sensitivity("general", "md") == "general"
    assert default_sensitivity("general", "csv") == "general"


def test_default_sensitivity_hr_csv():
    """HR CSV files have 'restricted' sensitivity."""
    assert default_sensitivity("hr", "csv") == "restricted"


def test_default_sensitivity_dept_internal():
    """Other departmental docs have 'internal' sensitivity."""
    assert default_sensitivity("finance", "md") == "internal"
    assert default_sensitivity("hr", "md") == "internal"
    assert default_sensitivity("marketing", "md") == "internal"


def test_discovery_with_nonexistent_base_dir():
    """Discovery returns empty list for non-existent directory."""
    items = discover_resources(base_dir="/nonexistent/path/12345")
    assert items == []


def test_allowed_roles_hr_aggregate():
    """HR aggregate docs are only accessible to HR and C-Level."""
    md = {"owner_dept": "hr", "doc_type": "aggregate", "sensitivity": "internal"}
    assert set(compute_allowed_roles(md)) == {"hr", "c_level"}


def test_allowed_roles_restricted_md_no_roles():
    """Restricted MD docs (not general/internal) return empty roles by default."""
    # This tests the fallback case - restricted sensitivity with md
    md = {"owner_dept": "finance", "doc_type": "md", "sensitivity": "restricted"}
    # Since sensitivity is restricted (not general/internal), rule doesn't match
    assert compute_allowed_roles(md) == []


def test_allowed_roles_empty_metadata():
    """Empty metadata returns no allowed roles."""
    assert compute_allowed_roles({}) == []


def test_allowed_roles_unknown_doc_type():
    """Unknown doc type returns no allowed roles."""
    md = {"owner_dept": "finance", "doc_type": "unknown", "sensitivity": "internal"}
    assert compute_allowed_roles(md) == []

