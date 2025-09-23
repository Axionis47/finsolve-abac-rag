from pathlib import Path
from app.ingest.runner import build_index


def test_build_index_from_temp_dir(tmp_path):
    # Create dir structure
    base = tmp_path / "resources" / "data"
    (base / "general").mkdir(parents=True)
    (base / "finance").mkdir(parents=True)

    # General doc
    (base / "general" / "gen.md").write_text("""
# General Title
This talks about company growth and values.
""".strip(), encoding="utf-8")

    # Finance doc
    (base / "finance" / "fin.md").write_text("""
# Finance Plan
The budget for next quarter includes allocations.
""".strip(), encoding="utf-8")

    index = build_index(base_dir=str(base))
    assert isinstance(index, list) and len(index) >= 2

    # Allowed roles: general visible to all
    gen = [it for it in index if it["owner_dept"] == "general"]
    assert gen and any("marketing" in it["allowed_roles"] for it in gen)

    # Finance internal md visible to finance + c_level
    fin = [it for it in index if it["owner_dept"] == "finance"]
    assert fin and all(set(it["allowed_roles"]) == {"finance", "c_level"} for it in fin)

