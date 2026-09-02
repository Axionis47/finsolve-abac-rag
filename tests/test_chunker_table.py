"""Tests for the table-aware chunker: table-atomic packing, row-group
splitting, and pseudo-heading promotion, against both synthetic files and the
two real corpus documents that motivated it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from finrag.ingest.chunker_table import chunk_markdown_table_aware
from finrag.ingest.md_common import HARD_MAX_TOKENS, estimate_tokens

DOC_ID = "test/doc"


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_small_table_stays_in_one_chunk_with_header(tmp_path: Path) -> None:
    """A small table, abutted by prose with no blank line, is never split."""
    text = (
        "# Doc Title\n\n"
        "## Section\n\n"
        "Some prose right before the table.\n"
        "| A | B |\n"
        "|---|---|\n"
        "| 1 | 2 |\n"
        "More prose right after, no blank line.\n"
    )
    path = _write(tmp_path, "small_table.md", text)
    chunks = chunk_markdown_table_aware(path, DOC_ID, "engineering", "internal")

    section_chunks = [c for c in chunks if c.section_slug == "section"]
    assert len(section_chunks) == 1
    chunk = section_chunks[0]
    assert "| A | B |" in chunk.text
    assert "|---|---|" in chunk.text
    assert "| 1 | 2 |" in chunk.text
    assert "Some prose right before the table." in chunk.text
    assert "More prose right after, no blank line." in chunk.text


def _big_table_lines(n_rows: int) -> list[str]:
    header = "| Name | Description |"
    separator = "|------|-------------|"
    rows = [
        f"| Item{i:03d} | A moderately long description of item number {i:03d} "
        "used to pad out the row so the table grows past the hard token limit |"
        for i in range(n_rows)
    ]
    return [header, separator, *rows]


def test_oversized_table_splits_into_row_groups_with_repeated_header(
    tmp_path: Path,
) -> None:
    table_lines = _big_table_lines(60)
    text = "# Doc Title\n\n## Big Table\n\n" + "\n".join(table_lines) + "\n"
    path = _write(tmp_path, "big_table.md", text)

    full_table_text = "\n".join(table_lines)
    assert estimate_tokens(full_table_text) > HARD_MAX_TOKENS  # sanity: test data is big enough

    chunks = chunk_markdown_table_aware(path, DOC_ID, "engineering", "internal")
    table_chunks = [c for c in chunks if c.section_slug == "big-table"]

    assert len(table_chunks) > 1
    ids = [c.chunk_id for c in table_chunks]
    assert len(set(ids)) == len(ids)
    for index, chunk in enumerate(table_chunks, start=1):
        assert chunk.chunk_id == f"{DOC_ID}#big-table/t{index}"
        assert "| Name | Description |" in chunk.text
        assert "|------|-------------|" in chunk.text
        assert estimate_tokens(chunk.text) <= HARD_MAX_TOKENS


def test_table_row_too_wide_to_fit_alone_raises(tmp_path: Path) -> None:
    """A single monster row can't be packed even alone in a fresh group.

    The table as a whole must still exceed HARD_MAX_TOKENS (that is what
    routes it to row-group splitting at all); a handful of ordinary rows plus
    one absurdly wide one satisfies that while isolating the failure to the
    per-row check.
    """
    header = "| Name | Description |"
    separator = "|------|-------------|"
    ordinary_rows = [f"| Item{i:03d} | A short description {i:03d} |" for i in range(5)]
    monster_row = "| Monster | " + ("padding " * 400) + "|"
    table_lines = [header, separator, *ordinary_rows, monster_row]
    text = "# Doc Title\n\n## Big Table\n\n" + "\n".join(table_lines) + "\n"
    path = _write(tmp_path, "monster_row.md", text)

    with pytest.raises(ValueError) as excinfo:
        chunk_markdown_table_aware(path, DOC_ID, "engineering", "internal")
    message = str(excinfo.value)
    assert "Big Table" in message
    assert str(HARD_MAX_TOKENS) in message


def test_table_header_alone_too_wide_raises(tmp_path: Path) -> None:
    """Header + separator alone over the hard maximum, with no data rows,
    must raise rather than silently produce zero chunks."""
    huge_cell = "padding " * 400
    header = f"| Name | {huge_cell}|"
    separator = "|------|-------------|"
    text = "# Doc Title\n\n## Big Table\n\n" + "\n".join([header, separator]) + "\n"
    path = _write(tmp_path, "huge_header.md", text)

    with pytest.raises(ValueError) as excinfo:
        chunk_markdown_table_aware(path, DOC_ID, "engineering", "internal")
    message = str(excinfo.value)
    assert "Big Table" in message
    assert str(HARD_MAX_TOKENS) in message


def test_oversized_prose_paragraph_still_raises(tmp_path: Path) -> None:
    """The ValueError contract for an unsplittable paragraph is unchanged."""
    huge_paragraph = "word " * 2000  # no blank line anywhere inside it
    text = f"# Doc Title\n\n## Section\n\n{huge_paragraph}\n"
    path = _write(tmp_path, "huge_paragraph.md", text)

    with pytest.raises(ValueError):
        chunk_markdown_table_aware(path, DOC_ID, "engineering", "internal")


def test_plain_horizontal_rule_creates_no_phantom_section(tmp_path: Path) -> None:
    """A dash rule preceded by a blank line is a divider, not a title."""
    text = (
        "# Doc Title\n\n"
        "## Section One\n\n"
        "First paragraph.\n\n"
        "---\n\n"
        "Second paragraph, still in Section One.\n"
    )
    path = _write(tmp_path, "plain_rule.md", text)
    chunks = chunk_markdown_table_aware(path, DOC_ID, "engineering", "internal")

    slugs = {c.section_slug for c in chunks}
    assert slugs == {"section-one"}


def test_financial_summary_gets_named_sections(corpus_dir: Path) -> None:
    """Pseudo-heading promotion turns the setext titles into real sections."""
    path = corpus_dir / "finance" / "financial_summary.md"
    chunks = chunk_markdown_table_aware(path, "finance/financial_summary", "finance", "internal")

    slugs = {c.section_slug for c in chunks}
    assert len(slugs) > 1
    assert "executive-summary" in slugs
    assert "appendices" in slugs


def test_marketing_report_gets_named_sections(corpus_dir: Path) -> None:
    path = corpus_dir / "marketing" / "marketing_report_2024.md"
    chunks = chunk_markdown_table_aware(
        path, "marketing/marketing_report_2024", "marketing", "internal"
    )

    slugs = {c.section_slug for c in chunks}
    assert len(slugs) > 1
    assert "executive-summary" in slugs
    assert "vendor-performance" in slugs


def test_engineering_doc_is_deterministic(corpus_dir: Path) -> None:
    path = corpus_dir / "engineering" / "engineering_master_doc.md"
    first = chunk_markdown_table_aware(path, "engineering/engineering_master_doc", "engineering", "internal")
    second = chunk_markdown_table_aware(path, "engineering/engineering_master_doc", "engineering", "internal")
    assert first == second


def test_real_corpus_chunks_carry_doc_type_and_caller_metadata(corpus_dir: Path) -> None:
    files = [
        (corpus_dir / "engineering" / "engineering_master_doc.md", "engineering", "internal"),
        (corpus_dir / "general" / "employee_handbook.md", "general", "general"),
        (corpus_dir / "finance" / "financial_summary.md", "finance", "internal"),
        (corpus_dir / "marketing" / "marketing_report_2024.md", "marketing", "internal"),
    ]
    for path, owner_dept, sensitivity in files:
        chunks = chunk_markdown_table_aware(path, f"{owner_dept}/{path.stem}", owner_dept, sensitivity)
        assert chunks, f"{path} produced no chunks"
        for chunk in chunks:
            assert chunk.doc_type == "md"
            assert chunk.owner_dept == owner_dept
            assert chunk.sensitivity == sensitivity


def test_employee_handbook_tables_stay_atomic(corpus_dir: Path) -> None:
    """The handbook's three pipe tables (benefits, leave, salary) each land in
    exactly one chunk together with their header row, never split apart."""
    path = corpus_dir / "general" / "employee_handbook.md"
    chunks = chunk_markdown_table_aware(path, "general/employee_handbook", "general", "general")

    tables = {
        "| Benefit | Details |": "| **Bonus** | As per Payment of Bonus Act; minimum 8.33% of basic salary |",
        "| Leave Type | Entitlement/Details |": "| **Leave Without Pay** | Allowed with approval after exhausting paid leave |",
        "| Component | Details |": "| **ESI** | For eligible employees |",
    }
    for header, last_row in tables.items():
        owners = [c for c in chunks if header in c.text and last_row in c.text]
        assert len(owners) == 1, f"table under {header!r} did not land whole in one chunk"
