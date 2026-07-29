"""Tests for the markdown chunker, run against two real corpus documents.

The chunk id list is pinned as a snapshot on purpose: gold labels in the eval
set point at these ids, so a change here has to be a deliberate one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from finrag.ingest.chunker import (
    HARD_MAX_TOKENS,
    chunk_markdown,
    estimate_tokens,
    extract_quarter,
    slugify,
)
from finrag.models import Chunk

MARKETING_DOC_ID = "marketing/marketing_report_q2_2024"
FINANCE_DOC_ID = "finance/quarterly_financial_report"

EXPECTED_MARKETING_IDS = [
    f"{MARKETING_DOC_ID}#executive-summary",
    f"{MARKETING_DOC_ID}#q2-marketing-overview",
    f"{MARKETING_DOC_ID}#q2-projections-targets",
    f"{MARKETING_DOC_ID}#q2-benchmarks",
    f"{MARKETING_DOC_ID}#q2-strategic-objectives",
    f"{MARKETING_DOC_ID}#campaign-highlights--digital-marketing",
    f"{MARKETING_DOC_ID}#campaign-highlights--influencer-partnerships",
    f"{MARKETING_DOC_ID}#campaign-highlights--email-retargeting",
    f"{MARKETING_DOC_ID}#performance-analysis",
    f"{MARKETING_DOC_ID}#recommendations-for-q3-2024",
    f"{MARKETING_DOC_ID}#conclusion",
]

EXPECTED_FINANCE_IDS = [
    f"{FINANCE_DOC_ID}#executive-summary",
    f"{FINANCE_DOC_ID}#q1-january-to-march-2024--quarterly-financial-overview",
    f"{FINANCE_DOC_ID}#q1-january-to-march-2024--quarterly-expense-breakdown",
    f"{FINANCE_DOC_ID}#q1-january-to-march-2024--cash-flow-analysis",
    f"{FINANCE_DOC_ID}#q1-january-to-march-2024--risks-mitigation",
    f"{FINANCE_DOC_ID}#q2-april-to-june-2024--quarterly-financial-overview",
    f"{FINANCE_DOC_ID}#q2-april-to-june-2024--quarterly-expense-breakdown",
    f"{FINANCE_DOC_ID}#q2-april-to-june-2024--cash-flow-analysis",
    f"{FINANCE_DOC_ID}#q2-april-to-june-2024--risks-mitigation",
    f"{FINANCE_DOC_ID}#q3-july-to-september-2024--quarterly-financial-overview",
    f"{FINANCE_DOC_ID}#q3-july-to-september-2024--quarterly-expense-breakdown",
    f"{FINANCE_DOC_ID}#q3-july-to-september-2024--cash-flow-analysis",
    f"{FINANCE_DOC_ID}#q3-july-to-september-2024--risks-mitigation",
    f"{FINANCE_DOC_ID}#q4-october-to-december-2024--quarterly-financial-overview",
    f"{FINANCE_DOC_ID}#q4-october-to-december-2024--quarterly-expense-breakdown",
    f"{FINANCE_DOC_ID}#q4-october-to-december-2024--cash-flow-analysis",
    f"{FINANCE_DOC_ID}#q4-october-to-december-2024--risks-mitigation",
    f"{FINANCE_DOC_ID}#2024-annual-summary",
    f"{FINANCE_DOC_ID}#recommendations-for-2025",
    f"{FINANCE_DOC_ID}#conclusion",
]


@pytest.fixture(scope="module")
def marketing_path(corpus_dir: Path) -> Path:
    return corpus_dir / "marketing" / "marketing_report_q2_2024.md"


@pytest.fixture(scope="module")
def finance_path(corpus_dir: Path) -> Path:
    return corpus_dir / "finance" / "quarterly_financial_report.md"


@pytest.fixture(scope="module")
def marketing_chunks(marketing_path: Path) -> list[Chunk]:
    return chunk_markdown(marketing_path, MARKETING_DOC_ID, "marketing", "internal")


@pytest.fixture(scope="module")
def finance_chunks(finance_path: Path) -> list[Chunk]:
    return chunk_markdown(finance_path, FINANCE_DOC_ID, "finance", "confidential")


def test_marketing_chunk_ids_match_snapshot(marketing_chunks: list[Chunk]) -> None:
    assert [chunk.chunk_id for chunk in marketing_chunks] == EXPECTED_MARKETING_IDS


def test_finance_chunk_ids_match_snapshot(finance_chunks: list[Chunk]) -> None:
    assert [chunk.chunk_id for chunk in finance_chunks] == EXPECTED_FINANCE_IDS


def test_chunk_ids_are_unique(
    marketing_chunks: list[Chunk], finance_chunks: list[Chunk]
) -> None:
    ids = [chunk.chunk_id for chunk in marketing_chunks + finance_chunks]
    assert len(set(ids)) == len(ids)


def test_carried_metadata_is_what_the_caller_passed(
    marketing_chunks: list[Chunk],
) -> None:
    for chunk in marketing_chunks:
        assert chunk.doc_id == MARKETING_DOC_ID
        assert chunk.owner_dept == "marketing"
        assert chunk.sensitivity == "internal"
        assert chunk.doc_type == "md"
        # The pipeline fills this in later by asking the policy engine.
        assert chunk.allowed_roles == ()
        assert chunk.payload == {}


def test_no_chunk_exceeds_the_hard_maximum(
    marketing_chunks: list[Chunk], finance_chunks: list[Chunk]
) -> None:
    for chunk in marketing_chunks + finance_chunks:
        assert estimate_tokens(chunk.text) <= HARD_MAX_TOKENS


def test_chunks_are_not_empty(
    marketing_chunks: list[Chunk], finance_chunks: list[Chunk]
) -> None:
    for chunk in marketing_chunks + finance_chunks:
        body = chunk.text[len(chunk.section_path) :].strip()
        assert body, f"{chunk.chunk_id} has a breadcrumb but no body"


def test_breadcrumb_is_prepended_to_the_text(finance_chunks: list[Chunk]) -> None:
    first = finance_chunks[1]
    expected_path = (
        "Quarterly Financial Report - FinSolve Technologies Inc. 2024 > "
        "Q1 - January to March 2024 > Quarterly Financial Overview"
    )
    assert first.section_path == expected_path
    assert first.text.startswith(expected_path)
    for chunk in finance_chunks:
        assert chunk.text.startswith(chunk.section_path)
        assert chunk.section_path.startswith(
            "Quarterly Financial Report - FinSolve Technologies Inc. 2024 > "
        )


def test_section_level_quarter_from_finance_headings(
    finance_chunks: list[Chunk],
) -> None:
    """The finance report carries all four quarters as H2 sections."""
    by_id = {chunk.chunk_id: chunk for chunk in finance_chunks}
    cash_flow_sections = {
        "Q1-2024": "q1-january-to-march-2024--cash-flow-analysis",
        "Q2-2024": "q2-april-to-june-2024--cash-flow-analysis",
        "Q3-2024": "q3-july-to-september-2024--cash-flow-analysis",
        "Q4-2024": "q4-october-to-december-2024--cash-flow-analysis",
    }
    for expected_quarter, slug in cash_flow_sections.items():
        assert by_id[f"{FINANCE_DOC_ID}#{slug}"].quarter == expected_quarter
    # Sections that span the whole year get no quarter, and the finance
    # filename has none to fall back on.
    assert by_id[f"{FINANCE_DOC_ID}#executive-summary"].quarter is None
    assert by_id[f"{FINANCE_DOC_ID}#2024-annual-summary"].quarter is None


def test_marketing_quarters_come_from_the_document(
    marketing_chunks: list[Chunk],
) -> None:
    by_id = {chunk.chunk_id: chunk for chunk in marketing_chunks}
    assert by_id[f"{MARKETING_DOC_ID}#executive-summary"].quarter == "Q2-2024"
    assert by_id[f"{MARKETING_DOC_ID}#q2-benchmarks"].quarter == "Q2-2024"
    # The deepest heading wins, so the forward-looking section is Q3.
    assert by_id[f"{MARKETING_DOC_ID}#recommendations-for-q3-2024"].quarter == "Q3-2024"


def test_quarter_falls_back_to_the_filename() -> None:
    """Both the tidy filename and the odd 'market_report' one must work."""
    assert extract_quarter("Some Title > Conclusion", "marketing_report_q2_2024.md") == "Q2-2024"
    assert extract_quarter("Some Title > Conclusion", "market_report_q4_2024.md") == "Q4-2024"
    assert extract_quarter("Some Title > Conclusion", "employee_handbook.md") is None
    # A heading beats the filename.
    assert extract_quarter("Report > Q1 Recap", "market_report_q4_2024.md") == "Q1-2024"
    # No year anywhere leaves a bare quarter rather than an invented year.
    assert extract_quarter("Report > Q3 Recap", "report.md") == "Q3"


def test_slugify_keeps_only_lowercase_alphanumerics_and_hyphens() -> None:
    assert slugify("Q2 - Projections & Targets") == "q2-projections-targets"
    assert slugify("  ### Risks & Mitigation  ") == "risks-mitigation"
    assert slugify("9.1 Short-Term (2025)") == "9-1-short-term-2025"


def test_chunking_is_deterministic(marketing_path: Path, finance_path: Path) -> None:
    for path, doc_id in ((marketing_path, MARKETING_DOC_ID), (finance_path, FINANCE_DOC_ID)):
        first = chunk_markdown(path, doc_id, "marketing", "internal")
        second = chunk_markdown(path, doc_id, "marketing", "internal")
        assert [chunk.to_dict() for chunk in first] == [chunk.to_dict() for chunk in second]


def test_oversized_section_splits_into_numbered_parts(tmp_path: Path) -> None:
    paragraph = "Revenue grew steadily across the region. " * 12  # about 120 tokens
    body = "\n\n".join(paragraph.strip() for _ in range(8))
    doc = tmp_path / "market_report_q4_2024.md"
    doc.write_text(f"# Big Report 2024\n\n## Long Section\n{body}\n", encoding="utf-8")

    chunks = chunk_markdown(doc, "marketing/big", "marketing", "internal")
    ids = [chunk.chunk_id for chunk in chunks]
    assert len(ids) > 1
    assert ids == [f"marketing/big#long-section/p{n}" for n in range(1, len(ids) + 1)]
    for chunk in chunks:
        # The slug stays put; only the id carries the part number.
        assert chunk.section_slug == "long-section"
        assert chunk.section_path == "Big Report 2024 > Long Section"
        assert chunk.quarter == "Q4-2024"
        assert estimate_tokens(chunk.text) <= HARD_MAX_TOKENS


def test_a_paragraph_over_the_hard_maximum_raises(tmp_path: Path) -> None:
    """One unbreakable paragraph past the hard max is an error, not a big chunk.

    There is no blank line inside it to cut at, so the chunker cannot honour the
    limit. It says so and names the document and section, rather than writing an
    oversize chunk that only shows up later as a truncated prompt.
    """
    sentence = "This sentence has no blank line anywhere inside it. "
    # Comfortably past HARD_MAX_TOKENS at roughly four characters per token.
    huge_paragraph = sentence * 200
    assert estimate_tokens(huge_paragraph) > HARD_MAX_TOKENS

    doc = tmp_path / "huge_2024.md"
    doc.write_text(f"# Huge Report 2024\n\n## Wall Of Text\n{huge_paragraph}\n", encoding="utf-8")

    with pytest.raises(ValueError) as caught:
        chunk_markdown(doc, "marketing/huge", "marketing", "internal")

    message = str(caught.value)
    assert "marketing/huge" in message
    assert "Huge Report 2024 > Wall Of Text" in message
    assert str(HARD_MAX_TOKENS) in message


def test_preamble_before_the_first_heading_becomes_its_own_chunk(tmp_path: Path) -> None:
    doc = tmp_path / "notes_2024.md"
    doc.write_text(
        "A note that sits above every heading.\n\n# Notes 2024\n\n## Detail\nBody text.\n",
        encoding="utf-8",
    )
    chunks = chunk_markdown(doc, "general/notes", "general", "internal")
    assert [chunk.chunk_id for chunk in chunks] == [
        "general/notes#preamble",
        "general/notes#detail",
    ]
    # The preamble is titled by the H1 of the document it opens.
    assert chunks[0].section_path == "Notes 2024"
    assert chunks[0].text.startswith("Notes 2024\n\nA note that sits above")
