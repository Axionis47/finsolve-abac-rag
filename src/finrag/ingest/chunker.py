"""Turn one markdown file into retrievable chunks.

First step of the ingest pipeline: read a corpus document, follow its heading
tree (H1 to H4), and emit one `Chunk` per section that has text of its own.
Each chunk carries a breadcrumb ("Doc Title > H2 > H3") in front of its body so
the near-identical quarterly reports stay tellable apart, and a `chunk_id` built
only from the document, the heading path and the part number. Ids are stable on
purpose: gold labels in the eval set must survive re-chunking. `allowed_roles`
is left empty; the pipeline fills it by asking the policy engine.

The heading-splitting, breadcrumb, slug and sizing machinery this module needs
lives in `md_common.py`, shared with `chunker_table.py`. The names are
re-exported here (as the underscore-prefixed originals, plus the public ones)
so existing `from finrag.ingest.chunker import ...` call sites keep working
unchanged.
"""

from __future__ import annotations

from pathlib import Path

from finrag.ingest.md_common import (
    CHARS_PER_TOKEN,
    HARD_MAX_TOKENS,
    HEADING_RE,
    PARAGRAPH_BREAK_RE,
    PREAMBLE_SLUG,
    QUARTER_IN_FILENAME_RE,
    QUARTER_IN_TEXT_RE,
    TARGET_MAX_TOKENS,
    YEAR_RE,
    Section,
    breadcrumb_and_slug,
    document_title,
    estimate_tokens,
    extract_quarter,
    is_horizontal_rule,
    slugify,
    split_sections,
)
from finrag.models import Chunk

# Re-exports under the module's original private names, kept for any caller
# that imported them before the move to md_common.py.
_Section = Section
_split_sections = split_sections
_document_title = document_title
_breadcrumb_and_slug = breadcrumb_and_slug
_is_horizontal_rule = is_horizontal_rule

__all__ = [
    "CHARS_PER_TOKEN",
    "HARD_MAX_TOKENS",
    "HEADING_RE",
    "PARAGRAPH_BREAK_RE",
    "PREAMBLE_SLUG",
    "QUARTER_IN_FILENAME_RE",
    "QUARTER_IN_TEXT_RE",
    "TARGET_MAX_TOKENS",
    "YEAR_RE",
    "chunk_markdown",
    "estimate_tokens",
    "extract_quarter",
    "slugify",
]


def _split_body_into_parts(body: str, used_tokens: int, doc_id: str, section: str) -> list[str]:
    """Pack paragraphs into parts that fit the target size.

    Blank lines are the only place we cut, so a paragraph past the hard maximum
    cannot be shrunk at all. We raise rather than write an oversize chunk that
    surfaces later as a truncated prompt. `doc_id` and `section` name the break.
    """
    paragraphs = []
    for block in PARAGRAPH_BREAK_RE.split(body):
        block = block.strip()
        if block and not is_horizontal_rule(block):
            paragraphs.append(block)

    parts: list[str] = []
    current: list[str] = []
    current_tokens = used_tokens

    for paragraph in paragraphs:
        paragraph_tokens = estimate_tokens(paragraph)
        if paragraph_tokens > HARD_MAX_TOKENS:
            raise ValueError(
                f"paragraph of about {paragraph_tokens} estimated tokens is over the hard "
                f"maximum of {HARD_MAX_TOKENS}, with no blank line to split on, in "
                f"doc {doc_id}, section '{section}'"
            )
        if current and current_tokens + paragraph_tokens > TARGET_MAX_TOKENS:
            parts.append("\n\n".join(current))
            current = []
            current_tokens = used_tokens
        current.append(paragraph)
        current_tokens += paragraph_tokens

    if current:
        parts.append("\n\n".join(current))
    return parts


def chunk_markdown(
    path: str | Path, doc_id: str, owner_dept: str, sensitivity: str
) -> list[Chunk]:
    """Read one markdown file and return its chunks in document order."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    sections = split_sections(text)
    doc_title = document_title(sections, path.stem)

    chunks: list[Chunk] = []
    for section in sections:
        breadcrumb, slug = breadcrumb_and_slug(section, doc_title)
        quarter = extract_quarter(breadcrumb, path.name)
        # The breadcrumb rides on every part, so it counts against the budget.
        used = estimate_tokens(breadcrumb)
        parts = _split_body_into_parts(section.body, used, doc_id, breadcrumb)

        for index, part in enumerate(parts, start=1):
            chunk_id = f"{doc_id}#{slug}"
            if len(parts) > 1:
                chunk_id = f"{chunk_id}/p{index}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    section_path=breadcrumb,
                    section_slug=slug,
                    text=f"{breadcrumb}\n\n{part}",
                    owner_dept=owner_dept,
                    doc_type="md",
                    sensitivity=sensitivity,
                    allowed_roles=(),
                    quarter=quarter,
                )
            )
    return chunks
