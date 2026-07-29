"""Turn one markdown file into retrievable chunks.

First step of the ingest pipeline: read a corpus document, follow its heading
tree (H1 to H4), and emit one `Chunk` per section that has text of its own.
Each chunk carries a breadcrumb ("Doc Title > H2 > H3") in front of its body so
the near-identical quarterly reports stay tellable apart, and a `chunk_id` built
only from the document, the heading path and the part number. Ids are stable on
purpose: gold labels in the eval set must survive re-chunking. `allowed_roles`
is left empty; the pipeline fills it by asking the policy engine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from finrag.models import Chunk

# A crude token estimate is enough for sizing: about four characters per token.
CHARS_PER_TOKEN = 4
# Sections are packed up to the target; the hard max leaves a little headroom.
# A paragraph past the hard max cannot be packed down, so the chunker raises
# rather than emit it. The real corpus never gets close: its largest is 400.
TARGET_MAX_TOKENS = 400
HARD_MAX_TOKENS = 480

HEADING_RE = re.compile(r"^(#{1,4})\s+(.*\S)\s*$")
PARAGRAPH_BREAK_RE = re.compile(r"\n\s*\n")
QUARTER_IN_TEXT_RE = re.compile(r"\bQ([1-4])\b")
# Filenames are lowercase and use underscores: marketing_report_q2_2024.md and
# the odd one out, market_report_q4_2024.md, both match this.
QUARTER_IN_FILENAME_RE = re.compile(r"(?:^|[^a-z0-9])q([1-4])(?![0-9])")
# Not \b...\b: in a filename the year sits next to an underscore, which counts
# as a word character, so a word boundary would never fire there.
YEAR_RE = re.compile(r"(?<![0-9])(20\d{2})(?![0-9])")
SLUG_KEEP = "abcdefghijklmnopqrstuvwxyz0123456789"

BREADCRUMB_SEPARATOR = " > "
SLUG_SEPARATOR = "--"
# Text that sits before the first heading has no heading path of its own.
PREAMBLE_SLUG = "preamble"


@dataclass(frozen=True)
class _Section:
    """One run of body text plus the heading titles above it. `heading_path`
    is indexed by heading level: entry 0 is the H1, entry 1 the H2, and so on.
    A level the document skipped is an empty string.
    """

    heading_path: tuple[str, ...]
    body: str


def estimate_tokens(text: str) -> int:
    """Rough token count used for every size decision in this file."""
    return len(text) // CHARS_PER_TOKEN


def slugify(text: str) -> str:
    """Lowercase the text and keep only letters, digits, and hyphens."""
    characters = []
    for character in text.lower():
        # Anything outside the keep set becomes a hyphen, collapsed just below.
        characters.append(character if character in SLUG_KEEP else "-")
    slug = "".join(characters)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")


def _humanize_stem(stem: str) -> str:
    """Fallback document title for a file with no H1: 'q2_report' -> 'Q2 Report'."""
    return stem.replace("_", " ").replace("-", " ").strip().title()


def _find_year(text: str) -> str | None:
    match = YEAR_RE.search(text)
    return match.group(1) if match else None


def _format_quarter(digit: str, year: str | None) -> str:
    """"3", "2024" -> "Q3-2024". Without a year we keep the bare quarter."""
    return f"Q{digit}" if year is None else f"Q{digit}-{year}"


def extract_quarter(section_path: str, filename: str) -> str | None:
    """Work out which quarter a section is about, or None if it spans the year.

    The heading path wins over the filename, and the deepest heading wins inside
    it, so "Recommendations for Q3 2024" in a Q2 report is labelled Q3. The year
    comes from the matching segment, then the rest of the path, then the filename.
    """
    segments = [segment.strip() for segment in section_path.split(">")]
    for segment in reversed(segments):
        match = QUARTER_IN_TEXT_RE.search(segment)
        if match:
            year = _find_year(segment) or _find_year(section_path) or _find_year(filename)
            return _format_quarter(match.group(1), year)

    match = QUARTER_IN_FILENAME_RE.search(filename.lower())
    if match:
        year = _find_year(filename) or _find_year(section_path)
        return _format_quarter(match.group(1), year)
    return None


def _is_horizontal_rule(paragraph: str) -> bool:
    """True for markdown dividers such as '---', which carry no meaning."""
    stripped = paragraph.strip()
    return len(stripped) >= 3 and set(stripped) <= {"-", "*", "_"}


def _split_sections(text: str) -> list[_Section]:
    """Walk the file once and cut it at every H1 to H4 heading."""
    sections: list[_Section] = []
    heading_stack: list[str] = []
    body_lines: list[str] = []
    in_code_fence = False

    for raw_line in text.splitlines():
        line = raw_line.rstrip()

        if line.lstrip().startswith("```"):
            in_code_fence = not in_code_fence
            body_lines.append(line)
            continue

        match = None if in_code_fence else HEADING_RE.match(line)
        if match is None:
            body_lines.append(line)
            continue

        # The heading closes the section above it.
        sections.append(_Section(tuple(heading_stack), "\n".join(body_lines)))
        body_lines = []

        level = len(match.group(1))
        del heading_stack[level - 1 :]
        while len(heading_stack) < level - 1:
            heading_stack.append("")  # the document skipped a level
        heading_stack.append(match.group(2).strip())

    sections.append(_Section(tuple(heading_stack), "\n".join(body_lines)))
    return [section for section in sections if section.body.strip()]


def _document_title(sections: list[_Section], stem: str) -> str:
    """The first H1 in the file, or a readable version of the filename."""
    for section in sections:
        if section.heading_path and section.heading_path[0]:
            return section.heading_path[0]
    return _humanize_stem(stem)


def _breadcrumb_and_slug(section: _Section, doc_title: str) -> tuple[str, str]:
    """Build the breadcrumb shown in the text and the slug used in the id."""
    if not section.heading_path:
        return doc_title, PREAMBLE_SLUG

    own_title = section.heading_path[0]
    deeper = [title for title in section.heading_path[1:] if title]
    breadcrumb_parts = [own_title or doc_title] + deeper
    slug_parts = list(deeper)
    # A second H1 would otherwise share a slug with the first one.
    if own_title and own_title != doc_title:
        slug_parts.insert(0, own_title)
    if not slug_parts:
        slug_parts = [own_title or doc_title]

    breadcrumb = BREADCRUMB_SEPARATOR.join(breadcrumb_parts)
    slug = SLUG_SEPARATOR.join(slugify(part) for part in slug_parts)
    return breadcrumb, slug


def _split_body_into_parts(body: str, used_tokens: int, doc_id: str, section: str) -> list[str]:
    """Pack paragraphs into parts that fit the target size.

    Blank lines are the only place we cut, so a paragraph past the hard maximum
    cannot be shrunk at all. We raise rather than write an oversize chunk that
    surfaces later as a truncated prompt. `doc_id` and `section` name the break.
    """
    paragraphs = []
    for block in PARAGRAPH_BREAK_RE.split(body):
        block = block.strip()
        if block and not _is_horizontal_rule(block):
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

    sections = _split_sections(text)
    doc_title = _document_title(sections, path.stem)

    chunks: list[Chunk] = []
    for section in sections:
        breadcrumb, slug = _breadcrumb_and_slug(section, doc_title)
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
