"""A second chunking strategy: like `chunker.py`, but tables survive intact.

`chunk_markdown` treats a pipe table as ordinary paragraph text, so a table
that straddles the target size gets cut wherever a blank line happens to
fall — which, for a table, is nowhere inside it, so it either survives whole
by luck or (past the hard maximum) blows up the section entirely. And two
corpus files, financial_summary.md and marketing_report_2024.md, have no `#`
headings at all: they mark sections the old typewriter way, an underlined
title, which `_split_sections` does not recognise, so today the whole file
comes back as a single "preamble" chunk.

`chunk_markdown_table_aware` reuses every piece of machinery in md_common.py
(section splitting, breadcrumbs, slugs, token estimates) and differs in
exactly three places:

1. Before splitting, dash-underline titles are promoted to real `## `
   headings (`_promote_pseudo_headings`), so the two setext-style files get
   real sections.
2. When packing a section body, a run of consecutive `|`-prefixed lines is
   treated as one atomic block, never separated from its header row, even
   when prose abuts it with no blank line (`_section_items`).
3. A table block whose estimated size exceeds `HARD_MAX_TOKENS` cannot be
   packed at all, so it is pulled out of the normal pack and split into row
   groups instead, each carrying its own copy of the header and separator
   row (`_split_table_rows`). Those chunks get an id suffix of `/t1`, `/t2`,
   ... — a different letter from prose's `/p1`, `/p2`, ... so the two
   sequences can share a section without colliding. Table numbering runs
   across every oversized table in a section, not per-table, since ids only
   need to be unique within the section.

LENGTH. This file is over the ~250 line guideline: the promotion pass, the
table/text splitter and the row-group splitter are three genuinely separate
concerns, but each is short and none is reusable outside `chunk_markdown_table_aware`,
so a split would trade one readable file for several that only make sense read
together.
"""

from __future__ import annotations

from pathlib import Path

from finrag.ingest.md_common import (
    HARD_MAX_TOKENS,
    PARAGRAPH_BREAK_RE,
    TARGET_MAX_TOKENS,
    breadcrumb_and_slug,
    document_title,
    estimate_tokens,
    extract_quarter,
    is_horizontal_rule,
    split_sections,
)
from finrag.models import Chunk


def _is_dash_only_rule(line: str) -> bool:
    """True for a line that is nothing but three or more dashes.

    Narrower than `is_horizontal_rule`: `***` and `___` dividers exist in
    markdown too, but the corpus's setext-style titles only ever use dashes,
    so only dashes are read as a candidate pseudo-heading underline.
    """
    stripped = line.strip()
    return len(stripped) >= 3 and set(stripped) == {"-"}


def _promote_pseudo_headings(text: str) -> str:
    """Rewrite dash-underline pseudo-headings as real `## ` headings.

    financial_summary.md and marketing_report_2024.md mark every section with
    a short title line followed by a line of dashes ("Executive Summary:" /
    "-------------"), never with `#`. `split_sections` only understands ATX
    headings, so left alone these files come back as one preamble.

    The pattern is promoted conservatively: the dash line must be dash-only,
    and the line directly above it must be short (<=80 chars), non-blank,
    not itself a table row, and not itself a dash-only line (which would mean
    two dividers in a row, not a title). A trailing ':' is stripped from the
    promoted title. This matters because a *plain* horizontal rule — the kind
    employee_handbook.md and quarterly_financial_report.md use between real
    `##` sections — is, in every instance in this corpus, preceded by a blank
    line, not a title; requiring a non-blank line above the dashes is what
    keeps those plain dividers untouched. Runs on the raw text, before section
    splitting, and skips fenced code blocks the same way `split_sections` does.
    """
    lines = text.split("\n")
    in_code_fence = False
    for index in range(1, len(lines)):
        line = lines[index]
        if line.lstrip().startswith("```"):
            in_code_fence = not in_code_fence
            continue
        if in_code_fence or not _is_dash_only_rule(line):
            continue

        title_line = lines[index - 1]
        title = title_line.strip()
        if not title or len(title) > 80 or "|" in title or _is_dash_only_rule(title_line):
            continue

        if title.endswith(":"):
            title = title[:-1].strip()
        lines[index - 1] = f"## {title}"
        lines[index] = ""  # the underline is consumed by the heading it named
    return "\n".join(lines)


def _raw_blocks(body: str) -> list[tuple[str, str]]:
    """Cut a section body into ("text", ...) and ("table", ...) runs, in order.

    A table run is a maximal run of consecutive lines that start with `|` —
    every corpus table sits directly under its heading with no blank line
    before it, and separator rows (`|---|---|`) are lines that start with `|`
    too, so they stay inside the run. Everything else, blank lines included,
    is a text run, re-split into paragraphs by the caller.
    """
    blocks: list[tuple[str, str]] = []
    text_lines: list[str] = []
    table_lines: list[str] = []

    def flush_text() -> None:
        if text_lines:
            blocks.append(("text", "\n".join(text_lines)))
            text_lines.clear()

    def flush_table() -> None:
        if table_lines:
            blocks.append(("table", "\n".join(table_lines)))
            table_lines.clear()

    for line in body.split("\n"):
        if line.strip().startswith("|"):
            flush_text()
            table_lines.append(line)
        else:
            flush_table()
            text_lines.append(line)
    flush_text()
    flush_table()
    return blocks


def _section_items(body: str) -> tuple[list[str], list[str]]:
    """Split a body into packable items, and the tables too big to pack.

    `items` holds prose paragraphs and table blocks that fit under the hard
    maximum, in document order — exactly what the greedy packer in
    `chunker.py` consumed before, just with tables added as a second kind of
    atomic unit alongside paragraphs. A table over the hard maximum cannot be
    an item (packing it whole would always blow the budget), so it is
    returned separately in `big_tables` for `_split_table_rows` to shrink.
    """
    items: list[str] = []
    big_tables: list[str] = []
    for kind, block in _raw_blocks(body):
        if kind == "table":
            if estimate_tokens(block) > HARD_MAX_TOKENS:
                big_tables.append(block)
            else:
                items.append(block)
            continue
        for paragraph in PARAGRAPH_BREAK_RE.split(block):
            paragraph = paragraph.strip()
            if paragraph and not is_horizontal_rule(paragraph):
                items.append(paragraph)
    return items, big_tables


def _pack_items(items: list[str], used_tokens: int, doc_id: str, section: str) -> list[str]:
    """Greedily pack pre-split items (paragraphs, small tables) to the target size.

    Identical in shape to `chunker.py`'s paragraph packer, generalised to any
    atomic item. Since oversized tables are filtered out before this runs, the
    over-hard-maximum case below can only fire for a prose paragraph — the
    contract `chunker.py` documents (raise rather than silently truncate).
    """
    parts: list[str] = []
    current: list[str] = []
    current_tokens = used_tokens

    for item in items:
        item_tokens = estimate_tokens(item)
        if item_tokens > HARD_MAX_TOKENS:
            raise ValueError(
                f"paragraph of about {item_tokens} estimated tokens is over the hard "
                f"maximum of {HARD_MAX_TOKENS}, with no blank line to split on, in "
                f"doc {doc_id}, section '{section}'"
            )
        if current and current_tokens + item_tokens > TARGET_MAX_TOKENS:
            parts.append("\n\n".join(current))
            current = []
            current_tokens = used_tokens
        current.append(item)
        current_tokens += item_tokens

    if current:
        parts.append("\n\n".join(current))
    return parts


def _split_table_rows(
    table_text: str, used_tokens: int, doc_id: str, section: str
) -> list[str]:
    """Split one oversized table into row groups, each re-headed.

    The first line is the header row and the second the separator row (the
    shape every corpus table uses); both are repeated on every group so a
    group is a valid, readable table on its own. Data rows are then packed
    greedily to the target size, `used_tokens` (the breadcrumb's weight)
    counted in from the start of each group, same as prose packing does.

    Mirrors `_pack_items`' refusal: a table only gets here because it is over
    the hard maximum as a whole, but the header, separator and breadcrumb
    alone (no data rows at all) or a single data row next to them can still
    be too wide to fit in any group, prose-paragraph style. Raise rather than
    emit an oversize chunk or, worse, silently drop the table.
    """
    lines = table_text.split("\n")
    prefix_lines = lines[:2]
    data_rows = lines[2:]
    prefix = "\n".join(prefix_lines)
    prefix_tokens = estimate_tokens(prefix)

    if used_tokens + prefix_tokens > HARD_MAX_TOKENS:
        raise ValueError(
            f"table header and separator row, plus the breadcrumb, are about "
            f"{used_tokens + prefix_tokens} estimated tokens, over the hard maximum of "
            f"{HARD_MAX_TOKENS}, in doc {doc_id}, section '{section}' — no row group can "
            f"ever fit, even with zero data rows"
        )

    groups: list[str] = []
    current_rows: list[str] = []
    current_tokens = used_tokens + prefix_tokens

    for row in data_rows:
        row_tokens = estimate_tokens(row)
        if used_tokens + prefix_tokens + row_tokens > HARD_MAX_TOKENS:
            raise ValueError(
                f"table row of about {row_tokens} estimated tokens, plus the header, "
                f"separator row and breadcrumb, is over the hard maximum of "
                f"{HARD_MAX_TOKENS} with no way to shrink further, in doc {doc_id}, "
                f"section '{section}'"
            )
        if current_rows and current_tokens + row_tokens > TARGET_MAX_TOKENS:
            groups.append("\n".join([prefix, *current_rows]))
            current_rows = []
            current_tokens = used_tokens + prefix_tokens
        current_rows.append(row)
        current_tokens += row_tokens

    if current_rows:
        groups.append("\n".join([prefix, *current_rows]))
    return groups


def chunk_markdown_table_aware(
    path: str | Path, doc_id: str, owner_dept: str, sensitivity: str
) -> list[Chunk]:
    """Read one markdown file and return its chunks, tables kept intact.

    Same signature and overall flow as `chunk_markdown`: sections, then a
    breadcrumb and slug per section, then a packed body, then one `Chunk` per
    part. Prose parts keep the `/pN` suffix rule unchanged (bare id for a
    single part, `/p1`, `/p2`, ... past that). Table row groups always carry
    a `/tN` suffix — they only exist because a table had to be split, so
    there is never a bare-id table chunk to collide with a bare-id prose one.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    text = _promote_pseudo_headings(text)

    sections = split_sections(text)
    doc_title = document_title(sections, path.stem)

    chunks: list[Chunk] = []
    for section in sections:
        breadcrumb, slug = breadcrumb_and_slug(section, doc_title)
        quarter = extract_quarter(breadcrumb, path.name)
        # The breadcrumb rides on every part, so it counts against the budget.
        used = estimate_tokens(breadcrumb)

        items, big_tables = _section_items(section.body)
        prose_parts = _pack_items(items, used, doc_id, breadcrumb)
        table_groups: list[str] = []
        for table_text in big_tables:
            table_groups.extend(_split_table_rows(table_text, used, doc_id, breadcrumb))

        chunk_id_base = f"{doc_id}#{slug}"

        for index, part in enumerate(prose_parts, start=1):
            chunk_id = chunk_id_base
            if len(prose_parts) > 1:
                chunk_id = f"{chunk_id_base}/p{index}"
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

        for index, group in enumerate(table_groups, start=1):
            chunks.append(
                Chunk(
                    chunk_id=f"{chunk_id_base}/t{index}",
                    doc_id=doc_id,
                    section_path=breadcrumb,
                    section_slug=slug,
                    text=f"{breadcrumb}\n\n{group}",
                    owner_dept=owner_dept,
                    doc_type="md",
                    sensitivity=sensitivity,
                    allowed_roles=(),
                    quarter=quarter,
                )
            )
    return chunks
