from __future__ import annotations
from typing import Dict, List


def chunk_markdown(text: str, source_path: str) -> List[Dict]:
    """
    Heading-aware chunker for Markdown.
    - Starts a new chunk on headings (lines that start with one or more '# ')
    - Tracks fenced code blocks (```), does not treat headings inside fences as headings
    - Builds section_path as a breadcrumb from H1..Hn
    Returns list of dicts: {"text": str, "section_path": str, "source_path": str}
    """
    lines = text.splitlines()
    chunks: List[Dict] = []
    in_code = False
    headings: List[str] = []  # indexed by level-1; e.g., h1 at idx0

    cur_lines: List[str] = []

    def flush():
        if cur_lines:
            section = " > ".join([h for h in headings if h]) or ""
            chunks.append({
                "text": "\n".join(cur_lines).strip(),
                "section_path": section,
                "source_path": source_path,
            })

    for raw in lines:
        line = raw.rstrip("\n")

        # Fence toggling (triple backticks). Do not try to parse language tag.
        if line.strip().startswith("```"):
            in_code = not in_code
            cur_lines.append(line)
            continue

        if not in_code and line.startswith("#"):
            # Count heading level (number of # before a space)
            i = 0
            while i < len(line) and line[i] == '#':
                i += 1
            if i < len(line) and i > 0 and line[i:i+1] == ' ':
                # Flush previous chunk before starting new section
                flush()
                cur_lines = []
                title = line[i+1:].strip() if len(line) > i+1 else ""
                # Update headings levels
                level = i
                # Ensure length
                if len(headings) < level:
                    headings += [""] * (level - len(headings))
                # Set this level and clear deeper ones
                headings[level-1] = title
                for j in range(level, len(headings)):
                    headings[j] = ""
                # Do not include the heading line itself in chunk text; chunks contain content under heading
                continue

        # Regular content line
        cur_lines.append(line)

    # Flush last chunk
    flush()
    # Remove empty-text chunks
    chunks = [c for c in chunks if c["text"]]
    return chunks

