from __future__ import annotations
import os
import re
from typing import Dict, List, Optional

from app.ingest.metadata_extractor import extract_temporal_metadata, extract_topics

# Rough token estimation: ~4 chars per token for English
CHARS_PER_TOKEN = 4
DEFAULT_MAX_TOKENS = 512
DEFAULT_OVERLAP_TOKENS = 50


def estimate_tokens(text: str) -> int:
    """Estimate token count from text length."""
    return len(text) // CHARS_PER_TOKEN


def split_text_recursive(
    text: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    separators: Optional[List[str]] = None,
) -> List[str]:
    """
    Recursively split text to fit within max_tokens.
    Tries separators in order: paragraphs, sentences, words.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]

    if estimate_tokens(text) <= max_tokens:
        return [text.strip()] if text.strip() else []

    # Try each separator
    for sep in separators:
        parts = text.split(sep)
        if len(parts) == 1:
            continue  # This separator didn't help

        chunks = []
        current = ""

        for part in parts:
            candidate = current + sep + part if current else part

            if estimate_tokens(candidate) <= max_tokens:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())
                # If this single part is too large, recurse with finer separator
                if estimate_tokens(part) > max_tokens:
                    remaining_seps = separators[separators.index(sep) + 1:]
                    if remaining_seps:
                        chunks.extend(split_text_recursive(part, max_tokens, overlap_tokens, remaining_seps))
                    else:
                        # Last resort: hard split by characters
                        chunk_size = max_tokens * CHARS_PER_TOKEN
                        for i in range(0, len(part), chunk_size):
                            chunks.append(part[i:i + chunk_size].strip())
                else:
                    current = part

        if current:
            chunks.append(current.strip())

        # Add overlap between chunks for context continuity
        if overlap_tokens > 0 and len(chunks) > 1:
            overlap_chars = overlap_tokens * CHARS_PER_TOKEN
            overlapped = [chunks[0]]
            for i in range(1, len(chunks)):
                prev = chunks[i - 1]
                if len(prev) > overlap_chars:
                    # Prepend end of previous chunk as context
                    prefix = prev[-overlap_chars:]
                    overlapped.append(prefix + " " + chunks[i])
                else:
                    overlapped.append(chunks[i])
            chunks = overlapped

        return [c for c in chunks if c]

    # Fallback: hard character split
    chunk_size = max_tokens * CHARS_PER_TOKEN
    return [text[i:i + chunk_size].strip() for i in range(0, len(text), chunk_size) if text[i:i + chunk_size].strip()]


def chunk_markdown(
    text: str,
    source_path: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> List[Dict]:
    """
    Heading-aware chunker for Markdown with token limits.

    - Starts a new chunk on headings (lines that start with one or more '# ')
    - Tracks fenced code blocks, does not treat headings inside fences as headings
    - Includes heading text in chunk body so embeddings and BM25 see it
    - Builds section_path as a breadcrumb from H1..Hn
    - Recursively splits chunks that exceed max_tokens
    - Extracts temporal metadata per chunk (not just per document)

    Returns list of dicts: {"text": str, "section_path": str, "source_path": str,
                           "year": str|None, "quarter": str|None, "topics": list}
    """
    # Extract document-level temporal metadata (used as fallback)
    filename = os.path.basename(source_path)
    doc_temporal = extract_temporal_metadata(text, filename)

    lines = text.splitlines()
    raw_chunks: List[Dict] = []
    in_code = False
    headings: List[str] = []  # indexed by level-1; e.g., h1 at idx0

    cur_lines: List[str] = []

    def flush():
        body = "\n".join(cur_lines).strip()
        if body:
            section = " > ".join([h for h in headings if h]) or ""
            # Use filename stem as fallback section for preamble content
            if not section:
                stem = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ")
                section = stem.title()
            raw_chunks.append({
                "text": body,
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
                # Include heading text in chunk body so embeddings see it
                cur_lines.append(title)
                continue

        # Regular content line
        cur_lines.append(line)

    # Flush last chunk
    flush()

    # Remove empty-text chunks and apply token limits
    final_chunks: List[Dict] = []
    for chunk in raw_chunks:
        if not chunk["text"]:
            continue

        # Check if chunk needs splitting
        if estimate_tokens(chunk["text"]) > max_tokens:
            sub_texts = split_text_recursive(chunk["text"], max_tokens, overlap_tokens)
            for idx, sub_text in enumerate(sub_texts):
                final_chunks.append({
                    "text": sub_text,
                    "section_path": chunk["section_path"],
                    "source_path": chunk["source_path"],
                    "chunk_index": idx,  # Track sub-chunk position
                })
        else:
            final_chunks.append(chunk)

    # Enrich all chunks with temporal metadata and topics
    for chunk in final_chunks:
        # Chunk-level temporal extraction (from the chunk's own text)
        chunk_temporal = extract_temporal_metadata(chunk["text"], filename)
        chunk["temporal_refs"] = chunk_temporal["mentioned_quarters"]

        # Use chunk-level quarter when available. If the chunk itself mentions
        # a specific quarter, use that. Otherwise check the section_path
        # breadcrumb (e.g., "Q2 - April to June"). Fall back to document-level
        # only as a last resort.
        chunk_quarter = chunk_temporal["primary_quarter"]
        if not chunk_quarter:
            sp_match = re.search(r'\bQ([1-4])\b', chunk.get("section_path", ""))
            if sp_match:
                chunk_quarter = f"Q{sp_match.group(1)}"
        if not chunk_quarter:
            chunk_quarter = doc_temporal["primary_quarter"]
        chunk["quarter"] = chunk_quarter

        # Same for year: prefer chunk-level, fall back to document-level
        chunk["year"] = chunk_temporal["primary_year"] or doc_temporal["primary_year"]

        # Extract topics from chunk text
        chunk["topics"] = extract_topics(chunk["text"])

    return final_chunks
