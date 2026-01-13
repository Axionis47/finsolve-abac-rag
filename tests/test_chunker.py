from app.ingest.chunker import chunk_markdown, estimate_tokens, split_text_recursive


def test_chunk_markdown_basic_headings_and_code_blocks():
    md = """
# Title
Intro paragraph.

## Section A
Text A1

```
# Not a heading inside code block
print("hello")
```

More A text.

### Sub A1
Details under sub.

## Section B
Text B1
""".strip()

    chunks = chunk_markdown(md, source_path="/tmp/doc.md")
    # Expect chunks: for Title (Intro paragraph), Section A (text + code + more), Sub A1, Section B
    sections = [c["section_path"] for c in chunks]
    assert sections[:2] == ["Title", "Title > Section A"]
    assert any("Not a heading inside code block" in c["text"] for c in chunks)
    # Sub-section breadcrumb should include parent
    assert any(c["section_path"] == "Title > Section A > Sub A1" for c in chunks)
    # Last section B
    assert sections[-1] == "Title > Section B"


# ============================================================================
# Additional tests for complete chunker coverage
# ============================================================================


def test_estimate_tokens():
    """Test token estimation (4 chars per token)."""
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("12345678") == 2
    assert estimate_tokens("a" * 100) == 25


def test_split_text_recursive_small_text():
    """Text under token limit is returned as-is."""
    result = split_text_recursive("Short text", max_tokens=100)
    assert result == ["Short text"]


def test_split_text_recursive_with_paragraphs():
    """Text is split on paragraph boundaries when exceeding token limit."""
    # Each paragraph is ~20+ chars = 5+ tokens. 3 paragraphs = 15+ tokens
    # Set max_tokens=8 to force splitting
    para1 = "Para one with some extra text here."  # ~35 chars = ~9 tokens
    para2 = "Para two with additional content."     # ~33 chars = ~8 tokens
    para3 = "Para three ends here nicely."          # ~30 chars = ~7 tokens
    text = f"{para1}\n\n{para2}\n\n{para3}"
    result = split_text_recursive(text, max_tokens=10)
    # With max 10 tokens (~40 chars), should split into multiple chunks
    assert len(result) >= 2


def test_split_text_recursive_empty_text():
    """Empty text returns empty list."""
    assert split_text_recursive("") == []
    assert split_text_recursive("   ") == []


def test_chunk_markdown_empty_input():
    """Empty markdown returns empty chunks."""
    chunks = chunk_markdown("", source_path="/tmp/empty.md")
    assert chunks == []


def test_chunk_markdown_no_headings():
    """Markdown without headings still returns chunks."""
    md = "Just some plain text without any headings."
    chunks = chunk_markdown(md, source_path="/tmp/plain.md")
    assert len(chunks) == 1
    assert chunks[0]["text"] == "Just some plain text without any headings."
    assert chunks[0]["section_path"] == ""


def test_chunk_markdown_preserves_source_path():
    """Source path is preserved in all chunks."""
    md = "# Title\nContent"
    chunks = chunk_markdown(md, source_path="/path/to/doc.md")
    assert all(c["source_path"] == "/path/to/doc.md" for c in chunks)


def test_chunk_markdown_nested_headings():
    """Deeply nested headings produce correct section paths."""
    md = """
# H1
## H2
### H3
#### H4
Content at H4 level
""".strip()
    chunks = chunk_markdown(md, source_path="/tmp/doc.md")
    # Find the chunk with H4 content
    h4_chunk = next((c for c in chunks if "Content at H4 level" in c["text"]), None)
    assert h4_chunk is not None
    assert h4_chunk["section_path"] == "H1 > H2 > H3 > H4"


def test_chunk_markdown_large_section_splits():
    """Large sections are split to respect token limits."""
    # Create a section larger than 512 tokens (512 * 4 = 2048 chars)
    large_text = "A" * 3000
    md = f"# Title\n{large_text}"
    chunks = chunk_markdown(md, source_path="/tmp/large.md", max_tokens=512)
    # Should produce multiple chunks
    assert len(chunks) >= 2
    # All chunks should have same section path
    assert all(c["section_path"] == "Title" for c in chunks)
    # Chunks should have chunk_index for sub-chunks
    assert any("chunk_index" in c for c in chunks)

