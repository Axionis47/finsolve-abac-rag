from app.ingest.chunker import chunk_markdown


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

