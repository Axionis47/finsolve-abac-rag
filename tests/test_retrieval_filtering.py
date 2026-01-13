from app.retrieval.filtering import prefilter_by_allowed_roles
from app.retrieval.bm25 import BM25, bm25_search, tokenize
from app.retrieval.hybrid import rrf_fuse, make_key


def test_prefilter_by_allowed_roles():
    items = [
        {"id": 1, "allowed_roles": ["engineering", "c_level"]},
        {"id": 2, "allowed_roles": ["hr", "c_level"]},
        {"id": 3, "allowed_roles": ["marketing", "employee"]},
        {"id": 4, "allowed_roles": []},
    ]
    eng = prefilter_by_allowed_roles(items, "engineering")
    assert {x["id"] for x in eng} == {1}
    hr = prefilter_by_allowed_roles(items, "hr")
    assert {x["id"] for x in hr} == {2}
    emp = prefilter_by_allowed_roles(items, "employee")
    assert {x["id"] for x in emp} == {3}


def test_prefilter_c_level_access():
    """C-level should see items where c_level is in allowed_roles."""
    items = [
        {"id": 1, "allowed_roles": ["engineering", "c_level"]},
        {"id": 2, "allowed_roles": ["hr", "c_level"]},
        {"id": 3, "allowed_roles": ["marketing", "employee"]},
    ]
    c_level = prefilter_by_allowed_roles(items, "c_level")
    assert {x["id"] for x in c_level} == {1, 2}


def test_prefilter_empty_items():
    """Empty items list returns empty."""
    assert prefilter_by_allowed_roles([], "any_role") == []


# ============================================================================
# BM25 Tests
# ============================================================================


def test_tokenize_basic():
    """Test basic tokenization."""
    assert tokenize("Hello World") == ["hello", "world"]
    assert tokenize("hello_world123") == ["hello_world123"]
    assert tokenize("") == []
    assert tokenize(None) == []


def test_bm25_basic_ranking():
    """Test BM25 ranks documents by term frequency and relevance."""
    docs = [
        "apple banana cherry",
        "apple apple apple",
        "banana banana",
    ]
    bm = BM25(docs)
    # Query for "apple" should rank doc 1 (3 apples) highest
    top = bm.topn("apple", 3)
    assert top[0] == 1  # doc with 3 apples


def test_bm25_search_function():
    """Test bm25_search wrapper function."""
    items = [
        {"text": "The quick brown fox", "id": 1},
        {"text": "Fox jumps over lazy dog", "id": 2},
        {"text": "Hello world", "id": 3},
    ]
    results = bm25_search(items, "fox", top_k=2)
    assert len(results) == 2
    # Both docs with "fox" should be returned
    assert all("fox" in r["text"].lower() for r in results)


def test_bm25_no_matches():
    """Test BM25 returns empty when no matches."""
    items = [
        {"text": "apple banana", "id": 1},
        {"text": "cherry date", "id": 2},
    ]
    results = bm25_search(items, "xyz123", top_k=5)
    assert results == []


# ============================================================================
# Hybrid Search / RRF Tests
# ============================================================================


def test_make_key():
    """Test key generation for deduplication."""
    item = {"source_path": "/path/doc.md", "section_path": "Section A"}
    assert make_key(item) == "/path/doc.md#Section A"

    item_no_section = {"source_path": "/path/doc.md"}
    assert make_key(item_no_section) == "/path/doc.md#"


def test_rrf_fuse_basic():
    """Test RRF fusion combines dense and sparse results."""
    dense = [
        {"source_path": "a.md", "section_path": "A", "text": "doc A"},
        {"source_path": "b.md", "section_path": "B", "text": "doc B"},
    ]
    sparse = [
        {"source_path": "b.md", "section_path": "B", "text": "doc B"},
        {"source_path": "c.md", "section_path": "C", "text": "doc C"},
    ]
    fused = rrf_fuse(dense, sparse, k=60, top_k=3)
    # doc B appears in both lists, should have highest score
    assert fused[0]["source_path"] == "b.md"
    assert len(fused) == 3


def test_rrf_fuse_empty_inputs():
    """Test RRF fusion with empty inputs."""
    assert rrf_fuse([], [], top_k=5) == []
    dense = [{"source_path": "a.md", "section_path": "A", "text": "doc A"}]
    assert len(rrf_fuse(dense, [], top_k=5)) == 1
    assert len(rrf_fuse([], dense, top_k=5)) == 1


def test_rrf_fuse_respects_top_k():
    """Test RRF fusion respects top_k limit."""
    dense = [
        {"source_path": f"{i}.md", "section_path": str(i), "text": f"doc {i}"}
        for i in range(10)
    ]
    fused = rrf_fuse(dense, [], k=60, top_k=3)
    assert len(fused) == 3

