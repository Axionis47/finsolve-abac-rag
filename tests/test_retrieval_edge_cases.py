"""
Edge case tests for the retrieval pipeline.

Tests probe:
- BM25 with empty/degenerate inputs
- Prefilter with missing metadata
- RRF fusion with empty/duplicate inputs
- Rerank with empty candidates
"""
from app.retrieval.bm25 import BM25, bm25_search, tokenize
from app.retrieval.filtering import prefilter_by_allowed_roles
from app.retrieval.hybrid import rrf_fuse, make_key
from app.retrieval.rerank import rerank


class TestBM25EdgeCases:
    """BM25 with degenerate inputs."""

    def test_empty_corpus(self):
        """Empty document list should not crash."""
        bm = BM25([])
        result = bm.topn("hello", 5)
        assert result == []

    def test_empty_query(self):
        """Empty query should return no results (no tokens to score)."""
        bm = BM25(["hello world", "foo bar"])
        result = bm.topn("", 5)
        assert result == []

    def test_unicode_only_query(self):
        """Emoji/CJK chars are not matched by TOKEN_RE [A-Za-z0-9_]."""
        tokens = tokenize("🔥💯🚀")
        assert tokens == []
        # BM25 search with emoji-only query
        results = bm25_search(
            [{"text": "hello world"}, {"text": "foo bar"}],
            "🔥💯",
            top_k=5,
        )
        assert results == []

    def test_all_punctuation_query(self):
        """Punctuation-only query should produce no tokens."""
        tokens = tokenize("!@#$%^&*()")
        assert tokens == []

    def test_single_doc_single_token(self):
        """Minimal case: one doc, one token."""
        bm = BM25(["hello"])
        result = bm.topn("hello", 1)
        assert result == [0]

    def test_query_token_not_in_corpus(self):
        """Query with terms not in any doc should return empty."""
        bm = BM25(["hello world", "foo bar"])
        result = bm.topn("zzzzz", 5)
        assert result == []

    def test_bm25_search_empty_items(self):
        """bm25_search with empty items list should return empty."""
        results = bm25_search([], "hello", top_k=5)
        assert results == []

    def test_bm25_search_items_missing_text(self):
        """Items without 'text' key should use empty string."""
        results = bm25_search([{"source": "a"}, {"source": "b"}], "hello", top_k=5)
        assert results == []


class TestPrefilterEdgeCases:
    """Prefilter with missing/degenerate metadata."""

    def test_no_allowed_roles_key(self):
        """Item without 'allowed_roles' should be filtered out."""
        items = [{"text": "hello", "source_path": "a.md"}]
        result = prefilter_by_allowed_roles(items, "engineering")
        assert result == []

    def test_allowed_roles_none(self):
        """allowed_roles=None should be treated as empty list."""
        items = [{"text": "hello", "allowed_roles": None}]
        result = prefilter_by_allowed_roles(items, "engineering")
        assert result == []

    def test_allowed_roles_empty_list(self):
        """allowed_roles=[] should match no role."""
        items = [{"text": "hello", "allowed_roles": []}]
        result = prefilter_by_allowed_roles(items, "engineering")
        assert result == []

    def test_role_in_allowed_roles(self):
        """Normal case: role is in allowed_roles."""
        items = [{"text": "hello", "allowed_roles": ["engineering", "c_level"]}]
        result = prefilter_by_allowed_roles(items, "engineering")
        assert len(result) == 1

    def test_empty_items_list(self):
        """Empty items list returns empty."""
        result = prefilter_by_allowed_roles([], "engineering")
        assert result == []


class TestRRFFusionEdgeCases:
    """RRF fusion with degenerate inputs."""

    def _item(self, src, sec="s1"):
        return {"source_path": src, "section_path": sec, "text": f"text from {src}"}

    def test_empty_dense(self):
        """Only sparse results should be returned."""
        sparse = [self._item("a.md"), self._item("b.md")]
        result = rrf_fuse([], sparse, top_k=5)
        assert len(result) == 2

    def test_empty_sparse(self):
        """Only dense results should be returned."""
        dense = [self._item("a.md"), self._item("b.md")]
        result = rrf_fuse(dense, [], top_k=5)
        assert len(result) == 2

    def test_both_empty(self):
        """Both empty should return empty list."""
        result = rrf_fuse([], [], top_k=5)
        assert result == []

    def test_dedup_same_source(self):
        """Same source_path#section_path in both lists should be deduplicated."""
        item = self._item("doc.md", "intro")
        result = rrf_fuse([item], [item], top_k=5)
        assert len(result) == 1

    def test_top_k_larger_than_results(self):
        """top_k > total items returns all available."""
        items = [self._item(f"{i}.md") for i in range(3)]
        result = rrf_fuse(items, [], top_k=100)
        assert len(result) == 3

    def test_top_k_zero(self):
        """top_k=0 returns empty list."""
        items = [self._item("a.md")]
        result = rrf_fuse(items, items, top_k=0)
        assert result == []

    def test_make_key_missing_fields(self):
        """make_key with missing fields should return '#' separator."""
        key = make_key({})
        assert key == "#"

    def test_make_key_none_fields(self):
        """make_key with None values converts to 'None' string via f-string."""
        key = make_key({"source_path": None, "section_path": None})
        # .get() returns None (not default '') when key exists with None value
        assert key == "None#None"


class TestRerankEdgeCases:
    """Rerank with degenerate inputs."""

    def test_empty_candidates(self):
        """Reranking empty list should return empty list."""
        result = rerank("hello", [], backend="simple", top_k=5)
        assert result == []

    def test_single_candidate(self):
        """Single candidate should be returned as-is."""
        items = [{"text": "hello world", "source_path": "a.md"}]
        result = rerank("hello", items, backend="simple", top_k=5)
        assert len(result) == 1
