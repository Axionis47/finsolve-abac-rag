"""
Tests for input validation schemas.
"""
import pytest
from pydantic import ValidationError

from app.schemas.validators import (
    ChatRequest,
    SearchRequest,
    HrRowsRequest,
    AdminReindexRequest,
    sanitize_text,
)


class TestSanitizeText:
    """Tests for sanitize_text function."""

    def test_removes_html_tags(self):
        """Removes HTML tags from input."""
        result = sanitize_text("<script>alert('xss')</script>Hello")
        assert "<script>" not in result
        assert "Hello" in result

    def test_strips_whitespace(self):
        """Strips leading and trailing whitespace."""
        result = sanitize_text("  hello world  ")
        assert result == "hello world"

    def test_normalizes_whitespace(self):
        """Normalizes internal whitespace."""
        result = sanitize_text("hello   world")
        assert result == "hello world"

    def test_handles_empty_string(self):
        """Handles empty string input."""
        result = sanitize_text("")
        assert result == ""

    def test_handles_none(self):
        """Handles None input gracefully."""
        result = sanitize_text(None)
        assert result is None

    def test_preserves_safe_characters(self):
        """Preserves safe special characters."""
        result = sanitize_text("Hello, world! How are you?")
        assert result == "Hello, world! How are you?"

    def test_removes_script_tags(self):
        """Removes script tags and content."""
        result = sanitize_text("before<script>bad()</script>after")
        assert "script" not in result.lower()
        assert "before" in result
        assert "after" in result

    def test_removes_html_attributes(self):
        """Removes tags with attributes."""
        result = sanitize_text('<div onclick="bad()">text</div>')
        assert "onclick" not in result.lower()
        assert "text" in result


class TestChatRequest:
    """Tests for ChatRequest model."""

    def test_valid_request(self):
        """Creates valid chat request."""
        req = ChatRequest(message="What is the vacation policy?")
        assert req.message == "What is the vacation policy?"
        assert req.top_k == 5  # Default

    def test_custom_top_k(self):
        """Accepts custom top_k value."""
        req = ChatRequest(message="Hello", top_k=10)
        assert req.top_k == 10

    def test_sanitizes_message(self):
        """Sanitizes message content."""
        req = ChatRequest(message="<script>bad</script>Hello")
        assert "<script>" not in req.message
        assert "Hello" in req.message

    def test_rejects_empty_message(self):
        """Rejects empty message."""
        with pytest.raises(ValidationError):
            ChatRequest(message="")

    def test_rejects_too_long_message(self):
        """Rejects message exceeding max length."""
        long_message = "a" * 10001
        with pytest.raises(ValidationError):
            ChatRequest(message=long_message)

    def test_top_k_minimum(self):
        """Enforces top_k minimum."""
        with pytest.raises(ValidationError):
            ChatRequest(message="Hello", top_k=0)

    def test_top_k_maximum(self):
        """Enforces top_k maximum."""
        with pytest.raises(ValidationError):
            ChatRequest(message="Hello", top_k=100)

    def test_optional_fields_defaults(self):
        """Optional fields have correct defaults."""
        req = ChatRequest(message="Hello")
        assert req.persist_dir is None
        assert req.base_dir is None
        assert req.rerank is False
        assert req.multihop is False
        assert req.multihop_auto is True

    def test_path_traversal_blocked(self):
        """Blocks path traversal attempts."""
        with pytest.raises(ValidationError):
            ChatRequest(message="Hello", persist_dir="../../../etc/passwd")


class TestSearchRequest:
    """Tests for SearchRequest model."""

    def test_valid_search_request(self):
        """Creates valid search request."""
        req = SearchRequest(query="search term")
        assert req.query == "search term"

    def test_sanitizes_query(self):
        """Sanitizes search query."""
        req = SearchRequest(query="<b>bold</b> search")
        assert "<b>" not in req.query
        assert "search" in req.query

    def test_default_top_k(self):
        """Default top_k value."""
        req = SearchRequest(query="test")
        assert req.top_k == 5

    def test_rerank_default(self):
        """Default rerank value."""
        req = SearchRequest(query="test")
        assert req.rerank is False

    def test_rejects_empty_query(self):
        """Rejects empty query."""
        with pytest.raises(ValidationError):
            SearchRequest(query="")

    def test_rejects_too_long_query(self):
        """Rejects query exceeding max length."""
        long_query = "a" * 5001
        with pytest.raises(ValidationError):
            SearchRequest(query=long_query)


class TestHrRowsRequest:
    """Tests for HrRowsRequest model."""

    def test_default_values(self):
        """Default limit and offset values."""
        req = HrRowsRequest()
        assert req.limit == 10
        assert req.offset == 0

    def test_custom_limit(self):
        """Accepts custom limit."""
        req = HrRowsRequest(limit=50)
        assert req.limit == 50

    def test_custom_offset(self):
        """Accepts custom offset."""
        req = HrRowsRequest(offset=100)
        assert req.offset == 100

    def test_limit_minimum(self):
        """Enforces limit minimum."""
        with pytest.raises(ValidationError):
            HrRowsRequest(limit=0)

    def test_limit_maximum(self):
        """Enforces limit maximum."""
        with pytest.raises(ValidationError):
            HrRowsRequest(limit=1001)

    def test_offset_minimum(self):
        """Enforces offset minimum."""
        with pytest.raises(ValidationError):
            HrRowsRequest(offset=-1)


class TestAdminReindexRequest:
    """Tests for AdminReindexRequest model."""

    def test_default_values(self):
        """Default values are None."""
        req = AdminReindexRequest()
        assert req.base_dir is None
        assert req.persist_dir is None

    def test_custom_paths(self):
        """Accepts custom paths."""
        req = AdminReindexRequest(base_dir="/data/docs", persist_dir="/data/chroma")
        assert req.base_dir == "/data/docs"
        assert req.persist_dir == "/data/chroma"

    def test_blocks_path_traversal_base_dir(self):
        """Blocks path traversal in base_dir."""
        with pytest.raises(ValidationError):
            AdminReindexRequest(base_dir="../../../etc")

    def test_blocks_path_traversal_persist_dir(self):
        """Blocks path traversal in persist_dir."""
        with pytest.raises(ValidationError):
            AdminReindexRequest(persist_dir="../../secret")
