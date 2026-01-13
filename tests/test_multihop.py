"""
Tests for Multi-Hop RAG functionality.

Tests cover:
1. Query complexity detection
2. Query decomposition
3. Context deduplication
4. Multi-hop retrieval flow
5. Integration with chat endpoint
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import base64

from app.services.multihop import (
    is_complex_query,
    decompose_query,
    deduplicate_contexts,
    deduplicate_citations,
    multihop_retrieve,
    multihop_query,
)


class TestQueryComplexityDetection:
    """Test the is_complex_query function."""
    
    def test_simple_query(self):
        """Simple queries should not be detected as complex."""
        assert not is_complex_query("What is the dress code?")
        assert not is_complex_query("Tell me about vacation policy")
        assert not is_complex_query("How do I submit expenses?")
    
    def test_comparison_query(self):
        """Queries with comparison keywords should be complex."""
        assert is_complex_query("Compare Q3 revenue to Q2")
        assert is_complex_query("What is the difference between premium and basic accounts?")
        assert is_complex_query("Marketing spend vs engineering budget")
    
    def test_multi_quarter_query(self):
        """Queries spanning multiple time periods should be complex."""
        assert is_complex_query("What was the revenue trend from Q1 to Q3?")
        assert is_complex_query("Compare Q2 and Q3 performance")
    
    def test_relationship_query(self):
        """Queries about relationships between concepts should be complex."""
        assert is_complex_query("How does marketing spend affect revenue?")
        assert is_complex_query("What is the relationship between employee count and productivity?")
    
    def test_multiple_questions(self):
        """Multiple question marks indicate complexity."""
        assert is_complex_query("What was Q3 revenue? And what was marketing spend?")
    
    def test_long_and_query(self):
        """Long queries with 'and' connecting concepts should be complex."""
        assert is_complex_query("What is the revenue and what is the marketing budget and how did they change?")


class TestQueryDecomposition:
    """Test the decompose_query function."""
    
    @patch('app.services.multihop.get_llm')
    def test_decomposition_success(self, mock_get_llm):
        """Test successful query decomposition."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '["What was Q2 revenue?", "What was Q3 revenue?"]'
        mock_llm.generate.return_value = mock_response
        mock_get_llm.return_value = mock_llm
        
        result = decompose_query("Compare Q2 and Q3 revenue")
        
        assert len(result) == 2
        assert "Q2" in result[0]
        assert "Q3" in result[1]
    
    @patch('app.services.multihop.get_llm')
    def test_decomposition_with_markdown(self, mock_get_llm):
        """Test decomposition when LLM wraps response in markdown."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '```json\n["Query 1", "Query 2"]\n```'
        mock_llm.generate.return_value = mock_response
        mock_get_llm.return_value = mock_llm
        
        result = decompose_query("Complex query")
        
        assert len(result) == 2
    
    @patch('app.services.multihop.get_llm')
    def test_decomposition_fallback(self, mock_get_llm):
        """Test fallback to original query on error."""
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = Exception("LLM error")
        mock_get_llm.return_value = mock_llm
        
        result = decompose_query("My complex query")
        
        assert result == ["My complex query"]
    
    @patch('app.services.multihop.get_llm')
    def test_decomposition_max_subqueries(self, mock_get_llm):
        """Test that decomposition respects max_subqueries limit."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '["Q1", "Q2", "Q3", "Q4", "Q5"]'
        mock_llm.generate.return_value = mock_response
        mock_get_llm.return_value = mock_llm
        
        result = decompose_query("Query", max_subqueries=3)
        
        assert len(result) == 3


class TestContextDeduplication:
    """Test context and citation deduplication."""
    
    def test_deduplicate_contexts(self):
        """Test removing duplicate contexts."""
        contexts = [
            {"source_path": "doc1.md", "section_path": "Section A", "text": "Text 1"},
            {"source_path": "doc2.md", "section_path": "Section B", "text": "Text 2"},
            {"source_path": "doc1.md", "section_path": "Section A", "text": "Text 1 duplicate"},
        ]
        
        result = deduplicate_contexts(contexts)
        
        assert len(result) == 2
        assert result[0]["text"] == "Text 1"  # First occurrence kept
    
    def test_deduplicate_citations(self):
        """Test removing duplicate citations."""
        citations = [
            {"source_path": "doc1.md", "section_path": "Section A", "rule": "rule1"},
            {"source_path": "doc1.md", "section_path": "Section A", "rule": "rule2"},
        ]

        result = deduplicate_citations(citations)

        assert len(result) == 1


class TestMultihopRetrieve:
    """Test the multi-hop retrieval function."""

    def test_multihop_retrieve_basic(self):
        """Test basic multi-hop retrieval."""
        def mock_retrieve(query):
            return (
                [{"source_path": f"doc_{query}.md", "section_path": "S1", "text": f"Content for {query}"}],
                [{"source_path": f"doc_{query}.md", "section_path": "S1", "rule": "test"}]
            )

        contexts, citations, metrics = multihop_retrieve(
            ["query1", "query2"],
            mock_retrieve,
        )

        assert len(contexts) == 2
        assert len(citations) == 2
        assert "sub_queries" in metrics
        assert metrics["contexts_after_dedup"] == 2

    def test_multihop_retrieve_deduplication(self):
        """Test that duplicate contexts are removed."""
        def mock_retrieve(query):
            # Both queries return the same document
            return (
                [{"source_path": "same_doc.md", "section_path": "S1", "text": "Same content"}],
                [{"source_path": "same_doc.md", "section_path": "S1", "rule": "test"}]
            )

        contexts, citations, metrics = multihop_retrieve(
            ["query1", "query2"],
            mock_retrieve,
        )

        assert len(contexts) == 1  # Deduplicated
        assert metrics["contexts_before_dedup"] == 2
        assert metrics["contexts_after_dedup"] == 1

    def test_multihop_retrieve_handles_errors(self):
        """Test that errors in retrieval are handled gracefully."""
        call_count = [0]

        def mock_retrieve(query):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Retrieval error")
            return (
                [{"source_path": "doc.md", "section_path": "S1", "text": "Content"}],
                [{"source_path": "doc.md", "section_path": "S1", "rule": "test"}]
            )

        contexts, citations, metrics = multihop_retrieve(
            ["query1", "query2"],
            mock_retrieve,
        )

        # Should still have results from successful query
        assert len(contexts) >= 1


class TestMultihopQuery:
    """Test the main multihop_query function."""

    def test_simple_query_no_multihop(self):
        """Simple queries should not trigger multi-hop."""
        def mock_retrieve(query):
            return (
                [{"source_path": "doc.md", "section_path": "S1", "text": "Content"}],
                [{"source_path": "doc.md", "section_path": "S1", "rule": "test"}]
            )

        contexts, citations, metrics = multihop_query(
            mock_retrieve,
            "What is the dress code?",
            auto_detect=True,
        )

        assert metrics["is_complex"] is False
        assert metrics["used_multihop"] is False

    @patch('app.services.multihop.decompose_query')
    def test_complex_query_uses_multihop(self, mock_decompose):
        """Complex queries should trigger multi-hop."""
        mock_decompose.return_value = ["Sub-query 1", "Sub-query 2"]

        def mock_retrieve(query):
            return (
                [{"source_path": f"doc_{query[:5]}.md", "section_path": "S1", "text": f"Content for {query}"}],
                [{"source_path": f"doc_{query[:5]}.md", "section_path": "S1", "rule": "test"}]
            )

        contexts, citations, metrics = multihop_query(
            mock_retrieve,
            "Compare Q2 and Q3 revenue",
            auto_detect=True,
        )

        assert metrics["is_complex"] is True
        assert metrics["used_multihop"] is True
        assert len(metrics["sub_queries"]) == 2


class TestChatEndpointIntegration:
    """Test multi-hop integration with the chat endpoint."""

    @pytest.fixture
    def client(self):
        from app.main import app
        return TestClient(app)

    def basic_auth(self, user: str, password: str):
        token = base64.b64encode(f"{user}:{password}".encode()).decode()
        return {"Authorization": f"Basic {token}"}

    def test_chat_without_multihop(self, client):
        """Test that chat works without multihop flag."""
        response = client.post(
            "/chat",
            json={"message": "What is the dress code?"},
            headers=self.basic_auth("Tony", "password123"),
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "multihop" not in data  # No multihop metrics when disabled

    def test_chat_with_multihop_simple_query(self, client):
        """Test multihop with a simple query (should detect and skip)."""
        response = client.post(
            "/chat",
            json={"message": "What is the dress code?", "multihop": True, "multihop_auto": True},
            headers=self.basic_auth("Tony", "password123"),
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        # With auto_detect, simple queries won't show multihop metrics
        if "multihop" in data:
            assert data["multihop"]["used_multihop"] is False

