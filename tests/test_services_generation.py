"""
Tests for LLM generation service.
"""
import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

from app.services.generation import generate_answer, _build_chat_prompt


@dataclass
class MockLLMResponse:
    """Mock LLM response."""
    text: str


class MockLLM:
    """Mock LLM backend."""
    model = "mock-model"

    def __init__(self, response_text: str = "Generated answer"):
        self.response_text = response_text
        self.call_count = 0
        self.last_messages = None

    def generate(self, messages, **kwargs):
        self.call_count += 1
        self.last_messages = messages
        return MockLLMResponse(text=self.response_text)


class TestBuildChatPrompt:
    """Tests for prompt building."""

    def test_builds_system_prompt(self):
        """Prompt includes system message with instructions."""
        messages = _build_chat_prompt("test query", [])
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "careful assistant" in messages[0]["content"].lower()

    def test_includes_citation_instructions(self):
        """System prompt includes citation instructions."""
        messages = _build_chat_prompt("test query", [])
        system = messages[0]["content"]
        assert "cite" in system.lower() or "citation" in system.lower()

    def test_includes_query_in_user_message(self):
        """User message includes the query."""
        messages = _build_chat_prompt("What is the policy?", [])
        user = messages[1]["content"]
        assert "What is the policy?" in user

    def test_includes_snippets_with_sources(self):
        """User message includes snippets with source info."""
        snippets = [
            {"text": "Snippet 1 content", "source_path": "doc1.md", "section_path": "Section A"},
            {"text": "Snippet 2 content", "source_path": "doc2.md", "section_path": "Section B"},
        ]
        messages = _build_chat_prompt("query", snippets)
        user = messages[1]["content"]
        assert "Snippet 1 content" in user
        assert "doc1.md" in user
        assert "[1]" in user
        assert "[2]" in user


class TestGenerateAnswer:
    """Tests for answer generation."""

    def test_returns_empty_context_message_when_no_snippets(self):
        """Returns default message when no snippets provided."""
        result = generate_answer("test", [], use_cache=False)
        assert "don't know" in result.lower()

    def test_calls_llm_with_messages(self, monkeypatch):
        """Calls LLM backend with properly formatted messages."""
        mock_llm = MockLLM(response_text="Test answer")
        monkeypatch.setattr("app.services.providers.get_llm", lambda: mock_llm)

        snippets = [{"text": "Context", "source_path": "test.md", "section_path": "Sec"}]
        result = generate_answer("What is this?", snippets, use_cache=False)

        assert mock_llm.call_count == 1
        assert "Test answer" in result

    def test_returns_extractive_fallback_on_llm_error(self, monkeypatch):
        """Returns extractive answer when LLM fails."""
        class FailingLLM:
            model = "failing-model"
            def generate(self, messages, **kwargs):
                raise RuntimeError("LLM API Error")

        monkeypatch.setattr("app.services.providers.get_llm", lambda: FailingLLM())

        snippets = [{"text": "Fallback content here", "source_path": "doc.md", "section_path": "Sec"}]
        result = generate_answer("query", snippets, use_cache=False)

        assert "LLM Error" in result
        assert "Fallback content" in result

    def test_uses_cache_when_enabled(self, monkeypatch):
        """Uses RAG cache fallback when no PF provided (legacy path)."""
        mock_llm = MockLLM(response_text="Fresh answer")
        monkeypatch.setattr("app.services.providers.get_llm", lambda: mock_llm)

        # Mock RAG cache returning a cached answer
        mock_rag_cache = MagicMock()
        mock_rag_cache.get_llm_answer.return_value = ("Cached answer", True)
        monkeypatch.setattr("app.services.generation.get_rag_cache", lambda: mock_rag_cache)

        snippets = [{"text": "Context", "source_path": "test.md", "section_path": "Sec"}]
        result = generate_answer("query", snippets, use_cache=True)

        # Should return cached answer via RAG cache fallback path
        assert result == "Cached answer"
        assert mock_rag_cache.get_llm_answer.called

    def test_computes_fresh_on_cache_miss(self, monkeypatch):
        """Computes fresh answer on cache miss via RAG cache fallback."""
        mock_llm = MockLLM(response_text="Fresh answer")
        monkeypatch.setattr("app.services.providers.get_llm", lambda: mock_llm)

        # Mock RAG cache that calls through to compute_fn
        mock_rag_cache = MagicMock()
        mock_rag_cache.get_llm_answer.side_effect = lambda pf, query, context_hash, model, compute_fn: (compute_fn(), False)
        monkeypatch.setattr("app.services.generation.get_rag_cache", lambda: mock_rag_cache)

        snippets = [{"text": "Context", "source_path": "test.md", "section_path": "Sec"}]
        result = generate_answer("query", snippets, use_cache=True)

        assert "Fresh answer" in result
        assert mock_llm.call_count == 1

    def test_uses_rag_cache_with_policy_fingerprint(self, monkeypatch):
        """Uses RAG cache when policy fingerprint provided."""
        mock_llm = MockLLM(response_text="PF-cached answer")
        monkeypatch.setattr("app.services.providers.get_llm", lambda: mock_llm)

        # Mock RAG cache
        mock_rag_cache = MagicMock()
        mock_rag_cache.get_llm_answer.return_value = ("Cached via PF", True)
        monkeypatch.setattr("app.services.generation.get_rag_cache", lambda: mock_rag_cache)

        # Create a mock PF
        mock_pf = MagicMock()
        mock_pf.hash = "test-pf-hash"

        snippets = [{"text": "Context", "source_path": "test.md", "section_path": "Sec"}]
        result = generate_answer("query", snippets, use_cache=True, pf=mock_pf)

        assert result == "Cached via PF"
        assert mock_rag_cache.get_llm_answer.called

    def test_skips_cache_when_disabled(self, monkeypatch):
        """Skips cache entirely when use_cache=False."""
        mock_llm = MockLLM(response_text="Direct answer")
        monkeypatch.setattr("app.services.providers.get_llm", lambda: mock_llm)

        # Mock RAG cache that should NOT be called
        mock_rag_cache = MagicMock()
        monkeypatch.setattr("app.services.generation.get_rag_cache", lambda: mock_rag_cache)

        snippets = [{"text": "Context", "source_path": "test.md", "section_path": "Sec"}]
        result = generate_answer("query", snippets, use_cache=False)

        assert "Direct answer" in result
        assert not mock_rag_cache.get_llm_answer.called


class TestGenerateAnswerEdgeCases:
    """Edge case tests for answer generation."""

    def test_handles_empty_text_in_snippet(self, monkeypatch):
        """Handles snippets with empty or None text."""
        mock_llm = MockLLM(response_text="Answer")
        monkeypatch.setattr("app.services.providers.get_llm", lambda: mock_llm)

        snippets = [
            {"text": "", "source_path": "doc.md", "section_path": "Sec"},
            {"text": None, "source_path": "doc2.md", "section_path": "Sec2"},
        ]
        result = generate_answer("query", snippets, use_cache=False)
        assert "Answer" in result

    def test_handles_missing_source_path(self, monkeypatch):
        """Handles snippets without source_path."""
        mock_llm = MockLLM(response_text="Answer")
        monkeypatch.setattr("app.services.providers.get_llm", lambda: mock_llm)

        snippets = [{"text": "Content"}]  # No source_path or section_path
        result = generate_answer("query", snippets, use_cache=False)
        assert "Answer" in result

    def test_strips_whitespace_from_response(self, monkeypatch):
        """Strips whitespace from LLM response."""
        mock_llm = MockLLM(response_text="  Answer with spaces  \n\n")
        monkeypatch.setattr("app.services.providers.get_llm", lambda: mock_llm)

        snippets = [{"text": "Context", "source_path": "doc.md", "section_path": "Sec"}]
        result = generate_answer("query", snippets, use_cache=False)
        assert result == "Answer with spaces"
