"""
Edge case tests for LLM generation service.

Tests probe:
- Empty/missing snippets
- Cache bypass
- Fallback PF creation
- LLM failure paths
- Prompt building edge cases
"""
from unittest.mock import patch, MagicMock
import logging

from app.services.generation import generate_answer, _build_chat_prompt
from app.cache import PolicyFingerprint


class TestGenerateAnswerEdgeCases:
    """generate_answer boundary conditions."""

    def test_empty_snippets_returns_idk(self):
        """Empty snippets list returns 'I don't know' without LLM call."""
        result = generate_answer("What is X?", [], use_cache=False)
        assert "I don't know" in result

    def test_none_snippets_list(self):
        """None-ish empty list returns 'I don't know'."""
        result = generate_answer("What is X?", [], use_cache=False)
        assert "I don't know" in result

    def test_use_cache_false_skips_rag_cache(self):
        """use_cache=False should not touch RAG cache, only call LLM directly."""
        mock_llm = MagicMock()
        mock_llm.model = "test-model"
        mock_response = MagicMock()
        mock_response.text = "Test answer"
        mock_llm.generate.return_value = mock_response

        with patch("app.services.providers.get_llm", return_value=mock_llm), \
             patch("app.services.generation.get_rag_cache") as mock_cache:
            result = generate_answer(
                "test query",
                [{"text": "context", "source_path": "a.md", "section_path": "s1"}],
                use_cache=False,
            )
            assert result == "Test answer"
            mock_cache.return_value.get_llm_answer.assert_not_called()

    def test_pf_none_uses_fallback(self, caplog):
        """pf=None with use_cache=True should create fallback PF and warn."""
        mock_llm = MagicMock()
        mock_llm.model = "test-model"
        mock_response = MagicMock()
        mock_response.text = "Cached answer"
        mock_llm.generate.return_value = mock_response

        mock_cache = MagicMock()
        mock_cache.get_llm_answer.return_value = ("Cached answer", True)

        with patch("app.services.providers.get_llm", return_value=mock_llm), \
             patch("app.services.generation.get_rag_cache", return_value=mock_cache), \
             caplog.at_level(logging.WARNING):
            result = generate_answer(
                "test query",
                [{"text": "context", "source_path": "a.md", "section_path": "s1"}],
                use_cache=True,
                pf=None,
            )
            assert result == "Cached answer"
            # Should warn about missing PF
            assert any("fallback PF" in r.message for r in caplog.records)

    def test_llm_failure_returns_extractive_fallback(self):
        """LLM exception should return an extractive fallback with error info."""
        mock_llm = MagicMock()
        mock_llm.model = "test-model"
        mock_llm.generate.side_effect = RuntimeError("API timeout")

        with patch("app.services.providers.get_llm", return_value=mock_llm):
            result = generate_answer(
                "test query",
                [{"text": "Some context text here", "source_path": "doc.md", "section_path": "intro"}],
                use_cache=False,
            )
            assert "LLM Error" in result
            assert "API timeout" in result
            assert "doc.md" in result


class TestBuildChatPromptEdgeCases:
    """_build_chat_prompt edge cases."""

    def test_special_chars_in_snippets(self):
        """Special characters should not break prompt construction."""
        snippets = [
            {
                "text": 'He said "hello" & <goodbye>',
                "source_path": "path/with spaces/doc.md",
                "section_path": "section#1",
            }
        ]
        messages = _build_chat_prompt("query with 'quotes'", snippets)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        # Ensure special chars are preserved, not escaped
        assert '"hello"' in messages[1]["content"]
        assert "<goodbye>" in messages[1]["content"]

    def test_empty_text_fields(self):
        """Snippets with empty/None text fields should not crash."""
        snippets = [
            {"text": None, "source_path": "a.md", "section_path": "s1"},
            {"text": "", "source_path": "b.md", "section_path": "s2"},
        ]
        messages = _build_chat_prompt("test query", snippets)
        assert len(messages) == 2
        # Should contain the query
        assert "test query" in messages[1]["content"]

    def test_many_snippets(self):
        """Large number of snippets should not crash."""
        snippets = [
            {"text": f"Snippet {i} content", "source_path": f"doc{i}.md", "section_path": "s1"}
            for i in range(50)
        ]
        messages = _build_chat_prompt("test", snippets)
        assert "[50]" in messages[1]["content"]
