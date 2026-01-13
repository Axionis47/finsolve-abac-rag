"""
Tests for the LLM/Embedding provider abstraction layer.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock

from app.services.llm_backend import LLMBackend, LLMResponse, OpenAIBackend, VertexAIBackend
from app.services.embed_backend import (
    EmbeddingBackend, EmbeddingResponse, OpenAIEmbeddingBackend, 
    VertexAIEmbeddingBackend, get_llm_backend, get_embedding_backend
)
from app.services.providers import (
    get_llm, get_embeddings, reset_providers, 
    embed_texts, generate, check_all_health
)


class TestLLMBackendInterface:
    """Test the LLM backend abstract interface."""
    
    def test_llm_response_dataclass(self):
        """Test LLMResponse dataclass structure."""
        resp = LLMResponse(
            text="Hello world",
            model="test-model",
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=100.5
        )
        assert resp.text == "Hello world"
        assert resp.model == "test-model"
        assert resp.prompt_tokens == 10
        assert resp.completion_tokens == 5
        assert resp.latency_ms == 100.5
        assert resp.cached is False  # default

    def test_llm_response_defaults(self):
        """Test LLMResponse default values."""
        resp = LLMResponse(text="test", model="m")
        assert resp.prompt_tokens == 0
        assert resp.completion_tokens == 0
        assert resp.cached is False
        assert resp.latency_ms == 0.0


class TestEmbeddingBackendInterface:
    """Test the Embedding backend abstract interface."""
    
    def test_embedding_response_dataclass(self):
        """Test EmbeddingResponse dataclass structure."""
        resp = EmbeddingResponse(
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            model="test-embed",
            total_tokens=20,
            cached_count=1,
            computed_count=1
        )
        assert len(resp.embeddings) == 2
        assert resp.model == "test-embed"
        assert resp.cached_count == 1
        assert resp.computed_count == 1


class TestBackendFactory:
    """Test the backend factory functions."""
    
    def test_get_llm_backend_openai(self, monkeypatch):
        """Test getting OpenAI LLM backend."""
        monkeypatch.setenv("LLM_BACKEND", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        backend = get_llm_backend("openai", api_key="test-key")
        assert isinstance(backend, OpenAIBackend)
        assert backend.model == "gpt-4o-mini"  # default

    def test_get_llm_backend_vertex(self, monkeypatch):
        """Test getting Vertex AI LLM backend."""
        monkeypatch.setenv("VERTEX_PROJECT", "test-project")
        backend = get_llm_backend("vertex", project="test-project")
        assert isinstance(backend, VertexAIBackend)
        assert backend.model == "gemini-2.0-flash-001"  # default

    def test_get_embedding_backend_openai(self, monkeypatch):
        """Test getting OpenAI embedding backend."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        backend = get_embedding_backend("openai", api_key="test-key")
        assert isinstance(backend, OpenAIEmbeddingBackend)
        assert backend.dimension == 1536

    def test_get_embedding_backend_vertex(self, monkeypatch):
        """Test getting Vertex AI embedding backend."""
        monkeypatch.setenv("VERTEX_PROJECT", "test-project")
        backend = get_embedding_backend("vertex", project="test-project")
        assert isinstance(backend, VertexAIEmbeddingBackend)
        assert backend.dimension == 768  # Vertex default

    def test_get_backend_unknown_raises(self):
        """Test that unknown backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown LLM backend"):
            get_llm_backend("unknown_backend")
        
        with pytest.raises(ValueError, match="Unknown embedding backend"):
            get_embedding_backend("unknown_backend")


class TestVertexAIBackend:
    """Test Vertex AI specific functionality."""
    
    def test_vertex_llm_initialization(self):
        """Test VertexAI LLM backend initialization."""
        backend = VertexAIBackend(
            project="my-project",
            location="us-east1",
            model="gemini-1.5-pro"
        )
        assert backend.project == "my-project"
        assert backend.location == "us-east1"
        assert backend.model == "gemini-1.5-pro"

    def test_vertex_embedding_dimensions(self):
        """Test Vertex AI embedding dimension mapping."""
        assert VertexAIEmbeddingBackend.DIMENSIONS["text-embedding-005"] == 768
        assert VertexAIEmbeddingBackend.DIMENSIONS["text-embedding-004"] == 768


class TestProviderSingletons:
    """Test provider singleton behavior."""
    
    def test_reset_providers_clears_cache(self, monkeypatch):
        """Test that reset_providers clears the cache."""
        monkeypatch.setenv("LLM_BACKEND", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        
        reset_providers()  # Clear any existing cache
        
        # This should work without errors
        reset_providers()


class TestHealthChecks:
    """Test health check functionality."""
    
    def test_check_all_health_structure(self, monkeypatch):
        """Test health check returns expected structure."""
        monkeypatch.setenv("LLM_BACKEND", "openai")
        monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
        
        reset_providers()
        health = check_all_health()
        
        assert "llm" in health
        assert "embedding" in health
        assert "backend" in health["llm"]
        assert "backend" in health["embedding"]

