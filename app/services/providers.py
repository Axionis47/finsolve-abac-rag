"""
Unified Provider Interface for LLM and Embedding Services.

This module provides a single entry point for all AI services with:
- Plug-and-play backend switching via environment variables
- Singleton instances for efficiency
- Simple, consistent API regardless of backend

Usage:
    from app.services.providers import get_llm, get_embeddings, embed_texts, generate

    # Get backend instances (singletons)
    llm = get_llm()
    embedder = get_embeddings()

    # Or use convenience functions directly
    embeddings = embed_texts(["text1", "text2"])
    response = generate([{"role": "user", "content": "Hello"}])
"""
from __future__ import annotations
import os
from typing import List, Dict, Any, Optional
from functools import lru_cache

from app.services.embed_backend import (
    get_llm_backend,
    get_embedding_backend,
    EmbeddingBackend,
    EmbeddingResponse,
)
from app.services.llm_backend import LLMBackend, LLMResponse


# ============================================================================
# Singleton Accessors
# ============================================================================

@lru_cache(maxsize=1)
def get_llm(backend_type: Optional[str] = None) -> LLMBackend:
    """
    Get the singleton LLM backend instance.
    
    Backend is selected from LLM_BACKEND env var or parameter.
    Options: openai, vllm, ollama, vertex
    """
    return get_llm_backend(backend_type)


@lru_cache(maxsize=1)
def get_embeddings(backend_type: Optional[str] = None) -> EmbeddingBackend:
    """
    Get the singleton embedding backend instance.
    
    Backend is selected from EMBEDDING_BACKEND env var or parameter.
    Options: openai, local, tei, vertex
    """
    return get_embedding_backend(backend_type)


def reset_providers():
    """Reset cached providers. Useful for testing or config changes."""
    get_llm.cache_clear()
    get_embeddings.cache_clear()


# ============================================================================
# Convenience Functions
# ============================================================================

def embed_texts(texts: List[str], use_cache: bool = True) -> List[List[float]]:
    """
    Embed texts using the configured backend.
    
    Returns list of embedding vectors.
    """
    embedder = get_embeddings()
    response = embedder.embed(texts, use_cache=use_cache)
    return response.embeddings


def generate(messages: List[Dict[str, str]], **kwargs) -> str:
    """
    Generate a completion from messages using the configured backend.
    
    Returns the generated text.
    """
    llm = get_llm()
    response = llm.generate(messages, **kwargs)
    return response.text


def generate_with_metadata(messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
    """
    Generate a completion and return full response with metadata.
    
    Includes: text, model, prompt_tokens, completion_tokens, latency_ms
    """
    llm = get_llm()
    return llm.generate(messages, **kwargs)


def embed_with_metadata(texts: List[str], use_cache: bool = True) -> EmbeddingResponse:
    """
    Embed texts and return full response with metadata.
    
    Includes: embeddings, model, cached_count, computed_count
    """
    embedder = get_embeddings()
    return embedder.embed(texts, use_cache=use_cache)


# ============================================================================
# Health Checks
# ============================================================================

def check_llm_health() -> Dict[str, Any]:
    """Check LLM backend health."""
    backend_type = os.getenv("LLM_BACKEND", "vertex")
    try:
        llm = get_llm()
        healthy = llm.health_check()
        return {
            "backend": backend_type,
            "model": getattr(llm, "model", "unknown"),
            "healthy": healthy,
        }
    except Exception as e:
        return {
            "backend": backend_type,
            "healthy": False,
            "error": str(e),
        }


def check_embedding_health() -> Dict[str, Any]:
    """Check embedding backend health."""
    backend_type = os.getenv("EMBEDDING_BACKEND", "vertex")
    try:
        embedder = get_embeddings()
        healthy = embedder.health_check()
        return {
            "backend": backend_type,
            "model": getattr(embedder, "model_name", getattr(embedder, "model", "unknown")),
            "dimension": embedder.dimension,
            "healthy": healthy,
        }
    except Exception as e:
        return {
            "backend": backend_type,
            "healthy": False,
            "error": str(e),
        }


def check_all_health() -> Dict[str, Any]:
    """Check health of all AI services."""
    return {
        "llm": check_llm_health(),
        "embedding": check_embedding_health(),
    }

