"""
Embedding Backend Abstraction Layer

Supports multiple backends:
- openai: OpenAI API embeddings (default, cloud)
- local: Sentence Transformers (runs on CPU/GPU locally)
- tei: Text Embeddings Inference server (HuggingFace)

All backends expose the same interface.
"""
from __future__ import annotations
import json
import os
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from app.services.cache import get_cache


@dataclass
class EmbeddingResponse:
    """Standardized response from any embedding backend."""
    embeddings: List[List[float]]
    model: str
    total_tokens: int = 0
    cached_count: int = 0
    computed_count: int = 0


class EmbeddingBackend(ABC):
    """Abstract base class for embedding backends."""
    
    @abstractmethod
    def embed(self, texts: List[str], **kwargs) -> EmbeddingResponse:
        """Embed a list of texts."""
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """Check if the backend is available."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass


class OpenAIEmbeddingBackend(EmbeddingBackend):
    """OpenAI API embedding backend."""
    
    DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.api_key = api_key
        self.model = model
        self._dimension = self.DIMENSIONS.get(model, 1536)
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    def embed(self, texts: List[str], use_cache: bool = True, **kwargs) -> EmbeddingResponse:
        from openai import OpenAI
        
        cache = get_cache() if use_cache else None
        cached_hits = {}
        texts_to_embed = []
        text_indices = []
        
        if cache:
            cached_hits = cache.get_embeddings_batch(texts, self.model)
            for i, text in enumerate(texts):
                if i not in cached_hits:
                    texts_to_embed.append(text)
                    text_indices.append(i)
        else:
            texts_to_embed = texts
            text_indices = list(range(len(texts)))
        
        if not texts_to_embed:
            return EmbeddingResponse(
                embeddings=[cached_hits[i] for i in range(len(texts))],
                model=self.model,
                cached_count=len(texts),
                computed_count=0,
            )
        
        client = OpenAI(api_key=self.api_key)
        resp = client.embeddings.create(model=self.model, input=texts_to_embed)
        new_embeddings = [d.embedding for d in resp.data]
        
        if cache:
            cache.set_embeddings_batch(texts_to_embed, self.model, new_embeddings)
        
        result = [None] * len(texts)
        for i, emb in cached_hits.items():
            result[i] = emb
        for idx, emb in zip(text_indices, new_embeddings):
            result[idx] = emb
        
        return EmbeddingResponse(
            embeddings=result,
            model=self.model,
            total_tokens=getattr(resp, "usage", {}).get("total_tokens", 0) if hasattr(resp, "usage") else 0,
            cached_count=len(cached_hits),
            computed_count=len(new_embeddings),
        )
    
    def health_check(self) -> bool:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            client.embeddings.create(model=self.model, input=["test"])
            return True
        except Exception:
            return False


class LocalEmbeddingBackend(EmbeddingBackend):
    """
    Local embedding using sentence-transformers.
    Runs on CPU or GPU, no external API needed.
    """
    
    DIMENSIONS = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "BAAI/bge-small-en-v1.5": 384,
        "BAAI/bge-base-en-v1.5": 768,
        "BAAI/bge-large-en-v1.5": 1024,
    }
    
    def __init__(self, model: str = "all-MiniLM-L6-v2", device: str = "auto"):
        self.model_name = model
        self.device = device
        self._model = None
        self._dimension = self.DIMENSIONS.get(model, 384)
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            device = self.device if self.device != "auto" else None
            self._model = SentenceTransformer(self.model_name, device=device)
            self._dimension = self._model.get_sentence_embedding_dimension()
        return self._model
    
    def embed(self, texts: List[str], use_cache: bool = True, **kwargs) -> EmbeddingResponse:
        cache = get_cache() if use_cache else None
        cached_hits = {}
        texts_to_embed = []
        text_indices = []
        
        if cache:
            cached_hits = cache.get_embeddings_batch(texts, self.model_name)
            for i, text in enumerate(texts):
                if i not in cached_hits:
                    texts_to_embed.append(text)
                    text_indices.append(i)
        else:
            texts_to_embed = texts
            text_indices = list(range(len(texts)))
        
        if not texts_to_embed:
            return EmbeddingResponse(
                embeddings=[cached_hits[i] for i in range(len(texts))],
                model=self.model_name,
                cached_count=len(texts),
                computed_count=0,
            )
        
        model = self._load_model()
        embeddings = model.encode(texts_to_embed, convert_to_numpy=True)
        new_embeddings = [emb.tolist() for emb in embeddings]
        
        if cache:
            cache.set_embeddings_batch(texts_to_embed, self.model_name, new_embeddings)
        
        result = [None] * len(texts)
        for i, emb in cached_hits.items():
            result[i] = emb
        for idx, emb in zip(text_indices, new_embeddings):
            result[idx] = emb
        
        return EmbeddingResponse(
            embeddings=result,
            model=self.model_name,
            cached_count=len(cached_hits),
            computed_count=len(new_embeddings),
        )
    
    def health_check(self) -> bool:
        try:
            self._load_model()
            return True
        except Exception:
            return False


class TEIBackend(EmbeddingBackend):
    """
    Text Embeddings Inference (HuggingFace) backend.
    High-performance embedding server, can run on GPU.

    Start with: docker run -p 8080:80 ghcr.io/huggingface/text-embeddings-inference:latest \
                --model-id BAAI/bge-small-en-v1.5
    """

    def __init__(self, base_url: str = "http://localhost:8080",
                 model: str = "BAAI/bge-small-en-v1.5"):
        self.base_url = base_url.rstrip("/")
        self.model_name = model
        self._dimension = LocalEmbeddingBackend.DIMENSIONS.get(model, 384)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: List[str], use_cache: bool = True, **kwargs) -> EmbeddingResponse:
        cache = get_cache() if use_cache else None
        cached_hits = {}
        texts_to_embed = []
        text_indices = []

        if cache:
            cached_hits = cache.get_embeddings_batch(texts, self.model_name)
            for i, text in enumerate(texts):
                if i not in cached_hits:
                    texts_to_embed.append(text)
                    text_indices.append(i)
        else:
            texts_to_embed = texts
            text_indices = list(range(len(texts)))

        if not texts_to_embed:
            return EmbeddingResponse(
                embeddings=[cached_hits[i] for i in range(len(texts))],
                model=self.model_name,
                cached_count=len(texts),
                computed_count=0,
            )

        # TEI API call
        url = f"{self.base_url}/embed"
        payload = {"inputs": texts_to_embed}
        req = urllib.request.Request(url, method="POST",
                                      data=json.dumps(payload).encode("utf-8"))
        req.add_header("Content-Type", "application/json")

        with urllib.request.urlopen(req, timeout=30) as resp:
            new_embeddings = json.loads(resp.read().decode("utf-8"))

        if cache:
            cache.set_embeddings_batch(texts_to_embed, self.model_name, new_embeddings)

        result = [None] * len(texts)
        for i, emb in cached_hits.items():
            result[i] = emb
        for idx, emb in zip(text_indices, new_embeddings):
            result[idx] = emb

        return EmbeddingResponse(
            embeddings=result,
            model=self.model_name,
            cached_count=len(cached_hits),
            computed_count=len(new_embeddings),
        )

    def health_check(self) -> bool:
        try:
            url = f"{self.base_url}/health"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False


# ============================================================================
# Backend Factory
# ============================================================================

def get_llm_backend(backend_type: str = None, **kwargs):
    """
    Factory function to get the appropriate LLM backend.

    Args:
        backend_type: "openai", "vllm", or "ollama"
        **kwargs: Backend-specific configuration

    Environment variables:
        LLM_BACKEND: Default backend type
        VLLM_BASE_URL: vLLM server URL (default: http://localhost:8000/v1)
        VLLM_MODEL: Model name for vLLM
        OLLAMA_BASE_URL: Ollama server URL (default: http://localhost:11434)
        OLLAMA_MODEL: Model name for Ollama
    """
    from app.services.llm_backend import OpenAIBackend, VLLMBackend, OllamaBackend
    from app.utils.config import get_settings

    backend_type = backend_type or os.getenv("LLM_BACKEND", "openai")
    settings = get_settings()

    if backend_type == "openai":
        return OpenAIBackend(
            api_key=kwargs.get("api_key") or settings.get("OPENAI_API_KEY", ""),
            model=kwargs.get("model") or settings.get("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        )
    elif backend_type == "vllm":
        return VLLMBackend(
            base_url=kwargs.get("base_url") or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
            model=kwargs.get("model") or os.getenv("VLLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
        )
    elif backend_type == "ollama":
        return OllamaBackend(
            base_url=kwargs.get("base_url") or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=kwargs.get("model") or os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
        )
    else:
        raise ValueError(f"Unknown LLM backend: {backend_type}")


def get_embedding_backend(backend_type: str = None, **kwargs):
    """
    Factory function to get the appropriate embedding backend.

    Args:
        backend_type: "openai", "local", or "tei"
        **kwargs: Backend-specific configuration

    Environment variables:
        EMBEDDING_BACKEND: Default backend type
        LOCAL_EMBEDDING_MODEL: Model for local embeddings
        TEI_BASE_URL: TEI server URL
    """
    from app.utils.config import get_settings

    backend_type = backend_type or os.getenv("EMBEDDING_BACKEND", "openai")
    settings = get_settings()

    if backend_type == "openai":
        return OpenAIEmbeddingBackend(
            api_key=kwargs.get("api_key") or settings.get("OPENAI_API_KEY", ""),
            model=kwargs.get("model") or settings.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        )
    elif backend_type == "local":
        return LocalEmbeddingBackend(
            model=kwargs.get("model") or os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            device=kwargs.get("device", "auto"),
        )
    elif backend_type == "tei":
        return TEIBackend(
            base_url=kwargs.get("base_url") or os.getenv("TEI_BASE_URL", "http://localhost:8080"),
            model=kwargs.get("model") or os.getenv("TEI_MODEL", "BAAI/bge-small-en-v1.5"),
        )
    else:
        raise ValueError(f"Unknown embedding backend: {backend_type}")

