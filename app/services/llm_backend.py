"""
LLM Backend Abstraction Layer

Supports multiple backends:
- openai: OpenAI API (default, cloud)
- vllm: Self-hosted vLLM server (OpenAI-compatible API)
- ollama: Local Ollama server

All backends expose the same interface for easy switching.
"""
from __future__ import annotations
import json
import os
import ssl
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from app.services.cache import get_cache, hash_context


@dataclass
class LLMResponse:
    """Standardized response from any LLM backend."""
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: bool = False
    latency_ms: float = 0.0


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""
    
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate a completion from messages."""
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """Check if the backend is available."""
        pass


class OpenAIBackend(LLMBackend):
    """OpenAI API backend."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", 
                 base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        import time
        t0 = time.perf_counter()
        
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.2),
        }
        
        req = urllib.request.Request(url, method="POST", 
                                      data=json.dumps(payload).encode("utf-8"))
        for k, v in headers.items():
            req.add_header(k, v)
        
        context = ssl.create_default_context()
        with urllib.request.urlopen(req, context=context, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        
        usage = body.get("usage", {})
        return LLMResponse(
            text=body["choices"][0]["message"]["content"].strip(),
            model=body.get("model", self.model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
    
    def health_check(self) -> bool:
        try:
            url = f"{self.base_url}/models"
            req = urllib.request.Request(url, method="GET")
            req.add_header("Authorization", f"Bearer {self.api_key}")
            context = ssl.create_default_context()
            with urllib.request.urlopen(req, context=context, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False


class VLLMBackend(LLMBackend):
    """
    vLLM backend - uses OpenAI-compatible API.
    
    vLLM serves an OpenAI-compatible endpoint, so we reuse the same logic
    but point to a different base_url (e.g., http://localhost:8000/v1).
    """
    
    def __init__(self, base_url: str = "http://localhost:8000/v1", 
                 model: str = "meta-llama/Llama-3.1-8B-Instruct"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        # No API key needed for local vLLM
        self._openai = OpenAIBackend(api_key="not-needed", model=model, base_url=base_url)
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        # vLLM uses same API as OpenAI
        return self._openai.generate(messages, **kwargs)
    
    def health_check(self) -> bool:
        try:
            url = f"{self.base_url}/health"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            # Try models endpoint as fallback
            try:
                url = f"{self.base_url}/models"
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    return resp.status == 200
            except Exception:
                return False


class OllamaBackend(LLMBackend):
    """Ollama backend for local CPU/GPU inference."""
    
    def __init__(self, base_url: str = "http://localhost:11434",
                 model: str = "llama3.2:3b"):
        self.base_url = base_url.rstrip("/")
        self.model = model
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        import time
        t0 = time.perf_counter()
        
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "stream": False,
            "options": {"temperature": kwargs.get("temperature", 0.2)},
        }
        
        req = urllib.request.Request(url, method="POST",
                                      data=json.dumps(payload).encode("utf-8"))
        req.add_header("Content-Type", "application/json")
        
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        
        return LLMResponse(
            text=body["message"]["content"].strip(),
            model=body.get("model", self.model),
            prompt_tokens=body.get("prompt_eval_count", 0),
            completion_tokens=body.get("eval_count", 0),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
    
    def health_check(self) -> bool:
        try:
            url = f"{self.base_url}/api/tags"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

