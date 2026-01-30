"""
Input validation schemas with sanitization.

Provides Pydantic models for validating and sanitizing user input
to prevent XSS, injection attacks, and enforce size limits.
"""
from __future__ import annotations
import re
from typing import Optional, List

from pydantic import BaseModel, Field, field_validator


def sanitize_text(text: str) -> str:
    """
    Sanitize text input by removing potentially dangerous content.

    - Strips leading/trailing whitespace
    - Removes HTML tags (basic XSS prevention)
    - Normalizes whitespace
    """
    if not text:
        return text

    # Remove HTML tags
    text = re.sub(r'<[^>]*>', '', text)

    # Normalize whitespace (but preserve single newlines)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The user's question or message",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of context snippets to retrieve",
    )
    persist_dir: Optional[str] = Field(
        default=None,
        max_length=500,
        description="ChromaDB persistence directory",
    )
    base_dir: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Base directory for documents",
    )
    rerank: bool = Field(
        default=False,
        description="Enable reranking of results",
    )
    reranker: Optional[str] = Field(
        default=None,
        pattern=r'^[a-zA-Z0-9_-]+$',
        description="Reranker type to use",
    )
    multihop: bool = Field(
        default=False,
        description="Enable multi-hop retrieval",
    )
    multihop_auto: bool = Field(
        default=True,
        description="Auto-detect when to use multi-hop",
    )

    @field_validator('message')
    @classmethod
    def sanitize_message(cls, v: str) -> str:
        """Sanitize the message field."""
        return sanitize_text(v)

    @field_validator('persist_dir', 'base_dir')
    @classmethod
    def validate_path(cls, v: Optional[str]) -> Optional[str]:
        """Validate path fields to prevent path traversal."""
        if v is None:
            return v
        # Block path traversal attempts
        if '..' in v or v.startswith('/etc') or v.startswith('/var'):
            raise ValueError("Invalid path")
        return v


class SearchRequest(BaseModel):
    """Request model for search endpoint."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Search query",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of results to return",
    )
    persist_dir: Optional[str] = Field(
        default=None,
        max_length=500,
        description="ChromaDB persistence directory",
    )
    base_dir: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Base directory for documents",
    )
    rerank: bool = Field(
        default=False,
        description="Enable reranking",
    )
    reranker: Optional[str] = Field(
        default=None,
        pattern=r'^[a-zA-Z0-9_-]+$',
        description="Reranker type",
    )

    @field_validator('query')
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        """Sanitize the query field."""
        return sanitize_text(v)

    @field_validator('persist_dir', 'base_dir')
    @classmethod
    def validate_path(cls, v: Optional[str]) -> Optional[str]:
        """Validate path fields."""
        if v is None:
            return v
        if '..' in v:
            raise ValueError("Invalid path")
        return v


class HrRowsRequest(BaseModel):
    """Request model for HR rows endpoint."""

    limit: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Maximum number of rows to return",
    )
    offset: int = Field(
        default=0,
        ge=0,
        description="Number of rows to skip",
    )


class AdminReindexRequest(BaseModel):
    """Request model for admin reindex endpoints."""

    base_dir: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Base directory for documents",
    )
    persist_dir: Optional[str] = Field(
        default=None,
        max_length=500,
        description="ChromaDB persistence directory",
    )

    @field_validator('base_dir', 'persist_dir')
    @classmethod
    def validate_path(cls, v: Optional[str]) -> Optional[str]:
        """Validate path fields."""
        if v is None:
            return v
        if '..' in v:
            raise ValueError("Invalid path")
        return v
