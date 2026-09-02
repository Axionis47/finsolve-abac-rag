"""Shared metadata for the three plug-and-play strategy registries."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Generic, TypeVar


class Requirement(StrEnum):
    """A lazily-created dependency a strategy needs."""

    EMBEDDER = "embedder"
    GRAPH = "graph"
    QUERY_LLM = "query_llm"
    RERANK_LLM = "rerank_llm"


T = TypeVar("T")
BundleT = TypeVar("BundleT")


@dataclass(frozen=True)
class StrategySpec(Generic[T, BundleT]):
    """One explicit registry entry and the information front ends display."""

    name: str
    description: str
    factory: Callable[[BundleT], T]
    requirements: frozenset[Requirement] = frozenset()

    def public_view(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "requirements": sorted(requirement.value for requirement in self.requirements),
        }
