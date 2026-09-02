"""Explicit registry for post-retrieval ranking strategies."""

from __future__ import annotations

from dataclasses import dataclass

from finrag.llm.base import LLMBackend
from finrag.reranking.base import Reranker
from finrag.reranking.identity import IdentityReranker
from finrag.reranking.llm import LLMReranker
from finrag.strategy import Requirement, StrategySpec

DEFAULT_RERANKER = "none"


@dataclass
class RerankingBundle:
    llm: LLMBackend | None = None


def _none(_bundle: RerankingBundle) -> Reranker:
    return IdentityReranker()


def _llm(bundle: RerankingBundle) -> Reranker:
    if bundle.llm is None:
        raise ValueError("llm reranking needs a rerank LLM")
    return LLMReranker(bundle.llm)


RERANKERS: dict[str, StrategySpec[Reranker, RerankingBundle]] = {
    "none": StrategySpec("none", "Keep the query strategy's candidate order.", _none),
    "llm": StrategySpec(
        "llm",
        "Ask Gemini once to reorder the candidate passages.",
        _llm,
        frozenset({Requirement.RERANK_LLM}),
    ),
}


def available_rerankers() -> list[str]:
    return sorted(RERANKERS)


def reranker_catalog() -> list[dict[str, object]]:
    return [RERANKERS[name].public_view() for name in available_rerankers()]


def reranker_requirements(name: str) -> frozenset[Requirement]:
    return _spec(name).requirements


def build_reranker(name: str, bundle: RerankingBundle) -> Reranker:
    return _spec(name).factory(bundle)


def _spec(name: str) -> StrategySpec[Reranker, RerankingBundle]:
    if name not in RERANKERS:
        known = ", ".join(available_rerankers())
        raise KeyError(f"unknown reranking strategy '{name}'. Known strategies: {known}")
    return RERANKERS[name]
