"""Explicit registry for query-planning strategies."""

from __future__ import annotations

from dataclasses import dataclass

from finrag.llm.base import LLMBackend
from finrag.models import Chunk
from finrag.querying.base import QueryStrategy
from finrag.querying.direct import DirectQueryStrategy
from finrag.querying.hyde import HyDEQueryStrategy
from finrag.querying.react import ReactQueryStrategy
from finrag.strategy import Requirement, StrategySpec

DEFAULT_QUERY_STRATEGY = "direct"


@dataclass
class QueryBundle:
    chunks: list[Chunk]
    llm: LLMBackend | None = None


def _direct(_bundle: QueryBundle) -> QueryStrategy:
    return DirectQueryStrategy()


def _hyde(bundle: QueryBundle) -> QueryStrategy:
    if bundle.llm is None:
        raise ValueError("hyde needs a query LLM")
    return HyDEQueryStrategy(bundle.llm)


def _react(bundle: QueryBundle) -> QueryStrategy:
    if bundle.llm is None:
        raise ValueError("react needs a query LLM")
    return ReactQueryStrategy(bundle.chunks, bundle.llm)


QUERY_STRATEGIES: dict[str, StrategySpec[QueryStrategy, QueryBundle]] = {
    "direct": StrategySpec(
        "direct", "Search once with the original question.", _direct
    ),
    "hyde": StrategySpec(
        "hyde",
        "Generate a hypothetical source passage, then retrieve with it.",
        _hyde,
        frozenset({Requirement.QUERY_LLM}),
    ),
    "react": StrategySpec(
        "react",
        "Plan up to four focused searches and select supporting evidence.",
        _react,
        frozenset({Requirement.QUERY_LLM}),
    ),
}


def available_query_strategies() -> list[str]:
    return sorted(QUERY_STRATEGIES)


def query_strategy_catalog() -> list[dict[str, object]]:
    return [QUERY_STRATEGIES[name].public_view() for name in available_query_strategies()]


def query_strategy_requirements(name: str) -> frozenset[Requirement]:
    return _spec(name).requirements


def build_query_strategy(name: str, bundle: QueryBundle) -> QueryStrategy:
    return _spec(name).factory(bundle)


def _spec(name: str) -> StrategySpec[QueryStrategy, QueryBundle]:
    if name not in QUERY_STRATEGIES:
        known = ", ".join(available_query_strategies())
        raise KeyError(f"unknown query strategy '{name}'. Known strategies: {known}")
    return QUERY_STRATEGIES[name]
