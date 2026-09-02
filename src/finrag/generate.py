"""One shared grounded answer stage over any composed search pipeline.

Query planning, retrieval, and reranking finish before this module generates an
answer. The generator always sees the original user question and real, permitted
corpus hits. HyDE text and ReAct planning output can therefore influence search
without becoming answer context. With no generation backend, the top real hit
is quoted as an explicit extractive fallback.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from finrag.config import Settings
from finrag.llm.base import LLMBackend
from finrag.llm.vertex import build_llm
from finrag.models import Chunk, Hit, ResourceAttrs
from finrag.pipeline import (
    SearchPipeline,
    StrategyRuntime,
    StrategySelection,
    strategy_catalog,
)
from finrag.policy.engine import RETRIEVE_ACTION, PolicyEngine
from finrag.prompting import (
    SYSTEM_PROMPT,
    build_prompt,
    extractive_answer,
    hit_view,
    is_refusal,
    parse_citations,
)
from finrag.querying.registry import available_query_strategies
from finrag.reranking.registry import available_rerankers
from finrag.retrieval.base import visible_ids
from finrag.retrieval.registry import IndexBundle, available_retrievers

DEFAULT_K = 5


class Answerer:
    """Holds the loaded corpus and backends, and answers one question at a time.

    Built once per process: artifacts and dependency clients are shared, while
    each requested strategy combination is cached by pipeline id.
    """

    def __init__(
        self,
        bundle: IndexBundle,
        policy: PolicyEngine | None = None,
        llm: LLMBackend | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.bundle = bundle
        self.policy = policy
        self.llm = llm
        self.settings = settings
        self.runtime = StrategyRuntime(
            bundle=bundle,
            settings=settings,
            answer_llm=llm,
        )
        self._pipelines: dict[str, SearchPipeline] = {}

    @property
    def chunks(self) -> list[Chunk]:
        return self.bundle.chunks

    def strategies(self) -> dict[str, dict[str, object]]:
        return strategy_catalog()

    def retrieval_strategies(self) -> list[str]:
        return available_retrievers()

    def query_strategies(self) -> list[str]:
        return available_query_strategies()

    def reranking_strategies(self) -> list[str]:
        return available_rerankers()

    def llm_name(self) -> str | None:
        """Which generation backend is in use, or None when answers are quotes."""
        if self.llm is None:
            return None
        return str(getattr(self.llm, "name", "llm"))

    def pipeline(self, selection: StrategySelection) -> SearchPipeline:
        """Build one strategy combination on first use and keep it."""
        if selection.pipeline_id not in self._pipelines:
            self._pipelines[selection.pipeline_id] = SearchPipeline(selection, self.runtime)
        return self._pipelines[selection.pipeline_id]

    def policy_refusal(self, role: str) -> dict[str, Any] | None:
        """Why the policy gives this role nothing at all, or None if it gives it something.

        This is the whole-corpus question, not the per-question one: an employee
        against a corpus with no general documents may see nothing whatever they
        ask, and saying so is more honest than an empty result list.
        """
        if self.policy is None or not self.chunks:
            return None

        visible = visible_ids(self.chunks, role)
        if visible:
            return None

        rules: list[str] = []
        for chunk in self.chunks:
            resource = ResourceAttrs(
                owner_dept=chunk.owner_dept,
                doc_type=chunk.doc_type,
                sensitivity=chunk.sensitivity,
            )
            decision = self.policy.decide(role, RETRIEVE_ACTION, resource)
            name = decision.rule_name or "default_deny"
            if name not in rules:
                rules.append(name)

        return {
            "reason": "policy_denied",
            "role": role,
            "rules": sorted(rules),
            "detail": (
                f"The policy permits role '{role}' no chunk of this corpus, so there is "
                "nothing to answer from."
            ),
        }

    def answer(
        self,
        question: str,
        role: str,
        retrieval_strategy: str = "bm25",
        query_strategy: str = "direct",
        reranking_strategy: str = "none",
        k: int = DEFAULT_K,
    ) -> dict[str, Any]:
        """Retrieve for this role and answer only from the hits."""
        selection = StrategySelection(
            retrieval_strategy=retrieval_strategy,
            query_strategy=query_strategy,
            reranking_strategy=reranking_strategy,
        )
        query_result = self.pipeline(selection).search(question, role, k=k)
        hits = query_result.hits

        result = self._empty_result(question, role, selection, k)
        result["generated_queries"] = list(query_result.generated_queries)
        result["query_trace"] = [item.to_dict() for item in query_result.trace]
        result["query_stats"] = dict(query_result.stats)
        result["audit"] = self.audit(role, hits, selection)

        if not hits:
            # No permitted hit is not a hard question, it is no context at all.
            # Never guess: say nothing and let the caller see the audit counts.
            return result

        result["hits"] = [hit_view(hit) for hit in hits]

        if self.llm is None:
            result["answer"] = extractive_answer(question, hits[0])
            result["citations"] = [hits[0].chunk_id]
            result["extractive"] = True
            result["reason"] = "ok"
            return result

        reply = self.llm.generate(build_prompt(question, hits), system=SYSTEM_PROMPT).strip()
        if is_refusal(reply):
            # The model read the excerpts and said they do not answer this. Pass
            # the refusal through rather than dressing it up as an answer.
            return result

        result["answer"] = reply
        result["citations"] = parse_citations(reply, hits)
        result["reason"] = "ok"
        return result

    def audit(
        self, role: str, hits: list[Hit], selection: StrategySelection
    ) -> dict[str, Any]:
        """Counts for the policy page: how much of the corpus the mask removed.

        Counts only. The ids of the masked chunks are exactly what a role is not
        allowed to learn, so they never leave this process.
        """
        total = len(self.chunks)
        visible = len(visible_ids(self.chunks, role)) if total else 0
        return {
            "role": role,
            "retrieval_strategy": selection.retrieval_strategy,
            "query_strategy": selection.query_strategy,
            "reranking_strategy": selection.reranking_strategy,
            "pipeline_id": selection.pipeline_id,
            "corpus_chunks": total,
            "visible_to_role": visible,
            "masked_out": total - visible,
            "hits_returned": len(hits),
            "llm": self.llm_name(),
        }

    def _empty_result(
        self, question: str, role: str, selection: StrategySelection, k: int
    ) -> dict[str, Any]:
        """The no-answer shape. Every answer starts here and fills fields in."""
        return {
            "question": question,
            "role": role,
            "retrieval_strategy": selection.retrieval_strategy,
            "query_strategy": selection.query_strategy,
            "reranking_strategy": selection.reranking_strategy,
            "pipeline_id": selection.pipeline_id,
            "k": k,
            "answer": None,
            "reason": "no_answer",
            "citations": [],
            "hits": [],
            "extractive": False,
        }


def load_bundle(settings: Settings) -> IndexBundle:
    """Load the artifacts ingest wrote into a bundle the registry understands."""
    from finrag.ingest.pipeline import read_chunks

    return IndexBundle(
        chunks=read_chunks(settings.chunks_path),
        bm25_dir=settings.bm25_dir,
        dense_path=settings.dense_path,
    )


@lru_cache(maxsize=1)
def get_answerer() -> Answerer:
    """The process-wide answerer, built from the environment on first use."""
    settings = Settings.from_env()
    return Answerer(
        bundle=load_bundle(settings),
        policy=PolicyEngine.from_yaml(settings.policy_path),
        llm=build_llm(settings),
        settings=settings,
    )


def answer(
    question: str,
    role: str,
    retrieval_strategy: str = "bm25",
    query_strategy: str = "direct",
    reranking_strategy: str = "none",
    k: int = DEFAULT_K,
) -> dict[str, Any]:
    """Answer one question as one role, using the process-wide answerer."""
    return get_answerer().answer(
        question,
        role,
        retrieval_strategy=retrieval_strategy,
        query_strategy=query_strategy,
        reranking_strategy=reranking_strategy,
        k=k,
    )
