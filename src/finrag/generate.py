"""The answer step: retrieve under the role mask, then answer only from what came back.

Order of business, and it is the whole design:

  1. build the retriever for the requested mode from the registry, so "bm25"
     means here exactly what it means to the CLI and the eval harness;
  2. retrieve. The retriever has already applied the role mask before ranking,
     so the hits this file sees are the only chunks that exist as far as the
     answer is concerned;
  3. no hits means no answer. We say so and stop. A model asked to answer with
     an empty context will happily invent a number, and an invented number in a
     finance tool is worse than a blank;
  4. otherwise number the excerpts, ask for an answer that cites them by number,
     and map those numbers back to chunk ids.

With no generation backend configured the answer is extractive: sentences of the
top hit quoted as they stand, flagged `extractive: true`. That is a weaker
answer, not a wrong one, and it keeps the app usable offline.

One mode skips step 4. `react` ran the model several times with the whole
transcript in front of it and finished by writing an answer (typically one call
per hop plus the finish; the hard ceiling is 2 x MAX_HOPS + 1 because every hop
may carry one parse retry). Asking one more call to write the answer again from
a handful of excerpts would cost more and know less, so its own answer is used
as it stands. Its citations were renumbered onto the hits it returned, so the
mapping from `[2]` back to a chunk id is the same one.

Nothing here re-checks the policy to decide what to show. The mask already ran;
re-filtering afterwards would imply the mask cannot be trusted, and then the
invariant would be a suggestion. The policy engine appears once, to explain a
role that may see nothing at all, and once more in the audit counts.

The prompt text, the citation parser and the extractive quote live next door in
`prompting.py`.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from finrag.config import Settings
from finrag.llm.base import LLMBackend
from finrag.llm.vertex import build_llm
from finrag.models import Chunk, Hit, ResourceAttrs
from finrag.policy.engine import RETRIEVE_ACTION, PolicyEngine
from finrag.prompting import (
    SYSTEM_PROMPT,
    build_prompt,
    extractive_answer,
    hit_view,
    is_refusal,
    parse_citations,
)
from finrag.retrieval.base import Retriever, visible_ids
from finrag.retrieval.registry import IndexBundle, available_modes, build_retriever

DEFAULT_MODE = "bm25"
DEFAULT_K = 5

# Modes whose query side has to be embedded, and modes that reorder with a model.
# Both mirror the constants in cli.py; duplicated rather than imported so the
# serving layer does not depend on the command line module.
MODES_NEEDING_AN_EMBEDDER: frozenset[str] = frozenset(
    {"dense", "hybrid", "hybrid_rerank", "react"}
)
MODES_NEEDING_AN_LLM: frozenset[str] = frozenset({"hybrid_rerank", "react"})
MODES_NEEDING_A_GRAPH: frozenset[str] = frozenset({"graphrag"})

# The one mode that writes its own answer. It ran the model once per hop with
# the whole transcript in front of it; a second generation pass would cost
# another call and know less than the agent already did.
REACT_MODE = "react"


class GraphUnavailableError(RuntimeError):
    """graphrag was asked for while the Neo4j server is not answering."""


class Answerer:
    """Holds the loaded corpus and backends, and answers one question at a time.

    Built once per process: loading the chunks and the BM25 index on every
    request was the old repo's mistake. Retrievers are cached per mode for the
    same reason.
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
        self._retrievers: dict[str, Retriever] = {}

    @property
    def chunks(self) -> list[Chunk]:
        return self.bundle.chunks

    def modes(self) -> list[str]:
        """Mode names a caller may ask for, straight from the registry."""
        return available_modes()

    def llm_name(self) -> str | None:
        """Which generation backend is in use, or None when answers are quotes."""
        if self.llm is None:
            return None
        return str(getattr(self.llm, "name", "llm"))

    def retriever(self, mode: str) -> Retriever:
        """The retriever for `mode`, built on first use and kept."""
        if mode in self._retrievers:
            return self._retrievers[mode]

        if mode in MODES_NEEDING_AN_EMBEDDER and self.bundle.embedder is None:
            # Only pay for an embedder when a mode that needs one is asked for;
            # building it may open a Vertex client.
            from finrag.retrieval.embed import build_embedder

            settings = self.settings or Settings.from_env()
            self.bundle.embedder = build_embedder(settings)

        if mode in MODES_NEEDING_AN_LLM and self.bundle.llm is None:
            self.bundle.llm = self._mode_llm(mode)

        if mode in MODES_NEEDING_A_GRAPH:
            # Probe before building: on a dead server the neo4j driver spends
            # about thirty seconds of retries and then raises something
            # unreadable. One socket poke turns that into an instant refusal.
            from finrag.graph.store import bolt_is_open

            if not bolt_is_open():
                raise GraphUnavailableError(
                    "graphrag needs the Neo4j container and it is not answering. "
                    "Start it with `docker start finrag-neo4j` and ask again."
                )

        built = build_retriever(mode, self.bundle)
        self._retrievers[mode] = built
        return built

    def _mode_llm(self, mode: str) -> LLMBackend:
        """The backend a retrieval mode reasons with, which is not the answering one.

        Two backends, one process: the one that writes answers, and this one. A
        process running offline on `FakeLLM` should rerank and run the agent with
        that same fake rather than reaching for credentials it was told not to
        use, so the answering backend is reused unless it is the real Vertex
        client, which is built with the wrong dials for structured output. See
        `llm/vertex.py`.
        """
        from finrag.llm.vertex import build_agent_llm, build_rerank_llm

        if self.llm is not None and getattr(self.llm, "name", "") != "vertex":
            return self.llm
        settings = self.settings or Settings.from_env()
        if mode == REACT_MODE:
            return build_agent_llm(settings)
        return build_rerank_llm(settings)

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
        mode: str = DEFAULT_MODE,
        k: int = DEFAULT_K,
    ) -> dict[str, Any]:
        """Retrieve for this role and answer only from the hits."""
        retriever = self.retriever(mode)
        agent_answer: str | None = None
        if mode == REACT_MODE:
            hits, agent_answer = retriever.search_with_answer(question, role, k=k)
        else:
            hits = retriever.search(question, role, k=k)

        result = self._empty_result(question, role, mode, k)
        result["audit"] = self.audit(role, hits, mode)

        if not hits:
            # No permitted hit is not a hard question, it is no context at all.
            # Never guess: say nothing and let the caller see the audit counts.
            return result

        result["hits"] = [hit_view(hit) for hit in hits]

        if mode == REACT_MODE:
            # The agent already wrote the answer, and its citation numbers were
            # renumbered onto these hits, so the usual parser still applies.
            if agent_answer is None or is_refusal(agent_answer):
                return result
            result["answer"] = agent_answer
            result["citations"] = parse_citations(agent_answer, hits)
            result["reason"] = "ok"
            return result

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

    def audit(self, role: str, hits: list[Hit], mode: str) -> dict[str, Any]:
        """Counts for the policy page: how much of the corpus the mask removed.

        Counts only. The ids of the masked chunks are exactly what a role is not
        allowed to learn, so they never leave this process.
        """
        total = len(self.chunks)
        visible = len(visible_ids(self.chunks, role)) if total else 0
        return {
            "role": role,
            "mode": mode,
            "corpus_chunks": total,
            "visible_to_role": visible,
            "masked_out": total - visible,
            "hits_returned": len(hits),
            "llm": self.llm_name(),
        }

    def _empty_result(self, question: str, role: str, mode: str, k: int) -> dict[str, Any]:
        """The no-answer shape. Every answer starts here and fills fields in."""
        return {
            "question": question,
            "role": role,
            "mode": mode,
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
        graph_path=settings.graph_path,
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
    mode: str = DEFAULT_MODE,
    k: int = DEFAULT_K,
) -> dict[str, Any]:
    """Answer one question as one role, using the process-wide answerer."""
    return get_answerer().answer(question, role, mode=mode, k=k)
