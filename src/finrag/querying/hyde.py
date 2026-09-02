"""Hypothetical Document Embeddings as a query strategy.

HyDE asks the model for a short passage that could answer the question and uses
that passage only as the retrieval query. The hypothetical text never becomes
answer context; reranking and generation continue to use the original question
and real corpus hits.
"""

from __future__ import annotations

from finrag.llm.base import LLMBackend
from finrag.prompting import is_refusal
from finrag.querying.base import QueryRequest, QueryResult, QueryTrace
from finrag.retrieval.base import Retriever

SYSTEM_PROMPT = (
    "You write a hypothetical passage for document retrieval. Return only a short "
    "passage that could appear in the source corpus and answer the question. Do not "
    "add citations, analysis, headings, or caveats."
)


def build_prompt(question: str) -> str:
    return (
        "Write a concise hypothetical source passage that would answer this question. "
        "Use likely domain terminology so semantic retrieval can find related evidence.\n\n"
        f"Question: {question}"
    )


class HyDEQueryStrategy:
    name = "hyde"

    def __init__(self, llm: LLMBackend) -> None:
        self.llm = llm
        self.calls = 0
        self.fallbacks = 0
        self.last_failure: str | None = None

    def execute(self, request: QueryRequest, retriever: Retriever) -> QueryResult:
        self.calls += 1
        generated = ""
        try:
            generated = self.llm.generate(build_prompt(request.question), system=SYSTEM_PROMPT)
        except Exception as error:  # noqa: BLE001 - a failed transform becomes direct search
            self.last_failure = f"call failed: {type(error).__name__}: {error}"

        generated = generated.strip()
        fallback = not generated or is_refusal(generated)
        if fallback:
            self.fallbacks += 1
            retrieval_query = request.question
            kind = "hyde_fallback"
            if not self.last_failure:
                self.last_failure = "the model returned no usable hypothetical passage"
        else:
            retrieval_query = generated
            kind = "hyde"

        hits = retriever.search(retrieval_query, request.role, k=request.k)
        return QueryResult(
            hits=hits,
            generated_queries=[retrieval_query],
            trace=[QueryTrace(kind=kind, query=retrieval_query, n_hits=len(hits))],
            stats={
                "llm_calls": 1,
                "searches": 1,
                "fallbacks": int(fallback),
                "last_failure": self.last_failure if fallback else None,
            },
        )
