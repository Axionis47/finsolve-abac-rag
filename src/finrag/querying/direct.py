"""The direct strategy: send the original question to the retriever once."""

from __future__ import annotations

from finrag.querying.base import QueryRequest, QueryResult, QueryTrace
from finrag.retrieval.base import Retriever


class DirectQueryStrategy:
    name = "direct"

    def execute(self, request: QueryRequest, retriever: Retriever) -> QueryResult:
        hits = retriever.search(request.question, request.role, k=request.k)
        return QueryResult(
            hits=hits,
            generated_queries=[request.question],
            trace=[QueryTrace(kind="direct", query=request.question, n_hits=len(hits))],
            stats={"searches": 1, "fallbacks": 0},
        )
