"""
Multi-Hop RAG Service

Handles complex queries that require information from multiple sources by:
1. Detecting if a query is complex (needs multi-hop)
2. Decomposing complex queries into simpler sub-queries
3. Retrieving context for each sub-query in parallel
4. Deduplicating and fusing contexts
5. Synthesizing a final answer from all contexts

This integrates cleanly with existing ABAC policies - each sub-query
goes through the same PDP filtering as single-hop queries.
"""
from __future__ import annotations
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple, Callable, Optional

from app.services.providers import get_llm

logger = logging.getLogger(__name__)


# Keywords that suggest a query needs multi-hop reasoning
COMPLEXITY_INDICATORS = [
    "compare", "contrast", "difference between", "versus", "vs",
    "relationship between", "how does .* affect", "impact of .* on",
    "before and after", "change from .* to", "trend",
    "why did .* when", "explain .* in context of",
    "combine", "integrate", "across", "multiple",
    "q1 and q2", "q2 and q3", "q3 and q4", "quarter",
]


def is_complex_query(query: str) -> bool:
    """
    Detect if a query likely needs multi-hop reasoning.
    
    Uses simple heuristics - can be enhanced with LLM classification if needed.
    """
    query_lower = query.lower()
    
    # Check for complexity indicators
    import re
    for pattern in COMPLEXITY_INDICATORS:
        if re.search(pattern, query_lower):
            return True
    
    # Check for multiple question marks or "and" connecting concepts
    if query.count("?") > 1:
        return True
    if " and " in query_lower and len(query.split()) > 10:
        return True
    
    return False


def decompose_query(query: str, max_subqueries: int = 3) -> List[str]:
    """
    Use LLM to decompose a complex query into simpler sub-queries.

    Returns a list of sub-queries that together can answer the original query.
    Sub-queries include synonyms/related terms for better retrieval.
    """
    llm = get_llm()

    system_prompt = """You are a query decomposition assistant for an HR/corporate document search system.
Break down complex questions into simpler sub-questions optimized for keyword search.

IMPORTANT RULES:
1. Return 2-4 simple, focused sub-questions
2. Each sub-question should be answerable independently
3. Include SYNONYMS and RELATED TERMS in each sub-question for better search matching
4. Return ONLY a JSON array of strings, no explanation
5. Keep sub-questions concise but include key search terms

Common synonym mappings to include:
- vacation → leave, PTO, time off, annual leave, privilege leave
- benefits → insurance, EPF, gratuity, wellness, compensation
- salary → pay, wages, compensation, payroll
- rules → policy, guidelines, procedures
- sick days → sick leave, medical leave

Example:
Query: "Compare employee vacation policy with benefits"
Output: ["What are the employee leave policies including vacation PTO annual leave sick leave?", "What employee benefits are offered including health insurance EPF gratuity wellness?"]

Example:
Query: "Compare Q3 revenue to Q2 and explain the marketing impact"
Output: ["What was the revenue income sales in Q2?", "What was the revenue income sales in Q3?", "What marketing spend budget campaigns occurred in Q2 and Q3?"]"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Decompose this query into search-optimized sub-questions:\n\n{query}"}
    ]

    try:
        response = llm.generate(messages, temperature=0.1)
        text = response.text.strip()

        # Parse JSON array from response
        # Handle cases where LLM wraps in markdown code blocks
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        sub_queries = json.loads(text)

        if isinstance(sub_queries, list) and all(isinstance(q, str) for q in sub_queries):
            # Limit number of sub-queries
            return sub_queries[:max_subqueries]
    except Exception as e:
        logger.warning(f"Query decomposition failed: {e}, falling back to original query")

    # Fallback: return original query
    return [query]


def deduplicate_contexts(contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate context snippets based on source+section path.
    Keeps the first occurrence of each unique snippet.
    """
    seen = set()
    unique = []

    for ctx in contexts:
        key = f"{ctx.get('source_path', '')}#{ctx.get('section_path', '')}"
        if key not in seen:
            seen.add(key)
            unique.append(ctx)

    return unique


def deduplicate_citations(citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate citations."""
    seen = set()
    unique = []

    for cit in citations:
        key = f"{cit.get('source_path', '')}#{cit.get('section_path', '')}"
        if key not in seen:
            seen.add(key)
            unique.append(cit)

    return unique


def multihop_retrieve(
    sub_queries: List[str],
    retrieve_fn: Callable[[str], Tuple[List[Dict], List[Dict]]],
    max_contexts_per_query: int = 3,
    max_total_contexts: int = 10,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Retrieve contexts for multiple sub-queries and fuse results.

    Args:
        sub_queries: List of decomposed sub-queries
        retrieve_fn: Function that takes a query and returns (contexts, citations)
        max_contexts_per_query: Max contexts to keep per sub-query
        max_total_contexts: Max total contexts to return

    Returns:
        Tuple of (all_contexts, all_citations, metrics)
    """
    all_contexts = []
    all_citations = []
    metrics = {
        "sub_queries": sub_queries,
        "retrieval_times_ms": [],
    }

    # Retrieve for each sub-query (can be parallelized)
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=min(len(sub_queries), 3)) as executor:
        future_to_query = {
            executor.submit(retrieve_fn, sq): sq
            for sq in sub_queries
        }

        for future in as_completed(future_to_query):
            query = future_to_query[future]
            try:
                t_start = time.perf_counter()
                contexts, citations = future.result()
                t_end = time.perf_counter()

                metrics["retrieval_times_ms"].append({
                    "query": query,
                    "time_ms": (t_end - t_start) * 1000
                })

                # Limit contexts per sub-query to avoid one dominating
                all_contexts.extend(contexts[:max_contexts_per_query])
                all_citations.extend(citations[:max_contexts_per_query])

            except Exception as e:
                logger.error(f"Retrieval failed for sub-query '{query}': {e}")

    metrics["total_retrieval_ms"] = (time.perf_counter() - t0) * 1000

    # Deduplicate
    unique_contexts = deduplicate_contexts(all_contexts)[:max_total_contexts]
    unique_citations = deduplicate_citations(all_citations)[:max_total_contexts]

    metrics["contexts_before_dedup"] = len(all_contexts)
    metrics["contexts_after_dedup"] = len(unique_contexts)

    return unique_contexts, unique_citations, metrics


def multihop_query(
    retrieve_fn: Callable[[str], Tuple[List[Dict], List[Dict]]],
    query: str,
    top_k: int = 5,
    auto_detect: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Main entry point for multi-hop RAG.

    Args:
        retrieve_fn: Function that takes a query string and returns (contexts, citations)
        query: The user's original query
        top_k: Number of contexts to return per sub-query
        auto_detect: If True, only use multi-hop for complex queries

    Returns:
        Tuple of (contexts, citations, multihop_metrics)
    """
    metrics = {
        "original_query": query,
        "is_complex": False,
        "used_multihop": False,
        "decomposition_ms": 0,
    }

    # Check if we should use multi-hop
    if auto_detect and not is_complex_query(query):
        # Simple query - just do single retrieval
        contexts, citations = retrieve_fn(query)
        return contexts[:top_k], citations[:top_k], metrics

    metrics["is_complex"] = True
    metrics["used_multihop"] = True

    # Decompose query
    t0 = time.perf_counter()
    sub_queries = decompose_query(query, max_subqueries=3)
    metrics["decomposition_ms"] = (time.perf_counter() - t0) * 1000
    metrics["sub_queries"] = sub_queries

    # If decomposition returned just the original query, treat as simple
    if len(sub_queries) == 1 and sub_queries[0] == query:
        contexts, citations = retrieve_fn(query)
        metrics["used_multihop"] = False
        return contexts[:top_k], citations[:top_k], metrics

    # Multi-hop retrieval
    contexts, citations, retrieval_metrics = multihop_retrieve(
        sub_queries,
        retrieve_fn,
        max_contexts_per_query=top_k,
        max_total_contexts=top_k * 2,  # Allow more contexts for complex queries
    )

    metrics.update(retrieval_metrics)

    return contexts, citations, metrics

