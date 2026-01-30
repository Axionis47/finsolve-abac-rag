#!/usr/bin/env python3
"""
Focused Multi-Hop RAG Testing Script.

Tests the multi-hop functionality with detailed output showing:
1. Query decomposition
2. Sub-query retrieval
3. Context fusion
4. Answer synthesis
5. Comparison with single-hop
"""
import httpx
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8000"
AUTH = ("clark", "chief")  # C-level for full access


def test_query(query: str, multihop: bool, top_k: int = 5) -> dict:
    """Test a single query and return detailed results."""
    client = httpx.Client(timeout=120.0)

    t0 = time.perf_counter()
    resp = client.post(
        f"{BASE_URL}/chat",
        json={"message": query, "top_k": top_k, "multihop": multihop},
        auth=AUTH
    )
    latency = (time.perf_counter() - t0) * 1000

    data = resp.json()

    return {
        "query": query,
        "multihop": multihop,
        "latency_ms": latency,
        "answer": data.get("answer", ""),
        "citations": data.get("citations", []),
        "metrics": data.get("metrics", {}),
    }


def run_multihop_test():
    """Run comprehensive multi-hop tests."""
    print("="*70)
    print("MULTI-HOP RAG FOCUSED TESTING")
    print("="*70)
    print(f"Start: {datetime.now().isoformat()}\n")

    # Complex queries that should benefit from multi-hop
    test_queries = [
        {
            "query": "Compare Q2 and Q3 financial performance including revenue and expenses",
            "description": "Cross-quarter financial comparison",
            "expected_decomposition": ["Q2 revenue", "Q3 revenue", "Q2 expenses", "Q3 expenses"]
        },
        {
            "query": "What is the relationship between marketing campaigns and revenue growth?",
            "description": "Cross-department correlation",
            "expected_decomposition": ["marketing campaigns", "revenue growth"]
        },
        {
            "query": "Compare the vacation policy with sick leave policy",
            "description": "HR policy comparison",
            "expected_decomposition": ["vacation policy", "sick leave policy"]
        },
        {
            "query": "How does the engineering architecture support the company's growth strategy?",
            "description": "Engineering-business alignment",
            "expected_decomposition": ["engineering architecture", "growth strategy"]
        },
        {
            "query": "What are the quarterly trends in operating expenses across Q1 to Q4?",
            "description": "Multi-quarter trend analysis",
            "expected_decomposition": ["Q1 expenses", "Q2 expenses", "Q3 expenses", "Q4 expenses"]
        },
    ]

    results = []

    for i, test in enumerate(test_queries, 1):
        query = test["query"]
        desc = test["description"]

        print(f"\n{'='*70}")
        print(f"TEST {i}: {desc}")
        print(f"{'='*70}")
        print(f"Query: {query}")
        print()

        # Single-hop test
        print("--- SINGLE-HOP ---")
        single = test_query(query, multihop=False)
        print(f"Latency: {single['latency_ms']:.0f}ms")
        print(f"Citations: {len(single['citations'])}")
        for c in single['citations'][:3]:
            print(f"  - {c.get('source_path', '').split('/')[-1]}#{c.get('section_path', '')[:40]}")
        print(f"Answer preview: {single['answer'][:200]}...")

        # Multi-hop test
        print("\n--- MULTI-HOP ---")
        multi = test_query(query, multihop=True)
        print(f"Latency: {multi['latency_ms']:.0f}ms")
        print(f"Citations: {len(multi['citations'])}")
        for c in multi['citations'][:5]:
            print(f"  - {c.get('source_path', '').split('/')[-1]}#{c.get('section_path', '')[:40]}")
        print(f"Answer preview: {multi['answer'][:200]}...")

        # Comparison
        print("\n--- COMPARISON ---")
        latency_diff = multi['latency_ms'] - single['latency_ms']
        context_diff = len(multi['citations']) - len(single['citations'])
        print(f"Latency overhead: {latency_diff:+.0f}ms ({latency_diff/single['latency_ms']*100 if single['latency_ms'] > 0 else 0:+.0f}%)")
        print(f"Context difference: {context_diff:+d} more contexts")

        # Check if multi-hop found different/more sources
        single_sources = set(c.get('source_path', '') for c in single['citations'])
        multi_sources = set(c.get('source_path', '') for c in multi['citations'])
        new_sources = multi_sources - single_sources
        if new_sources:
            print(f"New sources found by multi-hop:")
            for s in new_sources:
                print(f"  + {s.split('/')[-1]}")

        results.append({
            "query": query,
            "description": desc,
            "single_hop": {
                "latency_ms": single['latency_ms'],
                "num_citations": len(single['citations']),
            },
            "multi_hop": {
                "latency_ms": multi['latency_ms'],
                "num_citations": len(multi['citations']),
            },
            "latency_overhead_ms": latency_diff,
            "new_sources_found": len(new_sources),
        })

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    avg_single_latency = sum(r['single_hop']['latency_ms'] for r in results) / len(results)
    avg_multi_latency = sum(r['multi_hop']['latency_ms'] for r in results) / len(results)
    avg_overhead = sum(r['latency_overhead_ms'] for r in results) / len(results)
    total_new_sources = sum(r['new_sources_found'] for r in results)

    print(f"\nAverage Single-Hop Latency: {avg_single_latency:.0f}ms")
    print(f"Average Multi-Hop Latency: {avg_multi_latency:.0f}ms")
    print(f"Average Overhead: {avg_overhead:+.0f}ms")
    print(f"Total New Sources Found: {total_new_sources}")
    print(f"Tests where multi-hop found new sources: {sum(1 for r in results if r['new_sources_found'] > 0)}/{len(results)}")

    print(f"\nEnd: {datetime.now().isoformat()}")

    return results


if __name__ == "__main__":
    run_multihop_test()
