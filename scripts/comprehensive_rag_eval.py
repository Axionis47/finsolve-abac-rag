#!/usr/bin/env python3
"""
Comprehensive RAG Evaluation Script.

Tests:
1. Retrieval quality against ground truth
2. Access control enforcement across roles
3. Single-hop vs multi-hop comparison
4. Answer quality (faithfulness, relevance)
5. Policy enforcement verification
"""
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx


@dataclass
class TestResult:
    """Result of a single test case."""
    question_id: str
    question: str
    role: str
    expected_access: bool
    actual_access: bool
    answer: str = ""
    contexts: List[Dict] = field(default_factory=list)
    citations: List[Dict] = field(default_factory=list)
    latency_ms: float = 0.0
    error: Optional[str] = None

    # Metrics
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    faithfulness: Optional[float] = None
    answer_relevance: Optional[float] = None

    @property
    def access_correct(self) -> bool:
        return self.expected_access == self.actual_access


@dataclass
class MultihopComparison:
    """Comparison between single-hop and multi-hop."""
    question_id: str
    question: str
    role: str
    single_hop_answer: str
    multihop_answer: str
    single_hop_latency_ms: float
    multihop_latency_ms: float
    single_hop_contexts: int
    multihop_contexts: int
    latency_overhead_ms: float
    quality_improvement: Optional[float] = None


class RAGEvaluator:
    """Comprehensive RAG system evaluator."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.Client(timeout=120.0)

        # Load ground truth
        gt_path = Path(__file__).parent.parent / "evaluation" / "datasets" / "ground_truth.json"
        with open(gt_path) as f:
            self.ground_truth = json.load(f)

        # Test users with credentials
        self.users = {
            "employee": ("tony", "password123", "engineering"),  # Using tony as general employee
            "engineering": ("tony", "password123", "engineering"),
            "marketing": ("bruce", "securepass", "marketing"),
            "finance": ("sam", "financepass", "finance"),
            "hr": ("natasha", "hrpass123", "hr"),
            "c_level": ("clark", "chief", "c_level"),
        }

    def _make_request(self, endpoint: str, data: Dict, username: str, password: str) -> Tuple[int, Dict]:
        """Make authenticated request to API."""
        try:
            resp = self.client.post(
                f"{self.base_url}{endpoint}",
                json=data,
                auth=(username, password)
            )
            return resp.status_code, resp.json() if resp.status_code == 200 else {"error": resp.text}
        except Exception as e:
            return 500, {"error": str(e)}

    def _normalize_source(self, path: str) -> set:
        """Normalize source path to set of key terms for comparison."""
        path = path.lower()
        # Remove common prefixes
        for prefix in ['resources/data/', 'resources/', 'data/', 'general/', 'finance/',
                       'marketing/', 'engineering/', 'hr/']:
            path = path.replace(prefix, '')
        # Normalize separators and formatting
        path = path.replace(' > ', ' ').replace('#', ' ').replace('-', ' ')
        path = path.replace('.md', '').replace('_', ' ')
        # Split and return as set of terms
        return set(path.split())

    def _context_matches(self, ctx: Dict, gt: str) -> bool:
        """Check if a retrieved context matches a ground truth reference."""
        source = ctx.get("source_path", "")
        section = ctx.get("section_path", "")

        ctx_terms = self._normalize_source(f"{source} {section}")
        gt_terms = self._normalize_source(gt)

        # Check for significant overlap
        overlap = len(gt_terms & ctx_terms)
        if not gt_terms:
            return False
        return overlap >= max(1, len(gt_terms) / 2)

    def _compute_context_precision(self, retrieved_contexts: List[Dict],
                                    ground_truth_contexts: List[str]) -> float:
        """Compute context precision - fraction of retrieved contexts that are relevant."""
        if not retrieved_contexts:
            return 0.0
        if not ground_truth_contexts:
            return 1.0

        relevant_count = 0
        for ctx in retrieved_contexts:
            for gt in ground_truth_contexts:
                if self._context_matches(ctx, gt):
                    relevant_count += 1
                    break

        return relevant_count / len(retrieved_contexts)

    def _compute_context_recall(self, retrieved_contexts: List[Dict],
                                 ground_truth_contexts: List[str]) -> float:
        """Compute context recall - fraction of ground truth contexts retrieved."""
        if not ground_truth_contexts:
            return 1.0
        if not retrieved_contexts:
            return 0.0

        found_count = 0
        for gt in ground_truth_contexts:
            for ctx in retrieved_contexts:
                if self._context_matches(ctx, gt):
                    found_count += 1
                    break

        return found_count / len(ground_truth_contexts)

    def _compute_faithfulness(self, answer: str, contexts: List[Dict]) -> float:
        """Compute faithfulness - is answer grounded in context."""
        if not answer or not contexts:
            return 0.0

        # Combine context texts
        context_text = " ".join(ctx.get("text", "") for ctx in contexts).lower()
        context_words = set(context_text.split())

        # Check answer word overlap
        answer_words = set(answer.lower().split())
        # Remove common stopwords
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                     "being", "have", "has", "had", "do", "does", "did", "will",
                     "would", "could", "should", "may", "might", "must", "shall",
                     "to", "of", "in", "for", "on", "with", "at", "by", "from",
                     "as", "into", "through", "during", "before", "after", "and",
                     "or", "but", "if", "then", "than", "so", "because", "while",
                     "this", "that", "these", "those", "it", "its", "i", "you",
                     "he", "she", "they", "we", "my", "your", "his", "her", "their"}
        answer_words -= stopwords

        if not answer_words:
            return 1.0

        overlap = len(answer_words & context_words)
        return min(overlap / len(answer_words), 1.0)

    def _compute_answer_relevance(self, question: str, answer: str) -> float:
        """Compute answer relevance - does answer address the question."""
        if not answer:
            return 0.0

        # Extract key terms from question
        question_words = set(question.lower().split())
        stopwords = {"what", "how", "why", "when", "where", "who", "which",
                     "is", "are", "the", "a", "an", "does", "do", "can", "could",
                     "would", "should", "of", "in", "for", "to", "and", "or"}
        question_terms = question_words - stopwords

        answer_words = set(answer.lower().split())

        if not question_terms:
            return 1.0

        overlap = len(question_terms & answer_words)
        return min(overlap / len(question_terms), 1.0)

    def test_single_query(self, question: Dict, role: str,
                          multihop: bool = False) -> TestResult:
        """Test a single query for a specific role."""
        q_id = question["id"]
        q_text = question["question"]
        applicable_roles = question.get("applicable_roles", [])
        gt_contexts = question.get("ground_truth_contexts", [])

        # Determine expected access
        expected_access = role in applicable_roles or role == "c_level"

        # Get credentials
        username, password, _ = self.users.get(role, self.users["engineering"])

        # Make request
        t0 = time.perf_counter()
        status, response = self._make_request(
            "/chat",
            {"message": q_text, "top_k": 5, "multihop": multihop},
            username, password
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        result = TestResult(
            question_id=q_id,
            question=q_text,
            role=role,
            expected_access=expected_access,
            actual_access=status == 200 and "answer" in response,
            latency_ms=latency_ms,
        )

        if status == 200 and "answer" in response:
            result.answer = response.get("answer", "")
            result.citations = response.get("citations", [])

            # Get contexts from citations for metric computation
            contexts = [{"source_path": c.get("source_path", ""),
                        "section_path": c.get("section_path", ""),
                        "text": ""} for c in result.citations]
            result.contexts = contexts

            # Compute metrics
            result.context_precision = self._compute_context_precision(contexts, gt_contexts)
            result.context_recall = self._compute_context_recall(contexts, gt_contexts)
            result.faithfulness = self._compute_faithfulness(result.answer, contexts)
            result.answer_relevance = self._compute_answer_relevance(q_text, result.answer)
        elif status != 200:
            result.error = response.get("error", f"Status {status}")

        return result

    def test_access_control(self) -> List[TestResult]:
        """Test access control enforcement across all roles."""
        print("\n" + "="*60)
        print("ACCESS CONTROL TESTING")
        print("="*60)

        results = []
        questions = self.ground_truth["questions"]
        roles = ["engineering", "marketing", "finance", "hr", "c_level"]

        for role in roles:
            print(f"\nTesting role: {role}")
            role_results = []

            for q in questions:
                result = self.test_single_query(q, role)
                role_results.append(result)

                status = "✓" if result.access_correct else "✗"
                print(f"  {status} {q['id']}: expected={result.expected_access}, actual={result.actual_access}")

            results.extend(role_results)

            # Summary for role
            correct = sum(1 for r in role_results if r.access_correct)
            print(f"  Role {role}: {correct}/{len(role_results)} correct")

        return results

    def test_retrieval_quality(self) -> List[TestResult]:
        """Test retrieval quality against ground truth."""
        print("\n" + "="*60)
        print("RETRIEVAL QUALITY TESTING")
        print("="*60)

        results = []
        questions = self.ground_truth["questions"]

        # Use c_level for maximum access
        for q in questions:
            result = self.test_single_query(q, "c_level")
            results.append(result)

            print(f"\n{q['id']}: {q['question'][:50]}...")
            if result.context_precision is not None:
                print(f"  Context Precision: {result.context_precision:.2f}")
                print(f"  Context Recall: {result.context_recall:.2f}")
                print(f"  Faithfulness: {result.faithfulness:.2f}")
                print(f"  Answer Relevance: {result.answer_relevance:.2f}")
            else:
                print(f"  Error: {result.error}")

        return results

    def test_multihop_comparison(self) -> List[MultihopComparison]:
        """Compare single-hop vs multi-hop for complex queries."""
        print("\n" + "="*60)
        print("MULTI-HOP VS SINGLE-HOP COMPARISON")
        print("="*60)

        comparisons = []

        # Filter complex questions
        complex_questions = [q for q in self.ground_truth["questions"]
                           if q.get("is_complex") or q.get("requires_multihop")]

        print(f"\nTesting {len(complex_questions)} complex queries...")

        for q in complex_questions:
            print(f"\n{q['id']}: {q['question'][:50]}...")

            # Single-hop
            single_result = self.test_single_query(q, "c_level", multihop=False)

            # Multi-hop
            multihop_result = self.test_single_query(q, "c_level", multihop=True)

            comparison = MultihopComparison(
                question_id=q["id"],
                question=q["question"],
                role="c_level",
                single_hop_answer=single_result.answer[:200] + "..." if len(single_result.answer) > 200 else single_result.answer,
                multihop_answer=multihop_result.answer[:200] + "..." if len(multihop_result.answer) > 200 else multihop_result.answer,
                single_hop_latency_ms=single_result.latency_ms,
                multihop_latency_ms=multihop_result.latency_ms,
                single_hop_contexts=len(single_result.contexts),
                multihop_contexts=len(multihop_result.contexts),
                latency_overhead_ms=multihop_result.latency_ms - single_result.latency_ms,
            )

            # Compute quality improvement
            if single_result.faithfulness is not None and multihop_result.faithfulness is not None:
                single_score = (single_result.context_precision or 0 +
                              single_result.faithfulness or 0 +
                              single_result.answer_relevance or 0) / 3
                multihop_score = (multihop_result.context_precision or 0 +
                                multihop_result.faithfulness or 0 +
                                multihop_result.answer_relevance or 0) / 3
                comparison.quality_improvement = multihop_score - single_score

            comparisons.append(comparison)

            print(f"  Single-hop: {single_result.latency_ms:.0f}ms, {len(single_result.contexts)} contexts")
            print(f"  Multi-hop: {multihop_result.latency_ms:.0f}ms, {len(multihop_result.contexts)} contexts")
            print(f"  Latency overhead: {comparison.latency_overhead_ms:.0f}ms")
            if comparison.quality_improvement is not None:
                print(f"  Quality improvement: {comparison.quality_improvement:+.2f}")

        return comparisons

    def test_by_category(self) -> Dict[str, List[TestResult]]:
        """Test questions grouped by category."""
        print("\n" + "="*60)
        print("CATEGORY-BASED TESTING")
        print("="*60)

        results_by_category = {}
        questions = self.ground_truth["questions"]

        # Group by category
        categories = {}
        for q in questions:
            cat = q.get("category", "unknown")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(q)

        for cat, cat_questions in categories.items():
            print(f"\nCategory: {cat} ({len(cat_questions)} questions)")
            results = []

            for q in cat_questions:
                result = self.test_single_query(q, "c_level")
                results.append(result)

            results_by_category[cat] = results

            # Compute category averages
            valid_results = [r for r in results if r.context_precision is not None]
            if valid_results:
                avg_precision = sum(r.context_precision for r in valid_results) / len(valid_results)
                avg_recall = sum(r.context_recall for r in valid_results) / len(valid_results)
                avg_faithfulness = sum(r.faithfulness for r in valid_results) / len(valid_results)
                avg_relevance = sum(r.answer_relevance for r in valid_results) / len(valid_results)

                print(f"  Avg Precision: {avg_precision:.2f}")
                print(f"  Avg Recall: {avg_recall:.2f}")
                print(f"  Avg Faithfulness: {avg_faithfulness:.2f}")
                print(f"  Avg Relevance: {avg_relevance:.2f}")

        return results_by_category

    def generate_report(self,
                        access_results: List[TestResult],
                        quality_results: List[TestResult],
                        multihop_comparisons: List[MultihopComparison],
                        category_results: Dict[str, List[TestResult]]) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""

        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {},
            "access_control": {},
            "retrieval_quality": {},
            "multihop_analysis": {},
            "category_breakdown": {},
        }

        # Access control summary
        total_access = len(access_results)
        correct_access = sum(1 for r in access_results if r.access_correct)
        report["access_control"] = {
            "total_tests": total_access,
            "correct": correct_access,
            "accuracy": correct_access / total_access if total_access > 0 else 0,
            "by_role": {}
        }

        for role in ["engineering", "marketing", "finance", "hr", "c_level"]:
            role_results = [r for r in access_results if r.role == role]
            role_correct = sum(1 for r in role_results if r.access_correct)
            report["access_control"]["by_role"][role] = {
                "total": len(role_results),
                "correct": role_correct,
                "accuracy": role_correct / len(role_results) if role_results else 0
            }

        # Retrieval quality summary
        valid_quality = [r for r in quality_results if r.context_precision is not None]
        if valid_quality:
            report["retrieval_quality"] = {
                "total_tests": len(quality_results),
                "valid_results": len(valid_quality),
                "metrics": {
                    "context_precision": {
                        "mean": sum(r.context_precision for r in valid_quality) / len(valid_quality),
                        "min": min(r.context_precision for r in valid_quality),
                        "max": max(r.context_precision for r in valid_quality),
                    },
                    "context_recall": {
                        "mean": sum(r.context_recall for r in valid_quality) / len(valid_quality),
                        "min": min(r.context_recall for r in valid_quality),
                        "max": max(r.context_recall for r in valid_quality),
                    },
                    "faithfulness": {
                        "mean": sum(r.faithfulness for r in valid_quality) / len(valid_quality),
                        "min": min(r.faithfulness for r in valid_quality),
                        "max": max(r.faithfulness for r in valid_quality),
                    },
                    "answer_relevance": {
                        "mean": sum(r.answer_relevance for r in valid_quality) / len(valid_quality),
                        "min": min(r.answer_relevance for r in valid_quality),
                        "max": max(r.answer_relevance for r in valid_quality),
                    },
                },
                "avg_latency_ms": sum(r.latency_ms for r in valid_quality) / len(valid_quality),
            }

        # Multi-hop analysis
        if multihop_comparisons:
            report["multihop_analysis"] = {
                "total_comparisons": len(multihop_comparisons),
                "avg_latency_overhead_ms": sum(c.latency_overhead_ms for c in multihop_comparisons) / len(multihop_comparisons),
                "comparisons_with_improvement": sum(1 for c in multihop_comparisons
                                                     if c.quality_improvement and c.quality_improvement > 0),
                "avg_quality_improvement": sum(c.quality_improvement or 0 for c in multihop_comparisons) / len(multihop_comparisons),
            }

        # Category breakdown
        for cat, results in category_results.items():
            valid = [r for r in results if r.context_precision is not None]
            if valid:
                report["category_breakdown"][cat] = {
                    "total": len(results),
                    "valid": len(valid),
                    "avg_precision": sum(r.context_precision for r in valid) / len(valid),
                    "avg_recall": sum(r.context_recall for r in valid) / len(valid),
                    "avg_faithfulness": sum(r.faithfulness for r in valid) / len(valid),
                    "avg_relevance": sum(r.answer_relevance for r in valid) / len(valid),
                }

        # Overall summary
        report["summary"] = {
            "access_control_accuracy": report["access_control"]["accuracy"],
            "avg_context_precision": report["retrieval_quality"].get("metrics", {}).get("context_precision", {}).get("mean", 0),
            "avg_faithfulness": report["retrieval_quality"].get("metrics", {}).get("faithfulness", {}).get("mean", 0),
            "avg_answer_relevance": report["retrieval_quality"].get("metrics", {}).get("answer_relevance", {}).get("mean", 0),
            "multihop_beneficial": report["multihop_analysis"].get("avg_quality_improvement", 0) > 0,
        }

        return report


def main():
    """Run comprehensive RAG evaluation."""
    print("="*60)
    print("COMPREHENSIVE RAG EVALUATION")
    print("="*60)
    print(f"Start time: {datetime.now().isoformat()}")

    # Check if server is running
    evaluator = RAGEvaluator()

    try:
        resp = evaluator.client.get(f"{evaluator.base_url}/health")
        if resp.status_code != 200:
            print("Error: Server not healthy. Start the server first.")
            print("Run: python -m uvicorn app.main:app --reload")
            return
    except Exception as e:
        print(f"Error connecting to server: {e}")
        print("Start the server first: python -m uvicorn app.main:app --reload")
        return

    print("Server is healthy, starting evaluation...")

    # Run all tests
    access_results = evaluator.test_access_control()
    quality_results = evaluator.test_retrieval_quality()
    multihop_comparisons = evaluator.test_multihop_comparison()
    category_results = evaluator.test_by_category()

    # Generate report
    report = evaluator.generate_report(
        access_results, quality_results, multihop_comparisons, category_results
    )

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)

    print(f"\n📊 Overall Results:")
    print(f"  Access Control Accuracy: {report['summary']['access_control_accuracy']:.1%}")
    print(f"  Avg Context Precision: {report['summary']['avg_context_precision']:.2f}")
    print(f"  Avg Faithfulness: {report['summary']['avg_faithfulness']:.2f}")
    print(f"  Avg Answer Relevance: {report['summary']['avg_answer_relevance']:.2f}")
    print(f"  Multi-hop Beneficial: {report['summary']['multihop_beneficial']}")

    print(f"\n📁 Category Performance:")
    for cat, metrics in report["category_breakdown"].items():
        print(f"  {cat}: precision={metrics['avg_precision']:.2f}, faithfulness={metrics['avg_faithfulness']:.2f}")

    # Save report
    report_path = Path(__file__).parent.parent / "evaluation" / "reports"
    report_path.mkdir(parents=True, exist_ok=True)

    report_file = report_path / f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n📄 Report saved to: {report_file}")
    print(f"\nEnd time: {datetime.now().isoformat()}")

    return report


if __name__ == "__main__":
    main()
