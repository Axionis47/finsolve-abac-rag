"""
Policy-Aware Evaluation for ABAC RAG System.

Extends RAGAS evaluation to account for role-based access control,
ensuring evaluation metrics are computed with policy-filtered ground truth.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable

from app.evaluation.ragas_evaluator import RAGASEvaluator, EvaluationResult
from app.policy.pdp import PDP

logger = logging.getLogger(__name__)


@dataclass
class PolicyAwareResult:
    """Evaluation result with policy context."""
    role: str
    results: List[EvaluationResult]
    accessible_questions: int
    filtered_questions: int
    aggregated_metrics: Dict[str, Any]


class PolicyAwareEvaluator:
    """
    Policy-aware RAG evaluator.

    Filters ground truth based on user role permissions before evaluation,
    ensuring fair metrics that account for access control.
    """

    def __init__(
        self,
        pdp: PDP,
        base_evaluator: Optional[RAGASEvaluator] = None,
    ):
        """
        Initialize policy-aware evaluator.

        Args:
            pdp: Policy Decision Point for access control
            base_evaluator: Underlying RAGAS evaluator
        """
        self.pdp = pdp
        self.evaluator = base_evaluator or RAGASEvaluator()

    def filter_ground_truth_for_role(
        self,
        role: str,
        ground_truth_contexts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Filter ground truth contexts based on role permissions.

        Args:
            role: User role
            ground_truth_contexts: List of contexts with metadata

        Returns:
            Contexts the role can access
        """
        accessible = []

        for ctx in ground_truth_contexts:
            # Build resource from context metadata
            resource = {
                "owner_dept": ctx.get("owner_dept", "general"),
                "doc_type": ctx.get("doc_type", "md"),
                "sensitivity": ctx.get("sensitivity", "low"),
            }

            # Check if role can access
            decision = self.pdp.evaluate(
                subject={"role": role},
                resource=resource,
                action="retrieve_chunk",
            )

            if decision.effect == "permit":
                accessible.append(ctx)

        return accessible

    def evaluate_for_role(
        self,
        role: str,
        test_cases: List[Dict[str, Any]],
        query_fn: Callable[[str, str], Dict[str, Any]],
    ) -> PolicyAwareResult:
        """
        Evaluate RAG system for a specific role.

        Args:
            role: User role to evaluate
            test_cases: Test cases with questions and ground truth
            query_fn: Function to query RAG system (query, role) -> response

        Returns:
            PolicyAwareResult with role-specific metrics
        """
        results = []
        accessible_count = 0
        filtered_count = 0

        for case in test_cases:
            question = case["question"]
            applicable_roles = case.get("applicable_roles", [])

            # Check if this test case applies to the role
            if applicable_roles and role not in applicable_roles:
                filtered_count += 1
                continue

            accessible_count += 1

            # Filter ground truth contexts for this role
            gt_contexts = case.get("ground_truth_contexts", [])
            if gt_contexts and isinstance(gt_contexts[0], dict):
                filtered_gt = self.filter_ground_truth_for_role(role, gt_contexts)
                gt_context_texts = [c.get("text", "") for c in filtered_gt]
            else:
                gt_context_texts = gt_contexts if gt_contexts else None

            try:
                # Query the system as this role
                response = query_fn(question, role)

                result = self.evaluator.evaluate_full(
                    question=question,
                    answer=response.get("answer", ""),
                    contexts=[c.get("text", "") for c in response.get("contexts", [])],
                    ground_truth=case.get("ground_truth"),
                    ground_truth_contexts=gt_context_texts,
                    role=role,
                    latency_ms=response.get("latency_ms"),
                )
            except Exception as e:
                logger.error(f"Evaluation failed for {role}: {e}")
                result = EvaluationResult(
                    question=question,
                    answer="",
                    contexts=[],
                    role=role,
                    error=str(e),
                )

            results.append(result)

        # Aggregate results
        aggregated = self.evaluator.aggregate_results(results)

        return PolicyAwareResult(
            role=role,
            results=results,
            accessible_questions=accessible_count,
            filtered_questions=filtered_count,
            aggregated_metrics=aggregated,
        )

    def evaluate_all_roles(
        self,
        roles: List[str],
        test_cases: List[Dict[str, Any]],
        query_fn: Callable[[str, str], Dict[str, Any]],
    ) -> Dict[str, PolicyAwareResult]:
        """
        Evaluate RAG system across all roles.

        Args:
            roles: List of roles to evaluate
            test_cases: Test cases with questions and ground truth
            query_fn: Function to query RAG system

        Returns:
            Dictionary mapping role to PolicyAwareResult
        """
        results = {}

        for role in roles:
            logger.info(f"Evaluating role: {role}")
            results[role] = self.evaluate_for_role(role, test_cases, query_fn)

        return results

    def compare_multihop(
        self,
        question: str,
        role: str,
        query_fn_single: Callable[[str, str], Dict[str, Any]],
        query_fn_multihop: Callable[[str, str], Dict[str, Any]],
        ground_truth: Optional[str] = None,
        ground_truth_contexts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compare single-hop vs multi-hop retrieval for a query.

        Args:
            question: The query to evaluate
            role: User role
            query_fn_single: Single-hop query function
            query_fn_multihop: Multi-hop query function
            ground_truth: Optional ground truth answer
            ground_truth_contexts: Optional ground truth contexts

        Returns:
            Comparison results with metrics for both approaches
        """
        import time

        # Single-hop evaluation
        t0 = time.perf_counter()
        single_response = query_fn_single(question, role)
        single_latency = (time.perf_counter() - t0) * 1000

        single_result = self.evaluator.evaluate_full(
            question=question,
            answer=single_response.get("answer", ""),
            contexts=[c.get("text", "") for c in single_response.get("contexts", [])],
            ground_truth=ground_truth,
            ground_truth_contexts=ground_truth_contexts,
            role=role,
            latency_ms=single_latency,
        )

        # Multi-hop evaluation
        t0 = time.perf_counter()
        multihop_response = query_fn_multihop(question, role)
        multihop_latency = (time.perf_counter() - t0) * 1000

        multihop_result = self.evaluator.evaluate_full(
            question=question,
            answer=multihop_response.get("answer", ""),
            contexts=[c.get("text", "") for c in multihop_response.get("contexts", [])],
            ground_truth=ground_truth,
            ground_truth_contexts=ground_truth_contexts,
            role=role,
            latency_ms=multihop_latency,
        )

        # Compute improvements
        def safe_diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
            if a is None or b is None:
                return None
            return a - b

        return {
            "question": question,
            "role": role,
            "single_hop": single_result.to_dict(),
            "multi_hop": multihop_result.to_dict(),
            "comparison": {
                "latency_overhead_ms": multihop_latency - single_latency,
                "context_precision_diff": safe_diff(
                    multihop_result.context_precision,
                    single_result.context_precision,
                ),
                "context_recall_diff": safe_diff(
                    multihop_result.context_recall,
                    single_result.context_recall,
                ),
                "faithfulness_diff": safe_diff(
                    multihop_result.faithfulness,
                    single_result.faithfulness,
                ),
                "answer_relevance_diff": safe_diff(
                    multihop_result.answer_relevance,
                    single_result.answer_relevance,
                ),
            },
            "recommendation": self._recommend_approach(
                single_result, multihop_result, single_latency, multihop_latency
            ),
        }

    def _recommend_approach(
        self,
        single: EvaluationResult,
        multihop: EvaluationResult,
        single_latency: float,
        multihop_latency: float,
    ) -> str:
        """Recommend single-hop or multi-hop based on metrics."""
        # Score each approach
        def score(result: EvaluationResult) -> float:
            metrics = [
                result.context_precision,
                result.context_recall,
                result.faithfulness,
                result.answer_relevance,
            ]
            valid = [m for m in metrics if m is not None]
            return sum(valid) / len(valid) if valid else 0

        single_score = score(single)
        multihop_score = score(multihop)

        # Factor in latency (penalize if >2x slower)
        latency_ratio = multihop_latency / single_latency if single_latency > 0 else 1

        # Adjust multihop score based on latency overhead
        if latency_ratio > 2:
            multihop_score *= 0.9
        if latency_ratio > 3:
            multihop_score *= 0.8

        if multihop_score > single_score + 0.1:
            return "multi_hop"
        elif single_score > multihop_score + 0.1:
            return "single_hop"
        else:
            # If similar quality, prefer faster approach
            return "single_hop" if single_latency < multihop_latency else "multi_hop"
