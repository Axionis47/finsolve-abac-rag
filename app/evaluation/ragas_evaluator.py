"""
RAGAS Evaluator for RAG Pipeline.

Integrates RAGAS metrics for evaluating retrieval and generation quality.
Supports both online (per-query) and batch evaluation modes.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of a single evaluation."""
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None
    ground_truth_contexts: Optional[List[str]] = None

    # Retrieval metrics
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None

    # Generation metrics
    faithfulness: Optional[float] = None
    answer_relevance: Optional[float] = None

    # Metadata
    role: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    latency_ms: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "question": self.question,
            "answer": self.answer,
            "contexts": self.contexts,
            "ground_truth": self.ground_truth,
            "metrics": {
                "context_precision": self.context_precision,
                "context_recall": self.context_recall,
                "faithfulness": self.faithfulness,
                "answer_relevance": self.answer_relevance,
            },
            "metadata": {
                "role": self.role,
                "timestamp": self.timestamp,
                "latency_ms": self.latency_ms,
                "error": self.error,
            },
        }


class RAGASEvaluator:
    """
    Evaluator using RAGAS metrics.

    Supports evaluation of:
    - Retrieval quality (context precision, recall)
    - Generation quality (faithfulness, answer relevance)
    """

    def __init__(self, use_ragas: bool = True):
        """
        Initialize the evaluator.

        Args:
            use_ragas: Whether to use RAGAS library (requires installation)
        """
        self.use_ragas = use_ragas
        self._ragas_available = False

        if use_ragas:
            try:
                import ragas
                self._ragas_available = True
                logger.info("RAGAS library available for evaluation")
            except ImportError:
                logger.warning(
                    "RAGAS library not installed. Install with: pip install ragas"
                )
                self._ragas_available = False

    def evaluate_retrieval(
        self,
        question: str,
        retrieved_contexts: List[str],
        ground_truth_contexts: List[str],
    ) -> Dict[str, float]:
        """
        Evaluate retrieval quality.

        Args:
            question: The query
            retrieved_contexts: Contexts retrieved by the system
            ground_truth_contexts: Expected relevant contexts

        Returns:
            Dictionary with precision and recall scores
        """
        if not retrieved_contexts:
            return {"context_precision": 0.0, "context_recall": 0.0}

        if not ground_truth_contexts:
            return {"context_precision": 1.0, "context_recall": 1.0}

        if self._ragas_available:
            return self._evaluate_retrieval_ragas(
                question, retrieved_contexts, ground_truth_contexts
            )

        # Fallback: Simple overlap-based metrics
        return self._evaluate_retrieval_simple(
            retrieved_contexts, ground_truth_contexts
        )

    def _evaluate_retrieval_ragas(
        self,
        question: str,
        retrieved_contexts: List[str],
        ground_truth_contexts: List[str],
    ) -> Dict[str, float]:
        """Evaluate using RAGAS metrics."""
        try:
            from ragas.metrics import context_precision, context_recall
            from datasets import Dataset

            # Create dataset for RAGAS
            data = {
                "question": [question],
                "contexts": [retrieved_contexts],
                "ground_truth": [" ".join(ground_truth_contexts)],
            }
            dataset = Dataset.from_dict(data)

            # Compute metrics
            precision_score = context_precision.score(dataset)
            recall_score = context_recall.score(dataset)

            return {
                "context_precision": precision_score,
                "context_recall": recall_score,
            }
        except Exception as e:
            logger.error(f"RAGAS retrieval evaluation failed: {e}")
            return self._evaluate_retrieval_simple(
                retrieved_contexts, ground_truth_contexts
            )

    def _evaluate_retrieval_simple(
        self,
        retrieved_contexts: List[str],
        ground_truth_contexts: List[str],
    ) -> Dict[str, float]:
        """Simple overlap-based retrieval metrics."""
        # Normalize contexts for comparison
        retrieved_set = set(c.lower().strip() for c in retrieved_contexts)
        ground_truth_set = set(c.lower().strip() for c in ground_truth_contexts)

        # Count overlaps (using substring matching for flexibility)
        relevant_retrieved = 0
        for rc in retrieved_set:
            for gt in ground_truth_set:
                if gt in rc or rc in gt:
                    relevant_retrieved += 1
                    break

        precision = relevant_retrieved / len(retrieved_set) if retrieved_set else 0
        recall = relevant_retrieved / len(ground_truth_set) if ground_truth_set else 0

        return {
            "context_precision": precision,
            "context_recall": recall,
        }

    def evaluate_generation(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Evaluate generation quality.

        Args:
            question: The query
            answer: Generated answer
            contexts: Contexts used for generation
            ground_truth: Optional ground truth answer

        Returns:
            Dictionary with faithfulness and relevance scores
        """
        if not answer or not contexts:
            return {"faithfulness": 0.0, "answer_relevance": 0.0}

        if self._ragas_available:
            return self._evaluate_generation_ragas(
                question, answer, contexts, ground_truth
            )

        # Fallback: Simple heuristic-based metrics
        return self._evaluate_generation_simple(question, answer, contexts)

    def _evaluate_generation_ragas(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str],
    ) -> Dict[str, float]:
        """Evaluate using RAGAS metrics."""
        try:
            from ragas.metrics import faithfulness, answer_relevancy
            from datasets import Dataset

            data = {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
            }
            if ground_truth:
                data["ground_truth"] = [ground_truth]

            dataset = Dataset.from_dict(data)

            faith_score = faithfulness.score(dataset)
            relevance_score = answer_relevancy.score(dataset)

            return {
                "faithfulness": faith_score,
                "answer_relevance": relevance_score,
            }
        except Exception as e:
            logger.error(f"RAGAS generation evaluation failed: {e}")
            return self._evaluate_generation_simple(question, answer, contexts)

    def _evaluate_generation_simple(
        self,
        question: str,
        answer: str,
        contexts: List[str],
    ) -> Dict[str, float]:
        """Simple heuristic-based generation metrics."""
        # Faithfulness: Check if answer content appears in contexts
        combined_context = " ".join(contexts).lower()
        answer_words = set(answer.lower().split())
        context_words = set(combined_context.split())

        if not answer_words:
            faithfulness = 0.0
        else:
            overlap = len(answer_words & context_words)
            faithfulness = overlap / len(answer_words)

        # Relevance: Check if question keywords appear in answer
        question_words = set(question.lower().split())
        question_words -= {"what", "how", "why", "when", "where", "is", "the", "a", "an"}

        if not question_words:
            relevance = 1.0
        else:
            overlap = len(question_words & answer_words)
            relevance = overlap / len(question_words)

        return {
            "faithfulness": min(faithfulness, 1.0),
            "answer_relevance": min(relevance, 1.0),
        }

    def evaluate_full(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
        ground_truth_contexts: Optional[List[str]] = None,
        role: Optional[str] = None,
        latency_ms: Optional[float] = None,
    ) -> EvaluationResult:
        """
        Run full evaluation on a single query.

        Args:
            question: The query
            answer: Generated answer
            contexts: Retrieved contexts
            ground_truth: Optional ground truth answer
            ground_truth_contexts: Optional ground truth contexts
            role: User role for policy-aware evaluation
            latency_ms: Query latency

        Returns:
            EvaluationResult with all metrics
        """
        result = EvaluationResult(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
            ground_truth_contexts=ground_truth_contexts,
            role=role,
            latency_ms=latency_ms,
        )

        try:
            # Retrieval metrics
            if ground_truth_contexts:
                retrieval_metrics = self.evaluate_retrieval(
                    question, contexts, ground_truth_contexts
                )
                result.context_precision = retrieval_metrics["context_precision"]
                result.context_recall = retrieval_metrics["context_recall"]

            # Generation metrics
            generation_metrics = self.evaluate_generation(
                question, answer, contexts, ground_truth
            )
            result.faithfulness = generation_metrics["faithfulness"]
            result.answer_relevance = generation_metrics["answer_relevance"]

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            result.error = str(e)

        return result

    def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]],
        query_fn: Callable[[str], Dict[str, Any]],
    ) -> List[EvaluationResult]:
        """
        Evaluate a batch of test cases.

        Args:
            test_cases: List of test cases with 'question', 'ground_truth', etc.
            query_fn: Function to query the RAG system

        Returns:
            List of EvaluationResults
        """
        results = []

        for case in test_cases:
            question = case["question"]

            try:
                # Query the system
                response = query_fn(question)

                result = self.evaluate_full(
                    question=question,
                    answer=response.get("answer", ""),
                    contexts=[c.get("text", "") for c in response.get("contexts", [])],
                    ground_truth=case.get("ground_truth"),
                    ground_truth_contexts=case.get("ground_truth_contexts"),
                    role=case.get("role"),
                    latency_ms=response.get("latency_ms"),
                )
            except Exception as e:
                result = EvaluationResult(
                    question=question,
                    answer="",
                    contexts=[],
                    error=str(e),
                )

            results.append(result)

        return results

    def aggregate_results(
        self,
        results: List[EvaluationResult],
    ) -> Dict[str, Any]:
        """
        Aggregate evaluation results.

        Args:
            results: List of evaluation results

        Returns:
            Aggregated metrics with mean, min, max
        """
        valid_results = [r for r in results if r.error is None]

        if not valid_results:
            return {"error": "No valid results to aggregate"}

        def safe_mean(values: List[Optional[float]]) -> Optional[float]:
            filtered = [v for v in values if v is not None]
            return sum(filtered) / len(filtered) if filtered else None

        return {
            "count": len(results),
            "valid_count": len(valid_results),
            "error_count": len(results) - len(valid_results),
            "metrics": {
                "context_precision": {
                    "mean": safe_mean([r.context_precision for r in valid_results]),
                },
                "context_recall": {
                    "mean": safe_mean([r.context_recall for r in valid_results]),
                },
                "faithfulness": {
                    "mean": safe_mean([r.faithfulness for r in valid_results]),
                },
                "answer_relevance": {
                    "mean": safe_mean([r.answer_relevance for r in valid_results]),
                },
            },
        }
