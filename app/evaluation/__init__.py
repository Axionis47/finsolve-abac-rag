"""
RAG Evaluation Module using RAGAS metrics.

Provides evaluation capabilities for the RAG pipeline including:
- Context precision and recall
- Answer faithfulness and relevance
- Policy-aware evaluation per role
- Multi-hop vs single-hop comparison
"""
from app.evaluation.ragas_evaluator import RAGASEvaluator, EvaluationResult
from app.evaluation.policy_aware_eval import PolicyAwareEvaluator

__all__ = [
    "RAGASEvaluator",
    "EvaluationResult",
    "PolicyAwareEvaluator",
]
