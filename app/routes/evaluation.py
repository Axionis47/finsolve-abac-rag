"""
Evaluation routes for RAG quality metrics.

Admin-only endpoints for running RAGAS evaluations.
"""
from __future__ import annotations
import json
import os
from typing import Optional, List
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field

from app.routes.deps import authenticate, get_pdp
from app.utils.audit import gen_correlation_id
from app.evaluation import RAGASEvaluator, PolicyAwareEvaluator

router = APIRouter(prefix="/eval", tags=["evaluation"])


def _require_c_level(user: dict):
    """Ensure user is c_level for admin operations."""
    if user["role"] != "c_level":
        raise HTTPException(
            status_code=403,
            detail="Only c_level can access evaluation endpoints"
        )


class EvalRunRequest(BaseModel):
    """Request to run evaluation."""
    dataset_path: Optional[str] = Field(
        default=None,
        description="Path to ground truth dataset JSON"
    )
    roles: Optional[List[str]] = Field(
        default=None,
        description="Roles to evaluate (default: all)"
    )
    limit: Optional[int] = Field(
        default=None,
        ge=1,
        le=100,
        description="Limit number of questions to evaluate"
    )


class MultihopCompareRequest(BaseModel):
    """Request to compare single-hop vs multi-hop."""
    question: str = Field(..., description="Question to evaluate")
    role: str = Field(default="employee", description="Role for evaluation")
    ground_truth: Optional[str] = Field(default=None, description="Expected answer")


@router.get("/datasets")
def list_datasets(response: Response, user=Depends(authenticate)):
    """List available evaluation datasets."""
    _require_c_level(user)
    cid = gen_correlation_id()
    response.headers["X-Correlation-ID"] = cid

    datasets_dir = Path("evaluation/datasets")
    datasets = []

    if datasets_dir.exists():
        for f in datasets_dir.glob("*.json"):
            try:
                with open(f, "r") as fp:
                    data = json.load(fp)
                datasets.append({
                    "name": f.stem,
                    "path": str(f),
                    "version": data.get("version", "unknown"),
                    "question_count": len(data.get("questions", [])),
                })
            except Exception as e:
                datasets.append({
                    "name": f.stem,
                    "path": str(f),
                    "error": str(e),
                })

    return {
        "datasets": datasets,
        "correlation_id": cid,
    }


@router.post("/run")
def run_evaluation(
    req: EvalRunRequest,
    response: Response,
    user=Depends(authenticate),
):
    """
    Run RAGAS evaluation on a dataset.

    Evaluates the RAG system using ground truth data and returns
    metrics for context precision, recall, faithfulness, and relevance.
    """
    _require_c_level(user)
    cid = gen_correlation_id()
    response.headers["X-Correlation-ID"] = cid

    # Load dataset
    dataset_path = req.dataset_path or "evaluation/datasets/ground_truth.json"

    if not os.path.exists(dataset_path):
        raise HTTPException(
            status_code=404,
            detail=f"Dataset not found: {dataset_path}"
        )

    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    questions = dataset.get("questions", [])
    if req.limit:
        questions = questions[:req.limit]

    # Determine roles to evaluate
    roles = req.roles or [r["name"] for r in dataset.get("roles", [])]
    if not roles:
        roles = ["employee"]

    # Create evaluator
    evaluator = RAGASEvaluator(use_ragas=True)

    # Run simple evaluation (without actual RAG queries for now)
    # In production, this would call the actual RAG pipeline
    results_by_role = {}

    for role in roles:
        role_results = []
        for q in questions:
            applicable = q.get("applicable_roles", [])
            if applicable and role not in applicable:
                continue

            # Create a mock result (in production, query the RAG system)
            result = evaluator.evaluate_full(
                question=q["question"],
                answer=q.get("ground_truth", ""),  # Use GT as answer for baseline
                contexts=q.get("ground_truth_contexts", []),
                ground_truth=q.get("ground_truth"),
                ground_truth_contexts=q.get("ground_truth_contexts"),
                role=role,
            )
            role_results.append(result.to_dict())

        results_by_role[role] = {
            "results": role_results,
            "count": len(role_results),
            "aggregated": evaluator.aggregate_results(
                [evaluator.evaluate_full(
                    question=r["question"],
                    answer=r["answer"],
                    contexts=r["contexts"],
                ) for r in role_results]
            ) if role_results else {},
        }

    return {
        "dataset": dataset_path,
        "dataset_version": dataset.get("version"),
        "roles_evaluated": roles,
        "total_questions": len(questions),
        "results": results_by_role,
        "correlation_id": cid,
    }


@router.post("/compare/multihop")
def compare_multihop(
    req: MultihopCompareRequest,
    response: Response,
    user=Depends(authenticate),
):
    """
    Compare single-hop vs multi-hop retrieval for a query.

    Returns metrics for both approaches and a recommendation.
    """
    _require_c_level(user)
    cid = gen_correlation_id()
    response.headers["X-Correlation-ID"] = cid

    evaluator = RAGASEvaluator(use_ragas=True)

    # In production, these would be actual query functions
    # For now, return a placeholder response
    return {
        "question": req.question,
        "role": req.role,
        "message": "Multi-hop comparison requires live RAG system. "
                   "Use the PolicyAwareEvaluator.compare_multihop() method "
                   "programmatically for full comparison.",
        "example_usage": {
            "python": """
from app.evaluation import PolicyAwareEvaluator
from app.policy.pdp import PDP

pdp = PDP.load("docs/policy.yaml")
evaluator = PolicyAwareEvaluator(pdp)

result = evaluator.compare_multihop(
    question="Compare Q2 and Q3 revenue",
    role="finance",
    query_fn_single=lambda q, r: rag_query(q, r, multihop=False),
    query_fn_multihop=lambda q, r: rag_query(q, r, multihop=True),
)
"""
        },
        "correlation_id": cid,
    }


@router.get("/metrics/summary")
def metrics_summary(response: Response, user=Depends(authenticate)):
    """Get a summary of evaluation capabilities."""
    _require_c_level(user)
    cid = gen_correlation_id()
    response.headers["X-Correlation-ID"] = cid

    return {
        "available_metrics": {
            "retrieval": [
                {
                    "name": "context_precision",
                    "description": "Percentage of retrieved contexts that are relevant",
                    "range": "0.0 - 1.0",
                },
                {
                    "name": "context_recall",
                    "description": "Percentage of relevant contexts that were retrieved",
                    "range": "0.0 - 1.0",
                },
            ],
            "generation": [
                {
                    "name": "faithfulness",
                    "description": "Is the answer grounded in the provided context?",
                    "range": "0.0 - 1.0",
                },
                {
                    "name": "answer_relevance",
                    "description": "Does the answer address the question?",
                    "range": "0.0 - 1.0",
                },
            ],
        },
        "evaluation_modes": [
            "single_query",
            "batch_evaluation",
            "role_comparison",
            "multihop_comparison",
        ],
        "ragas_available": _check_ragas_available(),
        "correlation_id": cid,
    }


def _check_ragas_available() -> bool:
    """Check if RAGAS library is installed."""
    try:
        import ragas
        return True
    except ImportError:
        return False
