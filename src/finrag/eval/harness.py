"""The eval run: gold questions in, one report dict out.

This is the scoreboard for every rung. Give it a retriever, the policy engine
and the gold set; it asks each question in that question's role, scores the
answer with `metrics.py`, and returns the report. Saving that report to disk is
the job of the neighbouring `report_io.py`.

The three kinds of question are scored differently, because they are not asking
the same thing. "answer" questions get the full retrieval treatment. The T6
"refuse_role" and "no_answer" probes are scored on leakage alone and stay out of
the recall averages, since averaging them in would reward a retriever for
returning nothing. The ambiguous "retrieve_both" questions are scored only on
whether both conflicting sources came back.

Leakage is counted for every question regardless of kind. It must be zero.

Empty groups report null, never 0.0. A tier where no question labels a section
has not scored zero on sections, it was never measured there.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from finrag.eval import metrics
from finrag.policy.engine import PolicyEngine
from finrag.retrieval.base import Retriever

# Question kinds, derived from the gold set's `expected_behavior` field.
POLICY_BEHAVIORS = ("refuse_role", "no_answer")
BOTH_SOURCES_BEHAVIOR = "retrieve_both"

SCORING_RETRIEVAL = "retrieval"
SCORING_POLICY = "policy"
SCORING_BOTH_SOURCES = "both_sources"

# Floats are rounded before they reach the JSON so reports stay readable and
# two runs of the same fixture compare byte for byte.
DECIMALS = 4


def scoring_kind(expected_behavior: str) -> str:
    """Which of the three scoring rules this question falls under."""
    if expected_behavior in POLICY_BEHAVIORS:
        return SCORING_POLICY
    if expected_behavior == BOTH_SOURCES_BEHAVIOR:
        return SCORING_BOTH_SOURCES
    return SCORING_RETRIEVAL


class EvalHarness:
    """Runs the gold set against one retriever and reports the numbers."""

    def __init__(
        self,
        retriever: Retriever,
        policy: PolicyEngine,
        gold: Sequence[dict[str, Any]],
    ) -> None:
        self.retriever = retriever
        self.policy = policy
        self.gold = list(gold)

    def run(self, k_values: Sequence[int] = (5, 10)) -> dict[str, Any]:
        """Ask every gold question and build the report dict."""
        k_values = tuple(k_values)
        max_k = max(k_values)

        rows: list[dict[str, Any]] = []
        for entry in self.gold:
            rows.append(self._score_one(entry, k_values, max_k))

        return {
            "mode": getattr(self.retriever, "name", "unknown"),
            "k_values": list(k_values),
            "n_questions": len(rows),
            "overall": summarise(rows, k_values),
            "per_tier": self._per_tier(rows, k_values),
            "ambiguous": ambiguous_summary(rows, max_k),
            "policy_probes": policy_summary(rows),
            "per_question": rows,
        }

    def _score_one(
        self,
        entry: dict[str, Any],
        k_values: tuple[int, ...],
        max_k: int,
    ) -> dict[str, Any]:
        """Retrieve for one question and turn the result into a report row."""
        role = entry["role"]
        hits = self.retriever.search(entry["question"], role, k=max_k)

        leaked = metrics.leaked_chunk_ids(self.policy, role, hits)
        scores = metrics.score_question(hits, entry, k_values)

        row: dict[str, Any] = {
            "id": entry["id"],
            "tier": str(entry["tier"]),
            "role": role,
            "expected_behavior": entry.get("expected_behavior", "answer"),
            "scoring": scoring_kind(entry.get("expected_behavior", "answer")),
            "n_hits": len(hits),
            "retrieved_docs": metrics.retrieved_docs(hits, max_k),
            "leakage": len(leaked),
            "leaked_chunk_ids": leaked,
            "policy_pass": len(leaked) == 0,
            f"both_sources@{max_k}": metrics.both_sources_hit(
                hits, entry.get("gold_docs") or [], max_k
            ),
        }
        # The retrieval numbers are plain arithmetic, so we record them for
        # every question. The `scoring` field above decides which of them the
        # averages are allowed to use.
        for key in scores:
            row[key] = _round(scores[key])
        return row

    def _per_tier(
        self,
        rows: Sequence[dict[str, Any]],
        k_values: tuple[int, ...],
    ) -> dict[str, dict[str, Any]]:
        """One summary block per tier, keyed by the tier name, in sorted order."""
        tiers: list[str] = []
        for row in rows:
            if row["tier"] not in tiers:
                tiers.append(row["tier"])
        tiers.sort()

        per_tier: dict[str, dict[str, Any]] = {}
        for tier in tiers:
            tier_rows = _rows_where(rows, "tier", tier)
            per_tier[tier] = summarise(tier_rows, k_values)
        return per_tier


def _rows_where(rows: Sequence[dict[str, Any]], key: str, wanted: str) -> list[dict[str, Any]]:
    """The rows whose `key` equals `wanted`, keeping their order."""
    picked: list[dict[str, Any]] = []
    for row in rows:
        if row[key] == wanted:
            picked.append(row)
    return picked


def _round(value: Any) -> Any:
    """Round floats for the report; leave bools, ints and strings alone."""
    if isinstance(value, float):
        return round(value, DECIMALS)
    return value


def _mean(values: Sequence[float]) -> float:
    """Average, with an empty list scoring 0.0 rather than blowing up."""
    if not values:
        return 0.0
    total = 0.0
    for value in values:
        total += value
    return round(total / len(values), DECIMALS)


def _column(rows: Sequence[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        values.append(float(row[key]))
    return values


def _mean_or_none(values: Sequence[float]) -> float | None:
    """Average, or None when there was nothing to average and 0.0 would lie."""
    if not values:
        return None
    return _mean(values)


def summarise(rows: Sequence[dict[str, Any]], k_values: Sequence[int]) -> dict[str, Any]:
    """Average the retrieval metrics over the questions that earn them.

    Only `retrieval` questions feed recall, precision, F1 and MRR. Leakage is
    summed over every question in the group, because a policy breach anywhere
    is a policy breach.
    """
    scored = _rows_where(rows, "scoring", SCORING_RETRIEVAL)

    summary: dict[str, Any] = {"n": len(rows), "n_scored": len(scored)}
    for k in k_values:
        summary[f"recall@{k}"] = _mean(_column(scored, f"recall@{k}"))
        summary[f"precision@{k}"] = _mean(_column(scored, f"precision@{k}"))
        summary[f"f1@{k}"] = _mean(_column(scored, f"f1@{k}"))
    summary["mrr"] = _mean(_column(scored, "rr"))

    # A question with no gold sections cannot hit or miss one, so it is left out
    # of the denominator. An empty denominator makes both section metrics null.
    section_rows: list[dict[str, Any]] = []
    for row in scored:
        if row["has_gold_sections"]:
            section_rows.append(row)
    summary["n_section_scored"] = len(section_rows)
    summary["section_hit_rate"] = _mean_or_none(_column(section_rows, "section_hit"))
    summary["section_mrr"] = _mean_or_none(_column(section_rows, "section_rr"))

    leakage = 0
    leaking_questions = 0
    for row in rows:
        leakage += row["leakage"]
        if row["leakage"] > 0:
            leaking_questions += 1
    summary["leakage_count"] = leakage
    summary["n_leaking_questions"] = leaking_questions
    return summary


def policy_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """How the T6 policy probes did. A probe passes when nothing leaked.

    With no probes in the group the pass rate is null, not 1.0 and not 0.0.
    Neither number would be honest about a check that never ran.
    """
    probes = _rows_where(rows, "scoring", SCORING_POLICY)

    passed = 0
    for row in probes:
        if row["policy_pass"]:
            passed += 1

    return {
        "n": len(probes),
        "n_pass": passed,
        "pass_rate": _mean_or_none(_column(probes, "policy_pass")),
    }


def ambiguous_summary(rows: Sequence[dict[str, Any]], max_k: int) -> dict[str, Any]:
    """Share of ambiguous questions that returned every conflicting source."""
    both = _rows_where(rows, "scoring", SCORING_BOTH_SOURCES)
    key = f"both_sources@{max_k}"

    passed = 0
    for row in both:
        if row[key]:
            passed += 1

    return {
        "n": len(both),
        "n_pass": passed,
        f"both_sources_recall@{max_k}": _mean(_column(both, key)),
    }
