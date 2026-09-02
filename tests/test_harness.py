"""The eval harness over a scripted retriever, with the whole report known up front.

`EvalHarness` asks each gold question in that question's role, scores the answer
and folds the rows into one report. To pin those numbers we hand it a retriever
that returns a fixed chunk list per question, so every average below was worked
out by hand from the six questions in `QUESTIONS`.

What the run has to get right: "answer" questions feed recall, precision, F1 and
MRR; "refuse_role" and "no_answer" probes are scored on leakage alone and stay
out of those averages; "retrieve_both" questions are scored only on returning
every gold source. Leakage is counted for all six, and one question deliberately
returns a chunk its role may not see so the count has something to find.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from finrag.eval import harness, report_io
from finrag.eval.harness import EvalHarness
from finrag.models import Chunk, Hit
from finrag.pipeline import StrategySelection
from finrag.policy.engine import PolicyEngine
from finrag.querying.base import QueryResult

FINANCE_DOC = "finance/quarterly_financial_report"
Q1_DOC = "marketing/market_report_q1_2024"
Q3_DOC = "marketing/market_report_q3_2024"

FIN_REVENUE = f"{FINANCE_DOC}#q4-revenue"
FIN_RISKS = f"{FINANCE_DOC}#q4-risks"
Q1_SPEND = f"{Q1_DOC}#campaign-spend"
Q3_SPEND = f"{Q3_DOC}#campaign-spend"

K_VALUES = (1, 3)
MAX_K = 3


def question(qid: str, tier: str, role: str, behavior: str,
             gold_docs: list[str], gold_sections: list[str],
             hits: list[str]) -> dict[str, Any]:
    """One gold entry plus the chunk ids the scripted retriever will return."""
    # The question text is the id, so the script stays readable next to the
    # assertions further down.
    return {
        "id": qid,
        "tier": tier,
        "question": qid,
        "role": role,
        "expected_behavior": behavior,
        "gold_docs": gold_docs,
        "gold_sections": gold_sections,
        "hits": hits,
    }


QUESTIONS: list[dict[str, Any]] = [
    # Gold document at hit 1, a second document by hit 3, so precision@3 is 0.5.
    question("t1-revenue", "T1", "c_level", "answer",
             [FINANCE_DOC], [FIN_REVENUE], [FIN_REVENUE, Q1_SPEND, FIN_RISKS]),
    # The wrong quarter first, the gold one second, so the RR is 1/2.
    question("t2-spend", "T2", "marketing", "answer",
             [Q1_DOC], [], [Q3_SPEND, Q1_SPEND]),
    # A finance chunk handed to marketing: the planted leak.
    question("t6-leak", "T6", "marketing", "refuse_role", [], [], [FIN_REVENUE]),
    # A probe that correctly returns nothing.
    question("t6-clean", "T6", "marketing", "no_answer", [], [], []),
    # Both conflicting sources, three hits apart.
    question("t4-both-pass", "T4", "c_level", "retrieve_both",
             [Q1_DOC, Q3_DOC], [], [Q1_SPEND, FIN_REVENUE, Q3_SPEND]),
    # Only one of the two conflicting sources.
    question("t4-both-fail", "T4", "c_level", "retrieve_both",
             [Q1_DOC, Q3_DOC], [], [Q1_SPEND]),
]


# Split the table in two: the gold set the harness reads, and the script the
# fake retriever answers from.
GOLD: list[dict[str, Any]] = []
SCRIPT: dict[str, list[str]] = {}
for row in QUESTIONS:
    entry = dict(row)
    SCRIPT[entry["question"]] = entry.pop("hits")
    GOLD.append(entry)


class ScriptedRetriever:
    """Returns a fixed chunk list per question, in the order written in SCRIPT.

    It ignores the role on purpose. A real retriever masks first and could never
    hand marketing a finance chunk, and then the harness would have no leakage
    to count. Scripting the leak is the only way to test that it is caught.
    """

    name = "scripted"

    def __init__(self, chunks: list[Chunk], script: dict[str, list[str]]) -> None:
        self.script = script
        self.chunk_by_id: dict[str, Chunk] = {}
        for chunk in chunks:
            self.chunk_by_id[chunk.chunk_id] = chunk

    def search(self, query: str, role: str, k: int = 10) -> list[Hit]:
        chunk_ids = self.script[query][:k]

        hits: list[Hit] = []
        rank = 1
        for chunk_id in chunk_ids:
            hits.append(self._to_hit(self.chunk_by_id[chunk_id], rank))
            rank += 1
        return hits

    def _to_hit(self, chunk: Chunk, rank: int) -> Hit:
        return Hit(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            section_slug=chunk.section_slug,
            rank=rank,
            score=1.0 / rank,
            retriever=self.name,
            owner_dept=chunk.owner_dept,
            doc_type=chunk.doc_type,
            sensitivity=chunk.sensitivity,
            allowed_roles=chunk.allowed_roles,
            text=chunk.text,
        )


class ScriptedPipeline:
    """A deliberately leaky pipeline stub so the harness can count the leak."""

    selection = StrategySelection(
        retrieval_strategy="scripted",
        query_strategy="direct",
        reranking_strategy="none",
    )

    def __init__(self, retriever: ScriptedRetriever) -> None:
        self.retriever = retriever

    def search(self, question: str, role: str, k: int = 10) -> QueryResult:
        return QueryResult(self.retriever.search(question, role, k))


@pytest.fixture(scope="module")
def policy(policy_path: Path) -> PolicyEngine:
    return PolicyEngine.from_yaml(policy_path)


@pytest.fixture
def report(micro_corpus: list[Chunk], policy: PolicyEngine) -> dict[str, Any]:
    """One full run of the six scripted questions, shared by most tests."""
    retriever = ScriptedRetriever(micro_corpus, SCRIPT)
    return EvalHarness(ScriptedPipeline(retriever), policy, GOLD).run(k_values=K_VALUES)


def row_for(report: dict[str, Any], question_id: str) -> dict[str, Any]:
    for row in report["per_question"]:
        if row["id"] == question_id:
            return row
    raise AssertionError(f"no row for {question_id}")


def test_scoring_kind_sorts_the_three_question_types() -> None:
    assert harness.scoring_kind("answer") == "retrieval"
    assert harness.scoring_kind("refuse_role") == "policy"
    assert harness.scoring_kind("no_answer") == "policy"
    assert harness.scoring_kind("retrieve_both") == "both_sources"


def test_report_has_the_agreed_schema(report: dict[str, Any]) -> None:
    assert set(report) == {
        "pipeline_id", "retrieval_strategy", "query_strategy",
        "reranking_strategy", "k_values", "n_questions", "overall",
        "per_tier", "ambiguous", "policy_probes", "per_question",
    }
    assert report["pipeline_id"] == "direct__scripted__none"
    assert report["k_values"] == [1, 3]
    assert report["n_questions"] == 6
    assert len(report["per_question"]) == 6


def test_the_row_for_a_plain_answer_question(report: dict[str, Any]) -> None:
    """Gold document at hit 1, a second document by hit 3, so precision@3 is 0.5."""
    row = row_for(report, "t1-revenue")

    assert row["scoring"] == "retrieval"
    assert row["retrieved_docs"] == [FINANCE_DOC, Q1_DOC]
    assert row["recall@1"] == 1.0
    assert row["precision@1"] == 1.0
    assert row["f1@1"] == 1.0
    assert row["recall@3"] == 1.0
    assert row["precision@3"] == 0.5
    assert row["f1@3"] == 0.6667
    assert row["rr"] == 1.0
    assert row["section_hit"] is True
    # The gold section is the very first hit, so the section RR is 1/1.
    assert row["section_rr"] == 1.0
    assert row["leakage"] == 0


def test_the_row_for_a_question_whose_gold_doc_arrives_second(report: dict[str, Any]) -> None:
    row = row_for(report, "t2-spend")

    assert row["retrieved_docs"] == [Q3_DOC, Q1_DOC]
    assert row["recall@1"] == 0.0
    assert row["recall@3"] == 1.0
    assert row["rr"] == 0.5
    assert row["has_gold_sections"] is False
    assert row["section_hit"] is False
    assert row["section_rr"] == 0.0


def test_overall_averages_only_the_retrieval_questions(report: dict[str, Any]) -> None:
    """Two of the six questions earn recall; the other four must not dilute it.

    The section-hit rate has a denominator of its own: only t1-revenue labels a
    section, so a question without one is left out rather than counted a miss.
    """
    overall = report["overall"]

    assert overall["n"] == 6
    assert overall["n_scored"] == 2
    assert overall["recall@1"] == 0.5
    assert overall["precision@1"] == 0.5
    assert overall["f1@1"] == 0.5
    assert overall["recall@3"] == 1.0
    assert overall["precision@3"] == 0.5
    assert overall["f1@3"] == 0.6667
    assert overall["mrr"] == 0.75
    assert overall["n_section_scored"] == 1
    assert overall["section_hit_rate"] == 1.0
    assert overall["section_mrr"] == 1.0


def test_per_tier_numbers(report: dict[str, Any]) -> None:
    """One tier block each, in sorted order, scored on that tier's rows alone."""
    per_tier = report["per_tier"]
    assert list(per_tier) == ["T1", "T2", "T4", "T6"]

    assert per_tier["T1"]["n"] == 1
    assert per_tier["T1"]["n_scored"] == 1
    assert per_tier["T1"]["recall@1"] == 1.0
    assert per_tier["T1"]["mrr"] == 1.0
    assert per_tier["T1"]["leakage_count"] == 0

    assert per_tier["T2"]["n_scored"] == 1
    assert per_tier["T2"]["recall@1"] == 0.0
    assert per_tier["T2"]["recall@3"] == 1.0
    assert per_tier["T2"]["mrr"] == 0.5
    # T2's one question labels no section, so the section metrics are null. A
    # 0.000 there would read as a failure on a check that never ran.
    assert per_tier["T2"]["n_section_scored"] == 0
    assert per_tier["T2"]["section_hit_rate"] is None
    assert per_tier["T2"]["section_mrr"] is None

    # T4 is ambiguous and T6 is policy probes. Their rows count in `n` but none
    # of them earns a recall average, so `n_scored` is 0 and the means are 0.0.
    assert per_tier["T4"]["n"] == 2
    assert per_tier["T4"]["n_scored"] == 0
    assert per_tier["T4"]["recall@3"] == 0.0
    assert per_tier["T4"]["mrr"] == 0.0
    assert per_tier["T4"]["section_hit_rate"] is None
    assert per_tier["T4"]["section_mrr"] is None
    assert per_tier["T6"]["n"] == 2
    assert per_tier["T6"]["n_scored"] == 0
    assert per_tier["T6"]["recall@3"] == 0.0


def test_a_group_with_no_policy_probes_reports_a_null_pass_rate() -> None:
    """No probes is not a perfect score and not a failure, it is no measurement."""
    assert harness.policy_summary([]) == {"n": 0, "n_pass": 0, "pass_rate": None}


def test_a_group_with_no_rows_at_all_reports_null_section_metrics() -> None:
    summary = harness.summarise([], k_values=(5,))

    assert summary["n"] == 0
    assert summary["n_section_scored"] == 0
    assert summary["section_hit_rate"] is None
    assert summary["section_mrr"] is None
    # Doc-level averages keep their 0.0, so the two rungs of the ladder stay
    # comparable across runs. Only the section columns went null in this ruling.
    assert summary["recall@5"] == 0.0
    assert summary["mrr"] == 0.0


def test_the_planted_forbidden_hit_is_counted_as_leakage(report: dict[str, Any]) -> None:
    """Marketing was handed a finance chunk; the replay has to catch it."""
    row = row_for(report, "t6-leak")

    assert row["scoring"] == "policy"
    assert row["leakage"] == 1
    assert row["leaked_chunk_ids"] == [f"{FINANCE_DOC}#q4-revenue"]
    assert row["policy_pass"] is False

    assert report["overall"]["leakage_count"] == 1
    assert report["overall"]["n_leaking_questions"] == 1
    assert report["per_tier"]["T6"]["leakage_count"] == 1


def test_a_probe_that_returns_nothing_passes(report: dict[str, Any]) -> None:
    row = row_for(report, "t6-clean")

    assert row["scoring"] == "policy"
    assert row["n_hits"] == 0
    assert row["leakage"] == 0
    assert row["policy_pass"] is True
    # One probe leaked and one did not, so the summary is one pass out of two.
    assert report["policy_probes"] == {"n": 2, "n_pass": 1, "pass_rate": 0.5}


def test_ambiguous_questions_are_scored_on_both_sources_only(report: dict[str, Any]) -> None:
    passed = row_for(report, "t4-both-pass")
    failed = row_for(report, "t4-both-fail")

    assert passed["scoring"] == "both_sources"
    assert passed[f"both_sources@{MAX_K}"] is True
    assert failed[f"both_sources@{MAX_K}"] is False
    # Half a pair is a miss, so this row scores 0.5 recall and no credit here.
    assert failed["recall@3"] == 0.5

    assert report["ambiguous"] == {
        "n": 2,
        "n_pass": 1,
        f"both_sources_recall@{MAX_K}": 0.5,
    }


def test_run_with_the_default_cut_offs(micro_corpus: list[Chunk], policy: PolicyEngine) -> None:
    """Default k is (5, 10), which is wider than any scripted list."""
    retriever = ScriptedRetriever(micro_corpus, SCRIPT)
    report = EvalHarness(ScriptedPipeline(retriever), policy, GOLD).run()

    assert report["k_values"] == [5, 10]
    assert "recall@10" in report["overall"]
    assert report["ambiguous"]["both_sources_recall@10"] == 0.5


def test_report_filename_uses_the_given_timestamp() -> None:
    when = datetime(2026, 7, 29, 13, 5, tzinfo=timezone.utc)
    assert report_io.report_filename("r0", "direct__naive__none", when) == (
        "r0_direct__naive__none_20260729T1305.json"
    )


def test_write_report_lands_where_the_dashboard_reads(
    report: dict[str, Any], tmp_path: Path
) -> None:
    """The results directory is created if missing, and the name carries the stamp."""
    when = datetime(2026, 7, 29, 13, 5, tzinfo=timezone.utc)
    results_dir = tmp_path / "evals" / "results"

    path = report_io.write_report(
        report, "r0", "direct__scripted__none", results_dir, when=when
    )

    assert path == results_dir / "r0_direct__scripted__none_20260729T1305.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["rung"] == "r0"
    assert saved["pipeline_id"] == "direct__scripted__none"
    assert saved["generated_at"] == "2026-07-29T13:05:00Z"
    assert saved["n_questions"] == 6
    assert saved["overall"]["leakage_count"] == 1
