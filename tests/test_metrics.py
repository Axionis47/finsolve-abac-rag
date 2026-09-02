"""Metric math on hand-built result lists, every expected value worked out by hand.

`finrag.eval.metrics` is the scoring layer between a retriever and the eval
harness. Nothing here retrieves anything: the ranked lists are written out
below, so each expected number can be checked by reading the list rather than
by trusting the code under test.

Two rules carry most of the weight and both get their own tests:

  * scoring folds chunks down to documents, so several chunks of one document
    prove that document once, in the numerator and in the denominator;
  * an empty gold label scores 0.0. A question with nothing to find earns no
    free credit.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from finrag.eval import metrics
from finrag.models import Hit
from finrag.policy.engine import PolicyEngine

DOC_A = "finance/quarterly_financial_report"
DOC_B = "marketing/market_report_q1_2024"
DOC_C = "marketing/market_report_q3_2024"
DOC_D = "hr/aggregates"


def make_hit(chunk_id: str, rank: int) -> Hit:
    """A hit carrying only the fields the metrics read: id, doc and rank."""
    return Hit(
        chunk_id=chunk_id,
        doc_id=chunk_id.split("#")[0],
        section_slug=chunk_id.split("#")[-1],
        rank=rank,
        score=1.0 / rank,
        retriever="test",
        owner_dept="finance",
        doc_type="md",
        sensitivity="internal",
    )


def make_hits(chunk_ids: list[str]) -> list[Hit]:
    """Turn a list of chunk ids into a ranked hit list, ranks from 1."""
    hits = []
    rank = 1
    for chunk_id in chunk_ids:
        hits.append(make_hit(chunk_id, rank))
        rank += 1
    return hits


# The list every test below reads from. Six chunks, four documents, with
# document A appearing twice at the top and document B split across ranks 3
# and 6, so the "count a document once" rule has something to bite on.
#
#   rank 1  A#one     rank 4  C#one
#   rank 2  A#two     rank 5  D#one
#   rank 3  B#one     rank 6  B#two
RANKED = make_hits(
    [
        f"{DOC_A}#one",
        f"{DOC_A}#two",
        f"{DOC_B}#one",
        f"{DOC_C}#one",
        f"{DOC_D}#one",
        f"{DOC_B}#two",
    ]
)


def test_retrieved_docs_keeps_rank_order_and_drops_repeats() -> None:
    assert metrics.retrieved_docs(RANKED) == [DOC_A, DOC_B, DOC_C, DOC_D]


def test_retrieved_docs_cuts_at_k_hits_not_k_documents() -> None:
    """k counts hits, so the top 3 hits yield only 2 documents."""
    assert metrics.retrieved_docs(RANKED, 1) == [DOC_A]
    assert metrics.retrieved_docs(RANKED, 3) == [DOC_A, DOC_B]
    assert metrics.retrieved_docs(RANKED, 100) == [DOC_A, DOC_B, DOC_C, DOC_D]


def test_recall_at_k() -> None:
    """Gold is B and D. B arrives at hit 3, D only at hit 5."""
    gold = [DOC_B, DOC_D]
    assert metrics.recall_at_k(RANKED, gold, 2) == 0.0
    assert metrics.recall_at_k(RANKED, gold, 3) == 0.5
    assert metrics.recall_at_k(RANKED, gold, 5) == 1.0


def test_precision_at_k_divides_by_documents_returned_not_by_k() -> None:
    """The top 3 hits return 2 documents, one of them gold, so 1/2."""
    gold = [DOC_B, DOC_D]
    assert metrics.precision_at_k(RANKED, gold, 3) == 0.5
    # Top 5 hits return 4 documents, 2 of them gold.
    assert metrics.precision_at_k(RANKED, gold, 5) == 0.5
    # Top 2 hits are both document A, which is not gold.
    assert metrics.precision_at_k(RANKED, gold, 2) == 0.0


def test_precision_counts_one_document_once_however_many_chunks_it_won() -> None:
    """Three chunks of one gold document score 1.0, not 1/3."""
    hits = make_hits([f"{DOC_A}#one", f"{DOC_A}#two", f"{DOC_A}#three"])
    assert metrics.precision_at_k(hits, [DOC_A], 3) == 1.0
    assert metrics.recall_at_k(hits, [DOC_A], 3) == 1.0


def test_f1_at_k_is_the_harmonic_mean() -> None:
    gold = [DOC_B, DOC_D]
    # recall@5 = 1.0, precision@5 = 0.5 -> 2 * 1 * 0.5 / 1.5
    assert metrics.f1_at_k(RANKED, gold, 5) == pytest.approx(2 / 3)
    # recall@3 = precision@3 = 0.5, so the harmonic mean is 0.5 too.
    assert metrics.f1_at_k(RANKED, gold, 3) == 0.5


def test_f1_is_zero_when_both_sides_are_zero() -> None:
    """No division by zero when nothing gold came back."""
    assert metrics.f1_at_k(RANKED, ["nowhere/doc"], 5) == 0.0


def test_reciprocal_rank_counts_documents_over_the_whole_list() -> None:
    """First gold document, by document position, with no top-k cut."""
    assert metrics.reciprocal_rank(RANKED, [DOC_A]) == 1.0
    assert metrics.reciprocal_rank(RANKED, [DOC_B]) == 0.5
    assert metrics.reciprocal_rank(RANKED, [DOC_D]) == 0.25
    # Two gold documents: the earlier one sets the score.
    assert metrics.reciprocal_rank(RANKED, [DOC_D, DOC_B]) == 0.5


def test_reciprocal_rank_is_zero_when_nothing_gold_came_back() -> None:
    assert metrics.reciprocal_rank(RANKED, ["nowhere/doc"]) == 0.0


def test_empty_gold_scores_zero_everywhere() -> None:
    """An empty label is a missing label, so it earns nothing."""
    assert metrics.recall_at_k(RANKED, [], 5) == 0.0
    assert metrics.precision_at_k(RANKED, [], 5) == 0.0
    assert metrics.f1_at_k(RANKED, [], 5) == 0.0
    assert metrics.reciprocal_rank(RANKED, []) == 0.0
    assert metrics.both_sources_hit(RANKED, [], 5) is False
    assert metrics.section_recall_at_k(RANKED, [], 5) == 0.0
    assert metrics.all_sections_hit_at_k(RANKED, [], 5) is False


def test_empty_hit_list_scores_zero_everywhere() -> None:
    gold = [DOC_A, DOC_B]
    assert metrics.retrieved_docs([], 5) == []
    assert metrics.recall_at_k([], gold, 5) == 0.0
    assert metrics.precision_at_k([], gold, 5) == 0.0
    assert metrics.f1_at_k([], gold, 5) == 0.0
    assert metrics.reciprocal_rank([], gold) == 0.0
    assert metrics.section_hit([], [f"{DOC_A}#one"]) is False
    assert metrics.section_rr([], [f"{DOC_A}#one"]) == 0.0
    assert metrics.section_recall_at_k([], [f"{DOC_A}#one"], 5) == 0.0
    assert metrics.all_sections_hit_at_k([], [f"{DOC_A}#one"], 5) is False
    assert metrics.both_sources_hit([], gold, 5) is False


def test_section_matches_accepts_the_three_forms_and_nothing_else() -> None:
    """Exact id, part suffix, heading subtree. A shared word start is not a match."""
    gold = f"{DOC_A}#q4-revenue"

    assert metrics.section_matches(gold, gold) is True
    assert metrics.section_matches(f"{gold}/p2", gold) is True
    assert metrics.section_matches(f"{gold}--north-america", gold) is True
    # The sibling-slug hazard: a shorter gold prefix must not claim a longer,
    # different section that happens to start with the same letters.
    assert metrics.section_matches(f"{DOC_A}#q4-revenue", f"{DOC_A}#q4-rev") is False
    assert metrics.section_matches(f"{DOC_A}#q4-risks", gold) is False


def test_section_hit_matches_a_chunk_id_prefix() -> None:
    """Gold sections are prefixes, so a split section still matches."""
    hits = make_hits([f"{DOC_A}#q4-revenue/p2"])
    assert metrics.section_hit(hits, [f"{DOC_A}#q4-revenue"]) is True


def test_section_hit_matches_a_heading_subtree() -> None:
    hits = make_hits([f"{DOC_A}#q4-revenue--north-america"])
    assert metrics.section_hit(hits, [f"{DOC_A}#q4-revenue"]) is True


def test_section_hit_matches_an_exact_chunk_id() -> None:
    assert metrics.section_hit(RANKED, [f"{DOC_B}#two"]) is True


def test_section_hit_ignores_a_sibling_slug_that_shares_a_word_start() -> None:
    """`doc#q4-rev` naming `doc#q4-revenue` would be a false pass, so it does not."""
    hits = make_hits([f"{DOC_A}#q4-revenue"])
    assert metrics.section_hit(hits, [f"{DOC_A}#q4-rev"]) is False


def test_section_hit_is_false_for_a_section_nobody_returned() -> None:
    assert metrics.section_hit(RANKED, [f"{DOC_A}#q4-risks"]) is False


def test_section_hit_is_false_when_the_question_has_no_gold_sections() -> None:
    """Nothing to hit is not a hit, and the harness leaves these out of the rate."""
    assert metrics.section_hit(RANKED, []) is False


def test_section_rr_counts_hit_positions_not_document_positions() -> None:
    """The two A chunks at ranks 1 and 2 both take up a slot before B at 3.

    This is the difference from `reciprocal_rank`, which folds chunks down to
    documents first and would score B at position 2.
    """
    assert metrics.section_rr(RANKED, [f"{DOC_A}#one"]) == 1.0
    assert metrics.section_rr(RANKED, [f"{DOC_A}#two"]) == 0.5
    assert metrics.section_rr(RANKED, [f"{DOC_B}#one"]) == pytest.approx(1 / 3)
    assert metrics.section_rr(RANKED, [f"{DOC_B}#two"]) == pytest.approx(1 / 6)
    # `reciprocal_rank` on the same document scores it at document position 2.
    assert metrics.reciprocal_rank(RANKED, [DOC_B]) == 0.5


def test_section_rr_takes_the_earliest_of_several_gold_sections() -> None:
    gold = [f"{DOC_D}#one", f"{DOC_C}#one"]
    assert metrics.section_rr(RANKED, gold) == 0.25


def test_section_rr_is_zero_when_nothing_matched_or_nothing_was_labelled() -> None:
    assert metrics.section_rr(RANKED, [f"{DOC_A}#q4-risks"]) == 0.0
    assert metrics.section_rr(RANKED, []) == 0.0
    assert metrics.section_rr([], [f"{DOC_A}#one"]) == 0.0


def test_section_rr_uses_the_same_tight_match_as_section_hit() -> None:
    hits = make_hits([f"{DOC_A}#q4-revenue"])
    assert metrics.section_rr(hits, [f"{DOC_A}#q4-rev"]) == 0.0
    assert metrics.section_rr(hits, [f"{DOC_A}#q4-revenue"]) == 1.0


def test_section_coverage_requires_every_labelled_section() -> None:
    gold = [f"{DOC_A}#one", f"{DOC_B}#one", f"{DOC_D}#one"]
    assert metrics.section_recall_at_k(RANKED, gold, 3) == pytest.approx(2 / 3)
    assert metrics.all_sections_hit_at_k(RANKED, gold, 3) is False
    assert metrics.section_recall_at_k(RANKED, gold, 5) == 1.0
    assert metrics.all_sections_hit_at_k(RANKED, gold, 5) is True


def test_both_sources_hit_needs_every_gold_document() -> None:
    """The ambiguous questions want both conflicting sources or nothing."""
    gold = [DOC_B, DOC_C]
    assert metrics.both_sources_hit(RANKED, gold, 4) is True
    # At k=3 only B is back, and one of two is a failure, not a half.
    assert metrics.both_sources_hit(RANKED, gold, 3) is False


def test_both_sources_hit_with_three_gold_documents() -> None:
    assert metrics.both_sources_hit(RANKED, [DOC_A, DOC_B, DOC_C], 4) is True
    assert metrics.both_sources_hit(RANKED, [DOC_A, DOC_B, "nowhere/doc"], 6) is False


def test_score_question_returns_every_metric_for_both_cut_offs() -> None:
    gold_entry = {
        "gold_docs": [DOC_B, DOC_D],
        "gold_sections": [f"{DOC_B}#one"],
    }

    scores = metrics.score_question(RANKED, gold_entry, k_values=(3, 5))

    assert scores == {
        "recall@3": 0.5,
        "precision@3": 0.5,
        "f1@3": 0.5,
        "section_recall@3": 1.0,
        "all_sections@3": True,
        "recall@5": 1.0,
        "precision@5": 0.5,
        "f1@5": pytest.approx(2 / 3),
        "section_recall@5": 1.0,
        "all_sections@5": True,
        "rr": 0.5,
        "section_hit": True,
        "section_rr": pytest.approx(1 / 3),
        "has_gold_sections": True,
    }


def test_score_question_flags_a_question_with_no_gold_sections() -> None:
    """`has_gold_sections` is what keeps these out of the section metrics."""
    scores = metrics.score_question(RANKED, {"gold_docs": [DOC_B]}, k_values=(5,))
    assert scores["has_gold_sections"] is False
    assert scores["section_hit"] is False
    assert scores["section_rr"] == 0.0


@pytest.fixture(scope="module")
def engine(policy_path: Path) -> PolicyEngine:
    """The shipped policy, used only by the two leakage tests."""
    return PolicyEngine.from_yaml(policy_path)


def test_leakage_is_zero_on_a_clean_result_list(engine: PolicyEngine) -> None:
    hits = [
        Hit(
            chunk_id=f"{DOC_B}#one",
            doc_id=DOC_B,
            section_slug="one",
            rank=1,
            score=1.0,
            retriever="test",
            owner_dept="marketing",
            doc_type="md",
            sensitivity="internal",
        )
    ]
    assert metrics.leaked_chunk_ids(engine, "marketing", hits) == []
    assert metrics.leakage_count(engine, "marketing", hits) == 0


def test_leakage_names_the_chunks_the_role_may_not_see(engine: PolicyEngine) -> None:
    """Replaying through the policy catches a hit no `allowed_roles` stamp admits."""
    forbidden = Hit(
        chunk_id="hr/hr_data#row-FINEMP1000",
        doc_id="hr/hr_data",
        section_slug="row-FINEMP1000",
        rank=1,
        score=1.0,
        retriever="test",
        owner_dept="hr",
        doc_type="csv_row",
        sensitivity="restricted",
        # A wrong stamp must not save it: the replay asks the policy, not this.
        allowed_roles=("marketing",),
    )

    assert metrics.leaked_chunk_ids(engine, "marketing", [forbidden]) == [
        "hr/hr_data#row-FINEMP1000"
    ]
    assert metrics.leakage_count(engine, "marketing", [forbidden]) == 1
