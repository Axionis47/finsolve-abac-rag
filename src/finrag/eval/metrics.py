"""Scoring one result list against one gold question.

This sits between retrieval and the eval harness. A retriever hands back a
ranked list of `Hit`s; a gold entry says which documents and sections should
have come back. The functions here turn that pair into numbers.

Two ideas run through the whole file:

  * Scoring is at DOCUMENT level first. We fold the ranked chunks down to the
    documents they came from, in rank order, each document counted once. That
    keeps the numbers stable when we retune the chunker, which would otherwise
    change every score for no real reason.
  * Section level is the finer check, and after role masking it is the one that
    separates retrievers: did a returned chunk sit inside one of the gold
    sections, and how high did it rank? Gold sections are stored as chunk-id
    prefixes, so `finance/report#q4-revenue` still matches after a section is
    split into `.../p2`.

Everything except the leakage helpers is a pure function: same inputs, same
answer, no clocks, no files, no policy lookups.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from finrag.models import Hit
from finrag.policy.engine import PolicyEngine


def retrieved_docs(hits: Sequence[Hit], k: int | None = None) -> list[str]:
    """Document ids behind the top `k` hits, best first, each id listed once.

    `k` counts hits, not documents, because that is what a retriever actually
    returned. `k=None` uses the whole list.
    """
    if k is None:
        chosen = list(hits)
    else:
        chosen = list(hits)[:k]

    docs: list[str] = []
    for hit in chosen:
        if hit.doc_id not in docs:
            docs.append(hit.doc_id)
    return docs


def _count_gold_found(docs: Sequence[str], gold_docs: Sequence[str]) -> int:
    """How many distinct gold documents appear in `docs`."""
    gold = set(gold_docs)
    found = 0
    for doc in docs:
        if doc in gold:
            found += 1
    return found


def recall_at_k(hits: Sequence[Hit], gold_docs: Sequence[str], k: int) -> float:
    """Share of the gold documents that the top `k` hits reached.

    No gold documents means there is nothing to find, so the score is 0.0. We
    refuse to hand out free credit for an empty label.
    """
    gold = set(gold_docs)
    if not gold:
        return 0.0
    docs = retrieved_docs(hits, k)
    return _count_gold_found(docs, gold) / len(gold)


def precision_at_k(hits: Sequence[Hit], gold_docs: Sequence[str], k: int) -> float:
    """Share of the documents returned in the top `k` hits that were gold.

    The denominator is the number of distinct documents returned, not `k`,
    because several chunks of one document only prove that document once.
    """
    docs = retrieved_docs(hits, k)
    if not docs:
        return 0.0
    return _count_gold_found(docs, gold_docs) / len(docs)


def f1_at_k(hits: Sequence[Hit], gold_docs: Sequence[str], k: int) -> float:
    """Harmonic mean of recall@k and precision@k; 0.0 when both are 0."""
    recall = recall_at_k(hits, gold_docs, k)
    precision = precision_at_k(hits, gold_docs, k)
    if recall + precision == 0.0:
        return 0.0
    return 2 * recall * precision / (recall + precision)


def reciprocal_rank(hits: Sequence[Hit], gold_docs: Sequence[str]) -> float:
    """1 / position of the first gold document in the ranked document list.

    Positions count documents, start at 1, and run over the whole list rather
    than a top-k cut, so a late but correct answer still scores something.
    """
    gold = set(gold_docs)
    if not gold:
        return 0.0

    docs = retrieved_docs(hits)
    position = 1
    for doc in docs:
        if doc in gold:
            return 1.0 / position
        position += 1
    return 0.0


def section_matches(chunk_id: str, prefix: str) -> bool:
    """True when `chunk_id` is the gold section named by `prefix`, or inside it.

    A plain `startswith` would be wrong. Chunk ids are built from slugs, and
    slugs of sibling headings share leading text: the prefix `doc#q4-rev` would
    silently claim `doc#q4-revenue`, which is a different section. So we accept
    a match only at a real boundary in the id:

      * the ids are equal, the ordinary case;
      * the id continues with "/", which is the part suffix a long section gets
        when it is split (`doc#q4-revenue/p2`);
      * the id continues with "--", which is how the chunker joins a heading to
        its subheading (`doc#q4-revenue--north-america`), so naming a parent
        heading still covers the whole subtree under it.

    Anything else is a different section that merely starts with the same
    letters, and it does not count.
    """
    if chunk_id == prefix:
        return True
    if chunk_id.startswith(prefix + "/"):
        return True
    if chunk_id.startswith(prefix + "--"):
        return True
    return False


def section_hit(hits: Sequence[Hit], gold_sections: Sequence[str]) -> bool:
    """True when any returned chunk sits inside one of the gold sections."""
    if not gold_sections:
        return False
    for hit in hits:
        for prefix in gold_sections:
            if section_matches(hit.chunk_id, prefix):
                return True
    return False


def section_rr(hits: Sequence[Hit], gold_sections: Sequence[str]) -> float:
    """1 / the list position of the first hit inside a gold section.

    Positions count HITS, not documents, and start at 1. That is the difference
    from `reciprocal_rank` above, which folds chunks down to documents first.
    Here we want to know how far a reader would have to scroll before reaching
    the right passage, so every returned chunk takes up a slot.

    This is the headline metric from R0 onward. Once the role mask is applied
    most roles can see only a handful of documents, so document-level recall is
    nearly free and tells us little; section level is where a better retriever
    can still show a difference.
    """
    if not gold_sections:
        return 0.0

    position = 1
    for hit in hits:
        for prefix in gold_sections:
            if section_matches(hit.chunk_id, prefix):
                return 1.0 / position
        position += 1
    return 0.0


def matched_gold_sections(
    hits: Sequence[Hit], gold_sections: Sequence[str], k: int
) -> set[str]:
    """Distinct labelled sections represented by the first `k` hits."""
    matched: set[str] = set()
    for hit in list(hits)[:k]:
        for prefix in gold_sections:
            if section_matches(hit.chunk_id, prefix):
                matched.add(prefix)
    return matched


def section_recall_at_k(
    hits: Sequence[Hit], gold_sections: Sequence[str], k: int
) -> float:
    """Share of all required source sections represented in the first `k` hits."""
    gold = set(gold_sections)
    if not gold:
        return 0.0
    return len(matched_gold_sections(hits, gold_sections, k)) / len(gold)


def all_sections_hit_at_k(
    hits: Sequence[Hit], gold_sections: Sequence[str], k: int
) -> bool:
    """Whether the first `k` hits contain every required source section."""
    gold = set(gold_sections)
    if not gold:
        return False
    return matched_gold_sections(hits, gold_sections, k) == gold


def both_sources_hit(hits: Sequence[Hit], gold_docs: Sequence[str], k: int) -> bool:
    """True when EVERY gold document appears in the top `k` hits.

    This is how the ambiguous questions are scored. They ask about a value the
    corpus states two different ways, so the only right behaviour is to surface
    both conflicting sources. Finding one of the two is a failure, not a half.
    """
    if not gold_docs:
        return False
    docs = set(retrieved_docs(hits, k))
    for gold_doc in gold_docs:
        if gold_doc not in docs:
            return False
    return True


def leaked_chunk_ids(policy: PolicyEngine, role: str, hits: Sequence[Hit]) -> list[str]:
    """Chunk ids in `hits` that the policy would refuse this role.

    The check replays every hit through the policy engine rather than trusting
    the `allowed_roles` stamped at ingest, so a stale or wrong stamp is caught
    instead of hidden.
    """
    ids: list[str] = []
    for violation in policy.replay(role, hits):
        ids.append(violation.chunk_id)
    return ids


def leakage_count(policy: PolicyEngine, role: str, hits: Sequence[Hit]) -> int:
    """How many returned chunks the role was not allowed to see. Target: 0."""
    return len(leaked_chunk_ids(policy, role, hits))


def score_question(
    hits: Sequence[Hit],
    gold_entry: dict[str, Any],
    k_values: Sequence[int] = (5, 10),
) -> dict[str, Any]:
    """Every retrieval number for one question, as a flat dict.

    Pure: it reads only the hits and the gold entry. Leakage needs the policy
    engine, so the harness adds it separately.
    """
    gold_docs = gold_entry.get("gold_docs") or []
    gold_sections = gold_entry.get("gold_sections") or []

    scores: dict[str, Any] = {}
    for k in k_values:
        scores[f"recall@{k}"] = recall_at_k(hits, gold_docs, k)
        scores[f"precision@{k}"] = precision_at_k(hits, gold_docs, k)
        scores[f"f1@{k}"] = f1_at_k(hits, gold_docs, k)
        scores[f"section_recall@{k}"] = section_recall_at_k(hits, gold_sections, k)
        scores[f"all_sections@{k}"] = all_sections_hit_at_k(hits, gold_sections, k)

    scores["rr"] = reciprocal_rank(hits, gold_docs)
    scores["section_hit"] = section_hit(hits, gold_sections)
    scores["section_rr"] = section_rr(hits, gold_sections)
    scores["has_gold_sections"] = bool(gold_sections)
    return scores
