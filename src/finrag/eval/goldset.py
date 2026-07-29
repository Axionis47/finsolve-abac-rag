"""Reading the gold question set and checking that it still describes reality.

The gold set at `evals/gold/gold_set.json` is the scoreboard's question paper:
50 tiered questions, each naming the documents and sections that should come
back and the answer a person would accept. This module does two jobs.

  load_gold(path)                  read the file into a GoldSet
  validate(gold, chunks, policy)   shout if the gold set has drifted

Validation matters because the gold set points at the corpus by name. Rename a
heading, drop a document, or mistype a role and every score quietly becomes
meaningless instead of failing. So we check the labels against the real
`artifacts/chunks.jsonl` and the real policy engine, and raise one error that
lists everything wrong. Entries stay plain dicts because the eval harness
consumes dicts, and a dataclass here would only mean converting back and forth.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from finrag.models import ROLES, ResourceAttrs
from finrag.policy.engine import RETRIEVE_ACTION, PolicyEngine

# Every field a gold entry must carry. Anything else is a typo or a private
# note that would never be read, so both are errors.
REQUIRED_FIELDS: tuple[str, ...] = (
    "id", "tier", "question", "role", "applicable_roles", "expected_behavior",
    "gold_docs", "gold_sections", "distractor_docs", "gold_answer",
    "answer_source", "tags", "notes",
)  # fmt: skip

EXPECTED_BEHAVIORS: tuple[str, ...] = ("answer", "no_answer", "refuse_role", "retrieve_both")

# The tier composition the plan fixed. A gold set that no longer matches it is
# not the gold set the metrics were designed around.
TIER_COUNTS: dict[str, int] = {"T1": 8, "T2": 8, "T3": 10, "T4": 10, "T5": 6, "T6": 8}

# Questions tagged this way are scored only on returning BOTH conflicting
# sources, so they need at least two gold documents to be scorable at all.
AMBIGUOUS_TAG = "ambiguous"

# T6 is the ABAC negatives tier. Its whole point is asking questions in a role
# that must not reach the answer, so the role-visibility check skips it.
POLICY_TIER = "T6"


class GoldSetError(ValueError):
    """The gold set is malformed or no longer matches the corpus or policy."""


@dataclass(frozen=True)
class GoldSet:
    """One gold file: its version string and its questions as plain dicts."""

    version: str
    entries: list[dict[str, Any]]

    def __len__(self) -> int:
        return len(self.entries)


def load_gold(path: str | Path) -> GoldSet:
    """Read a gold file of the form {"version": ..., "questions": [...]}."""
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)

    if not isinstance(document, dict):
        raise GoldSetError(f"{path}: expected a JSON object with 'version' and 'questions'")

    questions = document.get("questions")
    if not isinstance(questions, list):
        raise GoldSetError(f"{path}: 'questions' must be a list")

    return GoldSet(version=str(document.get("version", "")), entries=list(questions))


@dataclass(frozen=True)
class _CorpusFacts:
    """What the validator needs from the built chunks.

    `resources_by_doc` holds one ResourceAttrs per distinct kind of chunk in a
    document, which is what the policy engine decides on.
    """

    chunk_ids: list[str]
    doc_ids: set[str]
    resources_by_doc: dict[str, list[ResourceAttrs]]


def _read_corpus_facts(chunks_path: str | Path) -> _CorpusFacts:
    """Load chunks.jsonl and reduce it to ids plus one resource per doc shape."""
    chunk_ids: list[str] = []
    doc_ids: set[str] = set()
    resources_by_doc: dict[str, list[ResourceAttrs]] = {}

    with open(chunks_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            chunk_ids.append(row["chunk_id"])
            doc_ids.add(row["doc_id"])

            resource = ResourceAttrs(
                owner_dept=row["owner_dept"],
                doc_type=row["doc_type"],
                sensitivity=row["sensitivity"],
            )
            known = resources_by_doc.setdefault(row["doc_id"], [])
            if resource not in known:
                known.append(resource)

    return _CorpusFacts(chunk_ids=chunk_ids, doc_ids=doc_ids, resources_by_doc=resources_by_doc)


def _matches_a_chunk(prefix: str, chunk_ids: Sequence[str]) -> bool:
    """True when at least one chunk id starts with this gold section prefix."""
    for chunk_id in chunk_ids:
        if chunk_id.startswith(prefix):
            return True
    return False


def _role_can_see(policy: PolicyEngine, role: str, resources: Sequence[ResourceAttrs]) -> bool:
    """True when the role may retrieve at least one chunk shape of a document."""
    for resource in resources:
        if policy.decide(role, RETRIEVE_ACTION, resource).permitted:
            return True
    return False


def _check_shape(entry: dict[str, Any], where: str, problems: list[str]) -> None:
    """Field names, role names and the expected behaviour vocabulary."""
    for field in REQUIRED_FIELDS:
        if field not in entry:
            problems.append(f"{where}: missing field '{field}'")
    for field in entry:
        if field not in REQUIRED_FIELDS:
            problems.append(f"{where}: unknown field '{field}'")

    tier = entry.get("tier")
    if tier not in TIER_COUNTS:
        problems.append(f"{where}: tier '{tier}' is not one of {sorted(TIER_COUNTS)}")

    behavior = entry.get("expected_behavior")
    if behavior not in EXPECTED_BEHAVIORS:
        problems.append(f"{where}: expected_behavior '{behavior}' is not known")

    if entry.get("role") not in ROLES:
        problems.append(f"{where}: role '{entry.get('role')}' is not one of {ROLES}")
    for role in entry.get("applicable_roles") or []:
        if role not in ROLES:
            problems.append(f"{where}: applicable_role '{role}' is not one of {ROLES}")

    if not entry.get("question"):
        problems.append(f"{where}: question is empty")


def _check_labels(
    entry: dict[str, Any], where: str, facts: _CorpusFacts, problems: list[str]
) -> None:
    """Gold documents exist and gold sections still name real chunks."""
    for doc_id in entry.get("gold_docs") or []:
        if doc_id not in facts.doc_ids:
            problems.append(f"{where}: gold doc '{doc_id}' is not in the chunks file")
    for doc_id in entry.get("distractor_docs") or []:
        if doc_id not in facts.doc_ids:
            problems.append(f"{where}: distractor doc '{doc_id}' is not in the chunks file")

    # answer_source is checked the same way as gold_sections: both are chunk-id
    # prefixes, and a typo in either points a later reader at nothing.
    for prefix in entry.get("gold_sections") or []:
        if not _matches_a_chunk(prefix, facts.chunk_ids):
            problems.append(f"{where}: gold section '{prefix}' matches no chunk id")
    for prefix in entry.get("answer_source") or []:
        if not _matches_a_chunk(prefix, facts.chunk_ids):
            problems.append(f"{where}: answer_source '{prefix}' matches no chunk id")


def _check_role_access(
    entry: dict[str, Any], where: str, facts: _CorpusFacts, policy: PolicyEngine, problems: list[str]
) -> None:
    """The asking role must be able to reach every gold document, except in T6."""
    if entry.get("tier") == POLICY_TIER:
        return

    role = entry.get("role")
    if role not in ROLES:
        return  # already reported by _check_shape

    for doc_id in entry.get("gold_docs") or []:
        resources = facts.resources_by_doc.get(doc_id)
        if not resources:
            continue  # already reported by _check_labels
        if not _role_can_see(policy, role, resources):
            problems.append(f"{where}: role '{role}' may not retrieve gold doc '{doc_id}'")


def _check_ambiguous(entry: dict[str, Any], where: str, problems: list[str]) -> None:
    """Ambiguous probes are scored on both sources, so they need two of them."""
    behavior = entry.get("expected_behavior")
    is_ambiguous = AMBIGUOUS_TAG in (entry.get("tags") or [])
    wants_both = behavior == "retrieve_both"

    if is_ambiguous and not wants_both:
        problems.append(f"{where}: tagged '{AMBIGUOUS_TAG}' but expected_behavior is '{behavior}'")
    if wants_both and not is_ambiguous:
        problems.append(f"{where}: expects both sources but is not tagged '{AMBIGUOUS_TAG}'")
    if is_ambiguous and len(entry.get("gold_docs") or []) < 2:
        problems.append(f"{where}: ambiguous entries need at least 2 gold docs")


def _check_tier_counts(entries: Sequence[dict[str, Any]], problems: list[str]) -> None:
    """The tier composition must still be 8/8/10/10/6/8."""
    counts: dict[str, int] = {}
    for entry in entries:
        tier = str(entry.get("tier"))
        counts[tier] = counts.get(tier, 0) + 1

    for tier in TIER_COUNTS:
        found = counts.get(tier, 0)
        if found != TIER_COUNTS[tier]:
            problems.append(f"tier {tier}: expected {TIER_COUNTS[tier]} questions, found {found}")


def validate(gold: GoldSet, chunks_path: str | Path, policy: PolicyEngine) -> None:
    """Check the gold set against the built chunks and the policy engine.

    Raises `GoldSetError` listing every problem found, so one run tells you
    everything to fix instead of one thing at a time. Returns None when clean.
    """
    facts = _read_corpus_facts(chunks_path)
    problems: list[str] = []
    seen_ids: set[str] = set()
    for position, entry in enumerate(gold.entries):
        where = f"question {entry.get('id') or position}"

        entry_id = entry.get("id")
        if entry_id in seen_ids:
            problems.append(f"{where}: duplicate id")
        seen_ids.add(entry_id)

        _check_shape(entry, where, problems)
        _check_labels(entry, where, facts, problems)
        _check_role_access(entry, where, facts, policy, problems)
        _check_ambiguous(entry, where, problems)

    _check_tier_counts(gold.entries, problems)

    if problems:
        raise GoldSetError("gold set is invalid:\n  " + "\n  ".join(problems))
