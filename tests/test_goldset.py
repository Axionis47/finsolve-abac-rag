"""Tests for the gold question set and its validator.

Two kinds of test live here. The first reads the real
`evals/gold/gold_set.json` against the real chunks and the real policy, so a
corpus edit that breaks a label fails the suite instead of silently ruining
every score. The second corrupts a copy of the gold set one field at a time and
checks the validator complains.

There is also a wording test for tier T2. Those questions only prove dense
retrieval beats lexical retrieval if they share no content word with the text
they are meant to find, which is easy to break when editing a question.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from finrag.eval.goldset import TIER_COUNTS, GoldSet, GoldSetError, load_gold, validate
from finrag.policy.engine import PolicyEngine

# Probes the plan named by hand. Each must survive any future edit of the set,
# so they are pinned by tag rather than by question wording.
MANDATORY_TAGS = (
    "instantpay-signups", "rto-rpo", "attrition-holiday", "spend-collision",
    "impressions-collision", "email-open-collision", "leads-collision",
    "na-saturation", "software-subscriptions", "influencer-roi",
    "salary-by-dept", "conversion-trend", "pii-salary", "revenue-role-denied",
)  # fmt: skip

MANDATORY_IDS = ("T1-02", "T2-02", "T6-01", "T6-02", "T6-03", "T6-04")


@pytest.fixture(scope="module")
def gold(settings) -> GoldSet:
    return load_gold(settings.gold_path)


@pytest.fixture(scope="module")
def policy(settings) -> PolicyEngine:
    return PolicyEngine.from_yaml(settings.policy_path)


@pytest.fixture(scope="module")
def chunks_path(settings) -> Path:
    return settings.chunks_path


# --------------------------------------------------------------- the real set


def test_real_gold_set_validates(gold, chunks_path, policy):
    validate(gold, chunks_path, policy)


def test_version_size_and_tier_counts(gold):
    counts: dict[str, int] = {}
    ids = []
    for entry in gold.entries:
        counts[entry["tier"]] = counts.get(entry["tier"], 0) + 1
        ids.append(entry["id"])

    assert gold.version == "2.0"
    assert len(gold.entries) == 50
    assert counts == TIER_COUNTS
    assert len(set(ids)) == len(ids), "ids must be unique"


def test_mandatory_probes_are_present(gold):
    all_tags = set()
    all_ids = set()
    for entry in gold.entries:
        all_ids.add(entry["id"])
        for tag in entry["tags"]:
            all_tags.add(tag)

    for tag in MANDATORY_TAGS:
        assert tag in all_tags, f"missing mandatory probe tagged '{tag}'"
    for entry_id in MANDATORY_IDS:
        assert entry_id in all_ids, f"missing mandatory probe '{entry_id}'"


def test_ambiguous_probes(gold):
    ambiguous = []
    for entry in gold.entries:
        if "ambiguous" in entry["tags"]:
            ambiguous.append(entry)

    assert len(ambiguous) == 4
    for entry in ambiguous:
        assert entry["tier"] == "T4"
        assert entry["expected_behavior"] == "retrieve_both"
        assert entry["role"] == "c_level"
        assert len(entry["gold_docs"]) >= 2


def test_t6_negatives_and_their_positive_control_twins(gold):
    behaviors = []
    by_id = {}
    for entry in gold.entries:
        by_id[entry["id"]] = entry
        if entry["tier"] == "T6":
            behaviors.append(entry["expected_behavior"])

    assert len(behaviors) == 8
    assert behaviors.count("answer") == 2, "T6 needs its two positive-control twins"
    assert behaviors.count("refuse_role") + behaviors.count("no_answer") == 6

    # Each twin asks the identical question in a different role, so the only
    # thing that can change the outcome is the policy layer.
    for negative, positive in (("T6-01", "T6-02"), ("T6-03", "T6-04")):
        assert by_id[negative]["question"] == by_id[positive]["question"]
        assert by_id[negative]["role"] != by_id[positive]["role"]


def test_every_entry_is_well_formed_prose(gold):
    for entry in gold.entries:
        if entry["expected_behavior"] in ("answer", "retrieve_both"):
            assert entry["gold_answer"].strip(), f"{entry['id']} has no gold answer"

        for section in entry["gold_sections"]:
            doc_id = section.split("#")[0]
            assert doc_id in entry["gold_docs"], f"{entry['id']}: {section} is not a gold doc"

        for field in ("question", "gold_answer", "notes"):
            assert "—" not in entry[field], f"{entry['id']}: em dash in {field}"


# ------------------------------------------------------- T2 wording discipline

# Words too common to count as a shared idea. Anything outside this list that
# appears in both the question and the target text is real overlap.
STOPWORDS = set(
    """a an the and or but if then than that this these those of in on at to for from by with
    into over under about as is are was were be been being do does did doing have has had having
    how what which who whom whose when where why can could may might must shall should will would
    we our us you your they their them it its he she his her i me my not no nor so such only own
    same too very just now here there all any both each few more most other some again against
    between during before after above below up down out off further once because while until
    upon per within across among along around also come came take took give gave make made say
    said long short set aside one two three four five""".split()
)


def content_words(text: str) -> set[str]:
    """Meaning-carrying words, plus a crude stem of each.

    The stem only has to make `customer` and `customers` collide, so that a
    near miss still counts as overlap.
    """
    found: set[str] = set()
    for raw in re.findall(r"[a-z0-9']+", text.lower()):
        word = raw.strip("'")
        if not word or word.isdigit() or word in STOPWORDS:
            continue
        found.add(word)
        for suffix in ("ies", "es", "s", "ing", "ed"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                found.add(word[: -len(suffix)])
                break
    return found


def gold_section_text(chunks_path: Path, prefixes: list[str]) -> str:
    """Every chunk whose id starts with one of these prefixes, joined up."""
    parts = []
    with open(chunks_path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            for prefix in prefixes:
                if row["chunk_id"].startswith(prefix):
                    parts.append(row["text"])
                    break
    return "\n".join(parts)


def test_t2_questions_share_no_content_word_with_their_target(gold, chunks_path):
    for entry in gold.entries:
        if entry["tier"] != "T2":
            continue
        target = gold_section_text(chunks_path, entry["gold_sections"])
        assert target, f"{entry['id']}: gold sections matched no chunk text"
        shared = content_words(entry["question"]) & content_words(target)
        assert not shared, f"{entry['id']} shares {sorted(shared)} with its gold section"


# ---------------------------------------------------------- corruption probes


def corrupted(gold: GoldSet, index: int, **changes) -> GoldSet:
    """A copy of the gold set with one entry's fields overwritten."""
    entries = copy.deepcopy(gold.entries)
    entries[index].update(changes)
    return GoldSet(version=gold.version, entries=entries)


def test_duplicate_id_fails(gold, chunks_path, policy):
    broken = corrupted(gold, 1, id=gold.entries[0]["id"])
    with pytest.raises(GoldSetError, match="duplicate id"):
        validate(broken, chunks_path, policy)


def test_unknown_field_fails(gold, chunks_path, policy):
    broken = corrupted(gold, 0, difficulty="hard")
    with pytest.raises(GoldSetError, match="unknown field"):
        validate(broken, chunks_path, policy)


def test_missing_field_fails(gold, chunks_path, policy):
    entries = copy.deepcopy(gold.entries)
    del entries[0]["notes"]
    with pytest.raises(GoldSetError, match="missing field"):
        validate(GoldSet(gold.version, entries), chunks_path, policy)


def test_unknown_gold_doc_fails(gold, chunks_path, policy):
    broken = corrupted(gold, 0, gold_docs=["finance/does_not_exist"], gold_sections=[])
    with pytest.raises(GoldSetError, match="is not in the chunks file"):
        validate(broken, chunks_path, policy)


def test_gold_section_that_matches_nothing_fails(gold, chunks_path, policy):
    doc_id = gold.entries[0]["gold_docs"][0]
    broken = corrupted(gold, 0, gold_sections=[f"{doc_id}#no-such-heading"])
    with pytest.raises(GoldSetError, match="matches no chunk id"):
        validate(broken, chunks_path, policy)


def test_role_that_cannot_see_its_gold_doc_fails(gold, chunks_path, policy):
    # T1-01 is a marketing question. Asked as engineering it is unanswerable,
    # and outside T6 that is a labelling mistake rather than a probe.
    broken = corrupted(gold, 0, role="engineering", applicable_roles=["engineering"])
    with pytest.raises(GoldSetError, match="may not retrieve gold doc"):
        validate(broken, chunks_path, policy)


def test_ambiguous_entry_with_one_gold_doc_fails(gold, chunks_path, policy):
    index = None
    for position, entry in enumerate(gold.entries):
        if "ambiguous" in entry["tags"]:
            index = position
            break
    assert index is not None

    keep = gold.entries[index]["gold_docs"][:1]
    keep_sections = []
    for section in gold.entries[index]["gold_sections"]:
        if section.startswith(keep[0]):
            keep_sections.append(section)

    broken = corrupted(gold, index, gold_docs=keep, gold_sections=keep_sections)
    with pytest.raises(GoldSetError, match="at least 2 gold docs"):
        validate(broken, chunks_path, policy)


def test_bad_tier_count_fails(gold, chunks_path, policy):
    broken = corrupted(gold, 0, tier="T5")
    with pytest.raises(GoldSetError, match="expected 8 questions"):
        validate(broken, chunks_path, policy)


def test_bad_expected_behavior_fails(gold, chunks_path, policy):
    broken = corrupted(gold, 0, expected_behavior="maybe")
    with pytest.raises(GoldSetError, match="expected_behavior"):
        validate(broken, chunks_path, policy)


def test_load_gold_rejects_a_bare_list(tmp_path):
    path = tmp_path / "gold.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(GoldSetError):
        load_gold(path)
