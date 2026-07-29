"""Contract tests: the shapes everything else is built on."""

from __future__ import annotations

import dataclasses
import json

import pytest

from finrag.models import ROLES, Chunk, Decision, Hit, ResourceAttrs
from finrag.retrieval.base import visible_ids


def test_roles_are_the_six_locked_roles() -> None:
    assert ROLES == (
        "finance",
        "marketing",
        "hr",
        "engineering",
        "c_level",
        "employee",
    )


def test_chunk_fields_match_the_contract() -> None:
    names = [f.name for f in dataclasses.fields(Chunk)]
    assert names == [
        "chunk_id",
        "doc_id",
        "section_path",
        "section_slug",
        "text",
        "owner_dept",
        "doc_type",
        "sensitivity",
        "allowed_roles",
        "quarter",
        "payload",
    ]


def test_hit_fields_match_the_contract() -> None:
    names = [f.name for f in dataclasses.fields(Hit)]
    assert names == [
        "chunk_id",
        "doc_id",
        "section_slug",
        "rank",
        "score",
        "retriever",
        "owner_dept",
        "doc_type",
        "sensitivity",
        "allowed_roles",
        "text",
        "debug",
    ]


def test_chunk_is_frozen(micro_corpus: list[Chunk]) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        micro_corpus[0].text = "rewritten"  # type: ignore[misc]


def test_chunk_json_round_trip(micro_corpus: list[Chunk]) -> None:
    for chunk in micro_corpus:
        line = chunk.to_json()
        assert "\n" not in line
        assert Chunk.from_json(line) == chunk


def test_chunk_round_trip_keeps_allowed_roles_a_tuple(micro_corpus: list[Chunk]) -> None:
    restored = Chunk.from_json(micro_corpus[0].to_json())
    assert isinstance(restored.allowed_roles, tuple)
    # Confidential finance markdown goes to c_level only, not even to finance.
    # test_policy.py asserts that against the real engine.
    assert restored.allowed_roles == ("c_level",)


def test_chunk_round_trip_keeps_payload_and_quarter(micro_corpus: list[Chunk]) -> None:
    row = next(c for c in micro_corpus if c.doc_type == "csv_row")
    restored = Chunk.from_json(row.to_json())
    assert restored.payload == row.payload
    assert restored.quarter is None

    quarterly = next(c for c in micro_corpus if c.quarter == "Q3")
    assert Chunk.from_json(quarterly.to_json()).quarter == "Q3"


def test_chunk_defaults_are_not_shared() -> None:
    first = Chunk(
        chunk_id="a#s",
        doc_id="a",
        section_path="A > S",
        section_slug="s",
        text="text",
        owner_dept="general",
        doc_type="md",
        sensitivity="general",
    )
    second = Chunk(
        chunk_id="b#s",
        doc_id="b",
        section_path="B > S",
        section_slug="s",
        text="text",
        owner_dept="general",
        doc_type="md",
        sensitivity="general",
    )
    first.payload["x"] = 1
    assert second.payload == {}
    assert first.allowed_roles == ()


def test_chunks_jsonl_round_trip_as_a_file(tmp_path, micro_corpus: list[Chunk]) -> None:
    path = tmp_path / "chunks.jsonl"
    path.write_text("\n".join(c.to_json() for c in micro_corpus) + "\n", encoding="utf-8")
    loaded = [Chunk.from_json(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert loaded == micro_corpus


def test_hit_to_dict_is_json_serialisable() -> None:
    hit = Hit(
        chunk_id="finance/quarterly_financial_report#q4-revenue",
        doc_id="finance/quarterly_financial_report",
        section_slug="q4-revenue",
        rank=1,
        score=0.42,
        retriever="bm25",
        owner_dept="finance",
        doc_type="md",
        sensitivity="confidential",
        allowed_roles=("finance", "c_level"),
        text="Q4 revenue reached 2.6 billion dollars.",
        debug={"bm25_score": 7.1},
    )
    payload = json.loads(json.dumps(hit.to_dict()))
    assert payload["allowed_roles"] == ["finance", "c_level"]
    assert payload["debug"] == {"bm25_score": 7.1}
    assert payload["rank"] == 1


def test_resource_attrs_and_decision() -> None:
    attrs = ResourceAttrs(owner_dept="finance", doc_type="md", sensitivity="confidential")
    assert (attrs.owner_dept, attrs.doc_type, attrs.sensitivity) == (
        "finance",
        "md",
        "confidential",
    )
    permit = Decision(effect="permit", rule_name="finance_reads_finance")
    deny = Decision(effect="deny")
    assert permit.permitted is True
    assert deny.permitted is False
    assert deny.rule_name is None


def test_visible_ids_masks_by_role(micro_corpus: list[Chunk]) -> None:
    # Both finance chunks in the fixture are confidential, so the finance role
    # sees neither of them. Only c_level does.
    assert visible_ids(micro_corpus, "finance") == frozenset()
    assert visible_ids(micro_corpus, "marketing") == frozenset(
        {
            "marketing/market_report_q1_2024#campaign-spend",
            "marketing/market_report_q3_2024#campaign-spend",
        }
    )
    assert visible_ids(micro_corpus, "c_level") == frozenset(c.chunk_id for c in micro_corpus)


def test_visible_ids_denies_unknown_and_unlisted_roles(micro_corpus: list[Chunk]) -> None:
    assert visible_ids(micro_corpus, "employee") == frozenset()
    assert visible_ids(micro_corpus, "not_a_role") == frozenset()
    assert visible_ids([], "finance") == frozenset()
