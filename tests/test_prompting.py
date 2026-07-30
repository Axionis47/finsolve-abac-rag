"""The pure text parts: prompt shape, citation parsing, the extractive quote."""

from __future__ import annotations

from finrag.models import Hit
from finrag.prompting import (
    REFUSAL_TOKEN,
    build_prompt,
    excerpt,
    extractive_answer,
    hit_view,
    is_refusal,
    parse_citations,
)


def make_hit(text: str, chunk_id: str = "marketing/doc#section") -> Hit:
    """A minimal hit. These functions only read text, ids and section names."""
    return Hit(
        chunk_id=chunk_id,
        doc_id="marketing/doc",
        section_slug="section",
        rank=1,
        score=1.0,
        retriever="bm25",
        owner_dept="marketing",
        doc_type="md",
        sensitivity="internal",
        allowed_roles=("marketing", "c_level"),
        text=text,
    )


def test_the_prompt_numbers_every_excerpt_and_names_its_chunk_id() -> None:
    hits = [make_hit("first text"), make_hit("second text", chunk_id="marketing/doc#other")]
    prompt = build_prompt("a question", hits)
    assert "[1] chunk_id: marketing/doc#section" in prompt
    assert "[2] chunk_id: marketing/doc#other" in prompt


def test_the_prompt_carries_the_question_and_the_refusal_instruction() -> None:
    prompt = build_prompt("what was Q4 revenue", [make_hit("some text")])
    assert "Question: what was Q4 revenue" in prompt
    assert REFUSAL_TOKEN in prompt


def test_an_excerpt_longer_than_the_limit_is_cut_and_marked() -> None:
    assert excerpt("abcdefghij", limit=4) == "abcd ..."


def test_an_excerpt_is_flattened_onto_one_line() -> None:
    # Newlines inside a numbered excerpt would look like the start of the next one.
    assert excerpt("two\nlines  here") == "two lines here"


def test_a_citation_becomes_the_chunk_id_of_that_excerpt() -> None:
    hits = [make_hit("one"), make_hit("two", chunk_id="marketing/doc#other")]
    assert parse_citations("as [2] shows", hits) == ["marketing/doc#other"]


def test_a_citation_number_with_no_matching_excerpt_is_dropped() -> None:
    hits = [make_hit("one")]
    assert parse_citations("as shown [9]", hits) == []


def test_a_repeated_citation_is_listed_once() -> None:
    hits = [make_hit("one")]
    assert parse_citations("[1] and again [1]", hits) == ["marketing/doc#section"]


def test_citations_come_back_in_the_order_they_were_cited() -> None:
    hits = [make_hit("one"), make_hit("two", chunk_id="marketing/doc#other")]
    assert parse_citations("[2] then [1]", hits) == [
        "marketing/doc#other",
        "marketing/doc#section",
    ]


def test_the_refusal_token_reads_as_a_refusal() -> None:
    assert is_refusal(f"  {REFUSAL_TOKEN}  ")


def test_an_empty_reply_reads_as_a_refusal() -> None:
    assert is_refusal("   ")


def test_a_real_answer_does_not_read_as_a_refusal() -> None:
    assert not is_refusal("Gross margin was 64% [1].")


def test_the_extractive_answer_picks_the_sentences_that_overlap_the_question() -> None:
    hit = make_hit(
        "The office moved in March. Campaign spend was 2.0 million dollars. "
        "The lift was replaced."
    )
    assert extractive_answer("what was campaign spend", hit) == (
        "Campaign spend was 2.0 million dollars."
    )


def test_the_extractive_answer_quotes_the_opening_sentence_when_nothing_overlaps() -> None:
    hit = make_hit("The lift was replaced. The office moved in March.")
    assert extractive_answer("kubernetes autoscaling", hit) == "The lift was replaced."


def test_a_hit_view_carries_no_payload_and_no_allowed_roles() -> None:
    view = hit_view(make_hit("some text"))
    assert "payload" not in view
    assert "allowed_roles" not in view
