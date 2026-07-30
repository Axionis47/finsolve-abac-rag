"""The offline fakes must be deterministic, or nothing downstream is reproducible."""

from __future__ import annotations

import numpy as np
import pytest

from finrag.llm.base import EmbeddingBackend, LLMBackend
from finrag.llm.fake import FakeEmbeddings, FakeLLM, ScriptedLLM


def test_fake_llm_returns_canned_response_on_exact_match() -> None:
    llm = FakeLLM({"extract triples": '{"triples": []}'})
    assert llm.generate("extract triples") == '{"triples": []}'


def test_fake_llm_stub_is_deterministic_and_prompt_specific(fake_llm: FakeLLM) -> None:
    first = fake_llm.generate("what was Q4 revenue?")
    second = FakeLLM().generate("what was Q4 revenue?")
    assert first == second
    assert first != fake_llm.generate("what was Q1 spend?")


def test_fake_llm_records_calls(fake_llm: FakeLLM) -> None:
    fake_llm.generate("hello", system="be terse")
    assert fake_llm.calls == [("hello", "be terse")]


def test_fake_llm_satisfies_the_protocol(fake_llm: FakeLLM) -> None:
    assert isinstance(fake_llm, LLMBackend)


def test_scripted_llm_answers_in_order_whatever_the_prompt() -> None:
    llm = ScriptedLLM(["first", "second"])
    assert llm.generate("anything") == "first"
    assert llm.generate("something else") == "second"


def test_scripted_llm_repeats_its_last_reply_once_the_script_runs_out() -> None:
    # A loop that asks more times than the script expected should keep going, so
    # the test that catches it is the one asserting the call count, not a crash.
    llm = ScriptedLLM(["only"])
    assert [llm.generate("a"), llm.generate("b")] == ["only", "only"]


def test_scripted_llm_needs_a_script() -> None:
    with pytest.raises(ValueError, match="at least one reply"):
        ScriptedLLM([])


def test_scripted_llm_satisfies_the_protocol() -> None:
    assert isinstance(ScriptedLLM(["x"]), LLMBackend)


def test_fake_embeddings_dim_and_shape(fake_embeddings: FakeEmbeddings) -> None:
    assert fake_embeddings.dim == 64
    vectors = fake_embeddings.embed(["alpha beta", "gamma delta", "epsilon"])
    assert vectors.shape == (3, 64)
    assert vectors.dtype == np.float32
    assert isinstance(fake_embeddings, EmbeddingBackend)


def test_fake_embeddings_are_l2_normalised(fake_embeddings: FakeEmbeddings) -> None:
    vectors = fake_embeddings.embed(
        ["quarterly revenue", "customer churn in the holiday quarter", "x"]
    )
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_fake_embeddings_are_deterministic(fake_embeddings: FakeEmbeddings) -> None:
    text = "North America saturation is the leading risk"
    first = fake_embeddings.embed([text])
    second = fake_embeddings.embed([text])
    third = FakeEmbeddings().embed([text, "unrelated"])[0]
    assert np.array_equal(first, second)
    assert np.array_equal(first[0], third)


def test_fake_embeddings_differ_for_different_texts(fake_embeddings: FakeEmbeddings) -> None:
    vectors = fake_embeddings.embed(
        [
            "Q1 campaign spend was 2.0 million dollars",
            "recovery time objective is four hours",
            "average salary by department",
        ]
    )
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            assert not np.array_equal(vectors[i], vectors[j])


def test_fake_embeddings_similarity_tracks_overlap(fake_embeddings: FakeEmbeddings) -> None:
    vectors = fake_embeddings.embed(
        [
            "quarterly campaign spend report",
            "quarterly campaign spend summary",
            "recovery point objective fifteen minutes",
        ]
    )
    near = float(vectors[0] @ vectors[1])
    far = float(vectors[0] @ vectors[2])
    assert near > far


def test_empty_text_gives_the_zero_vector(fake_embeddings: FakeEmbeddings) -> None:
    vectors = fake_embeddings.embed(["", "  "])
    assert np.array_equal(vectors[0], np.zeros(fake_embeddings.dim, dtype=np.float32))
    assert np.array_equal(vectors[1], vectors[0])


def test_embed_of_empty_list_has_the_right_shape(fake_embeddings: FakeEmbeddings) -> None:
    assert fake_embeddings.embed([]).shape == (0, 64)


def test_dim_must_be_positive() -> None:
    with pytest.raises(ValueError):
        FakeEmbeddings(dim=0)
