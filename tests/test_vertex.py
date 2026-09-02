"""VertexLLM: construction, request config, retry. No network, ever.

Every call in this file goes to a stub client defined below, so the suite runs
with no credentials and no google SDK reachable.
"""

from __future__ import annotations

import pytest

from finrag.config import Settings
from finrag.llm.base import LLMBackend
from finrag.llm.fake import FakeLLM
from finrag.llm.vertex import (
    AGENT_MAX_OUTPUT_TOKENS,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_VERTEX_MODEL,
    RERANK_MAX_OUTPUT_TOKENS,
    RERANK_RETRY_TRIES,
    RETRY_TRIES,
    VertexLLM,
    build_query_llm,
    build_llm,
    build_rerank_llm,
    model_name,
)


class StubReply:
    """What generate_content returns: an object with a `.text`."""

    def __init__(self, text: str | None) -> None:
        self.text = text


class StubModels:
    """Hands back the queued replies in order; an Exception in the queue is raised."""

    def __init__(self, replies: list[object]) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, object]] = []

    def generate_content(self, model: str, contents: str, config: dict) -> StubReply:
        self.calls.append({"model": model, "contents": contents, "config": config})
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        assert isinstance(item, StubReply)
        return item


class StubClient:
    """Stands in for a google-genai client."""

    def __init__(self, *replies: object) -> None:
        self.models = StubModels(list(replies))


def test_vertex_llm_satisfies_the_llm_backend_protocol() -> None:
    llm = VertexLLM(project="demo-project", location="us-central1")
    assert isinstance(llm, LLMBackend)


def test_construction_without_a_project_is_refused() -> None:
    with pytest.raises(ValueError, match="project id"):
        VertexLLM(project="", location="us-central1")


def test_no_client_is_built_until_a_call_is_made() -> None:
    # The registry and the API import this module whether or not anyone asks for
    # a real model, so construction must never need credentials.
    llm = VertexLLM(project="demo-project", location="us-central1")
    assert llm._client is None


def test_the_request_config_pins_temperature_to_zero() -> None:
    llm = VertexLLM(project="demo-project", location="us-central1")
    config = llm.request_config(system=None)
    assert config["temperature"] == 0.0


def test_the_system_instruction_is_left_out_when_there_is_none() -> None:
    llm = VertexLLM(project="demo-project", location="us-central1")
    assert "system_instruction" not in llm.request_config(system=None)


def test_the_system_instruction_is_passed_through_when_given() -> None:
    llm = VertexLLM(project="demo-project", location="us-central1")
    config = llm.request_config(system="be brief")
    assert config["system_instruction"] == "be brief"


def test_generate_returns_the_reply_text() -> None:
    client = StubClient(StubReply("Q4 revenue was 2.6 billion dollars."))
    llm = VertexLLM(project="demo-project", location="us-central1", client=client)
    assert llm.generate("what was Q4 revenue?") == "Q4 revenue was 2.6 billion dollars."


def test_generate_sends_the_prompt_and_the_configured_model() -> None:
    client = StubClient(StubReply("ok"))
    llm = VertexLLM(
        project="demo-project",
        location="us-central1",
        model="gemini-2.0-flash",
        client=client,
    )
    llm.generate("hello")
    call = client.models.calls[0]
    assert call["model"] == "gemini-2.0-flash"
    assert call["contents"] == "hello"


def test_a_reply_with_no_text_reads_as_an_empty_string() -> None:
    # A blocked or empty candidate gives text=None. The answer step treats "" as
    # a refusal, which is the behaviour we want, so this must not raise.
    client = StubClient(StubReply(None))
    llm = VertexLLM(project="demo-project", location="us-central1", client=client)
    assert llm.generate("hello") == ""


def test_one_transient_failure_is_retried_and_then_succeeds(monkeypatch) -> None:
    monkeypatch.setattr("finrag.llm.vertex.RETRY_SLEEP_SECONDS", 0.0)
    client = StubClient(RuntimeError("503 unavailable"), StubReply("second time lucky"))
    llm = VertexLLM(project="demo-project", location="us-central1", client=client)
    assert llm.generate("hello") == "second time lucky"


def test_generate_gives_up_after_the_retry_budget(monkeypatch) -> None:
    monkeypatch.setattr("finrag.llm.vertex.RETRY_SLEEP_SECONDS", 0.0)
    failures: list[object] = [RuntimeError("503 unavailable") for _ in range(RETRY_TRIES)]
    client = StubClient(*failures)
    llm = VertexLLM(project="demo-project", location="us-central1", client=client)
    with pytest.raises(RuntimeError, match=f"failed {RETRY_TRIES} times"):
        llm.generate("hello")


def test_build_llm_returns_the_offline_fake_when_the_environment_asks_for_it(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FINRAG_LLM", "fake")
    assert isinstance(build_llm(Settings()), FakeLLM)


def test_build_llm_returns_no_backend_when_generation_is_switched_off(monkeypatch) -> None:
    monkeypatch.setenv("FINRAG_LLM", "off")
    assert build_llm(Settings(vertex_project="demo-project")) is None


def test_build_llm_returns_no_backend_when_no_project_is_configured(monkeypatch) -> None:
    # No project means no Vertex call is possible. Answering extractively is
    # better than refusing to serve, so this is None rather than an exception.
    monkeypatch.delenv("FINRAG_LLM", raising=False)
    assert build_llm(Settings(vertex_project=None)) is None


def test_build_llm_returns_a_vertex_backend_when_a_project_is_configured(monkeypatch) -> None:
    monkeypatch.delenv("FINRAG_LLM", raising=False)
    monkeypatch.delenv("VERTEX_MODEL", raising=False)
    llm = build_llm(Settings(vertex_project="demo-project"))
    assert isinstance(llm, VertexLLM)
    assert llm.model == DEFAULT_VERTEX_MODEL


def test_the_model_name_comes_from_the_environment_when_it_is_set(monkeypatch) -> None:
    monkeypatch.delenv("FINRAG_LLM", raising=False)
    monkeypatch.setenv("VERTEX_MODEL", "gemini-3.6-flash")
    llm = build_llm(Settings(vertex_project="demo-project"))
    assert llm.model == "gemini-3.6-flash"


def test_the_default_model_is_not_the_retired_one(monkeypatch) -> None:
    # gemini-2.0-flash answers 404 for this project. Falling back to it here would
    # make the real backend fail on a clean checkout, which is the trap this
    # default exists to avoid.
    monkeypatch.delenv("VERTEX_MODEL", raising=False)
    assert model_name() != "gemini-2.0-flash"


# The rerank client, which is the same class with different dials. R3's reranker
# was written against its own vertex.py; at integration that file was replaced by
# this one and `build_rerank_llm` is the seam.


def test_thinking_is_left_out_of_the_config_unless_it_is_asked_for() -> None:
    # The answer step wants the model's own default, so nothing about thinking
    # should appear in a plain config.
    llm = VertexLLM(project="demo-project", location="us-central1")
    assert "thinking_config" not in llm.request_config(system=None)


def test_a_thinking_budget_of_zero_is_sent_as_a_nested_config() -> None:
    llm = VertexLLM(project="demo-project", location="us-central1", thinking_budget=0)
    config = llm.request_config(system=None)
    assert config["thinking_config"] == {"thinking_budget": 0}


def test_the_rerank_client_switches_thinking_off_and_caps_the_reply() -> None:
    # Both of these are the reason R3's rerank calls took a second instead of
    # several and came back parseable. See DECISIONS, R3.
    llm = build_rerank_llm(Settings(vertex_project="demo-project"))
    assert llm.thinking_budget == 0
    assert llm.max_output_tokens == RERANK_MAX_OUTPUT_TOKENS
    assert llm.max_output_tokens < DEFAULT_MAX_OUTPUT_TOKENS


def test_the_rerank_client_uses_the_rerank_model_setting() -> None:
    settings = Settings(vertex_project="demo-project", vertex_rerank_model="gemini-9.9-flash")
    assert build_rerank_llm(settings).model == "gemini-9.9-flash"


def test_the_rerank_client_retries_once_so_the_reranker_owns_the_retry() -> None:
    # LLMReranker counts its own calls and retries three times. A retry loop
    # underneath it would make the reported call count a third of the truth.
    llm = build_rerank_llm(Settings(vertex_project="demo-project"))
    assert llm.retry_tries == RERANK_RETRY_TRIES == 1


def test_the_rerank_client_makes_exactly_one_call_per_generate() -> None:
    client = StubClient(RuntimeError("503 unavailable"))
    llm = build_rerank_llm(Settings(vertex_project="demo-project"))
    llm._client = client
    with pytest.raises(RuntimeError, match="failed 1 times"):
        llm.generate("hello")
    assert len(client.models.calls) == 1


def test_the_rerank_client_ignores_the_generation_backend_switch(monkeypatch) -> None:
    # FINRAG_LLM=fake exists so the API can serve with no credentials. A rerank
    # number published from a fake would be a lie, so this builder never reads it.
    monkeypatch.setenv("FINRAG_LLM", "fake")
    assert isinstance(build_rerank_llm(Settings(vertex_project="demo-project")), VertexLLM)


def test_the_agent_client_keeps_thinking_off_but_doubles_the_output_cap() -> None:
    # A rerank reply is twenty numbers. An agent reply is a thought plus, on the
    # last hop, the whole answer, and a truncated JSON object does not parse.
    llm = build_query_llm(Settings(vertex_project="demo-project"))
    assert llm.thinking_budget == 0
    assert llm.max_output_tokens == AGENT_MAX_OUTPUT_TOKENS
    assert llm.max_output_tokens > RERANK_MAX_OUTPUT_TOKENS


def test_the_agent_client_retries_once_so_the_loop_owns_the_retry() -> None:
    # The loop counts every call it makes, and its error-note retry is the retry.
    llm = build_query_llm(Settings(vertex_project="demo-project"))
    assert llm.retry_tries == RERANK_RETRY_TRIES == 1


def test_the_agent_client_refuses_rather_than_returning_none(monkeypatch) -> None:
    monkeypatch.delenv("FINRAG_LLM", raising=False)
    with pytest.raises(RuntimeError, match="direct query strategy"):
        build_query_llm(Settings(vertex_project=None))


def test_the_rerank_client_refuses_rather_than_returning_none(monkeypatch) -> None:
    # The answer step can quote text with no model. A selected LLM reranker
    # cannot silently become the independent `none` strategy.
    monkeypatch.delenv("FINRAG_LLM", raising=False)
    with pytest.raises(RuntimeError, match="reranker none"):
        build_rerank_llm(Settings(vertex_project=None))
