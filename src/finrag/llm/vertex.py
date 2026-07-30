"""Vertex AI text generation: the real `LLMBackend` behind the answer step.

Shaped to match `retrieval/embed.py` on purpose, because the two solve the same
problem twice:

  * the client is built on first use, not in `__init__`, so importing this module
    and constructing the object need no credentials;
  * one transient failure does not sink a request, so a call is retried a few
    times and then given up on with a message that says what failed;
  * temperature is 0, because a grounded answer with citations should come out
    the same way twice. Creativity here would read as the system changing its
    mind about what the corpus says.

`build_llm` is what the serving layer calls. It may return None, meaning "no
generation backend is configured", and `generate.py` then answers extractively
from the retrieved text. That keeps the whole app usable with no credentials at
all, which is what a reviewer on a plane gets.

TWO CALLERS, TWO SHAPES. Answering wants a paragraph and tolerates None. R3's
reranker wants a permutation of twenty numbers, cannot work at all without a
model, and must not spend output tokens on thinking. `build_rerank_llm` at the
bottom of this file is that second shape: same class, different dials, and a
refusal instead of a None. The two builders live here rather than in
`retrieval/rerank.py` so every Vertex client this project opens is constructed in
one file.
"""

from __future__ import annotations

import os
import time
from typing import Any

from finrag.config import DEFAULT_VERTEX_LOCATION, DEFAULT_VERTEX_MODEL
from finrag.llm.base import LLMBackend
from finrag.llm.fake import FakeLLM

# Which generation backend the serving layer should use. Read from the
# environment rather than kept in `Settings`, because it picks a backend for one
# process (`FINRAG_LLM=off finrag serve`) rather than describing the deployment.
LLM_BACKEND_ENV = "FINRAG_LLM"

# "vertex" calls the real API, "fake" is the deterministic offline stand-in, and
# "off" means answer extractively with no model at all.
DEFAULT_LLM_BACKEND = "vertex"

# Where the model name comes from: VERTEX_MODEL, the same variable `config.py`
# reads, falling back to the shared default there. Verified against the real API
# from this repo, in project `plotpointe`, location us-central1:
#
#   gemini-2.0-flash   404, retired (it was the default until R6)
#   gemini-2.5-flash   works in us-central1        <- the default
#   gemini-3.6-flash   works only on VERTEX_LOCATION=global (404 in us-central1)
#
# The default itself lives in `config.py` so `Settings.vertex_model` and this
# module cannot drift apart into two different answers.
VERTEX_MODEL_ENV = "VERTEX_MODEL"

# The newest flash we could reach at all, for the wiring notes. Needs the global
# endpoint, so it is not the default while .env pins a region.
NEWEST_FLASH_MODEL = "gemini-3.6-flash"
NEWEST_FLASH_LOCATION = "global"

# A grounded answer over five short excerpts is a paragraph, not an essay. The
# cap is a cost guard, not a style choice.
DEFAULT_MAX_OUTPUT_TOKENS = 1024

# Same shape as the embedder's retry: three tries two seconds apart rides out a
# rate-limit blip without hiding a real outage.
RETRY_TRIES = 3
RETRY_SLEEP_SECONDS = 2.0

# A reranking answer is a JSON list of twenty numbers, about 80 tokens. The cap
# guards against a model that decides to explain itself, it is not a real limit.
RERANK_MAX_OUTPUT_TOKENS = 512

# Gemini 2.5 thinks by default and bills the thinking against max_output_tokens.
# On the rerank task that spent the whole budget and returned nothing parseable
# (R3 measured it: several seconds and an empty reply, versus about one second
# and `[2, 1, 3]` with the budget at zero). A permutation of twenty numbers has
# nothing for a thinking budget to buy, so the reranker pins it off.
NO_THINKING_BUDGET = 0

# How many tries the reranker's client makes for itself: one. `LLMReranker` has
# its own retry loop and counts every call, so a second retry underneath it would
# make the reported call count a third of the truth.
RERANK_RETRY_TRIES = 1

# Triple extraction asks for up to twelve JSON objects about one chunk, which is
# longer than a rerank permutation and shorter than an answer. Too small a cap
# truncates the array, and a truncated array does not parse, so the chunk is
# counted as a fallback and its facts are simply lost.
EXTRACTION_MAX_OUTPUT_TOKENS = 2048


class VertexLLM:
    """Single-turn generation through the google-genai SDK in Vertex mode.

    Pass `client` to hand in an already-built (or stub) client; leave it out and
    a real one is built on the first call.
    """

    name = "vertex"

    def __init__(
        self,
        project: str,
        location: str,
        model: str = DEFAULT_VERTEX_MODEL,
        temperature: float = 0.0,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        thinking_budget: int | None = None,
        retry_tries: int = RETRY_TRIES,
        client: Any | None = None,
    ) -> None:
        if not project:
            raise ValueError("VertexLLM needs a project id")
        self.project = project
        self.location = location
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        # None means "leave the model's own default alone". 0 switches thinking
        # off, which is what a caller wanting structured output should pass.
        self.thinking_budget = thinking_budget
        self.retry_tries = retry_tries
        self._client = client

    def client(self) -> Any:
        """The google-genai client, built once and kept.

        Imported inside the method so this module imports on a machine with no
        google SDK installed, which is how the offline test suite runs.
        """
        if self._client is None:
            from google import genai

            self._client = genai.Client(
                vertexai=True, project=self.project, location=self.location
            )
        return self._client

    def request_config(self, system: str | None) -> dict[str, Any]:
        """The per-call config, as a plain dict.

        A dict rather than `types.GenerateContentConfig` so nothing in this file
        imports the SDK at module level; google-genai accepts either.
        """
        config: dict[str, Any] = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
        }
        if system:
            config["system_instruction"] = system
        if self.thinking_budget is not None:
            # A nested plain dict, for the same reason the outer one is a dict:
            # google-genai validates it into a ThinkingConfig itself, so this file
            # still imports with no SDK present.
            config["thinking_config"] = {"thinking_budget": self.thinking_budget}
        return config

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Return the model's reply, retried a few times on any transport error."""
        last_error: Exception | None = None
        for attempt in range(self.retry_tries):
            try:
                response = self.client().models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=self.request_config(system),
                )
                return _reply_text(response)
            except Exception as error:  # noqa: BLE001 - any transport failure retries
                last_error = error
                if attempt + 1 < self.retry_tries:
                    time.sleep(RETRY_SLEEP_SECONDS)

        raise RuntimeError(
            f"Vertex generation failed {self.retry_tries} times with model "
            f"{self.model}: {last_error}"
        ) from last_error


def _reply_text(response: Any) -> str:
    """The text of a generate_content response, or "" when it carried none.

    A blocked or empty candidate gives `response.text is None`. Returning ""
    lets the caller treat it as "the model said nothing", which the answer step
    already handles as a refusal, instead of raising on an AttributeError.
    """
    text = getattr(response, "text", None)
    if text is None:
        return ""
    return str(text)


def backend_name() -> str:
    """Which generation backend this process should use."""
    return os.environ.get(LLM_BACKEND_ENV, DEFAULT_LLM_BACKEND).strip().lower()


def model_name() -> str:
    """The Vertex model to ask for. See the note on DEFAULT_VERTEX_MODEL."""
    return os.environ.get(VERTEX_MODEL_ENV) or DEFAULT_VERTEX_MODEL


def build_extraction_llm(settings: Any) -> VertexLLM:
    """The client `finrag graph --llm` reads triples with.

    Same reasoning as the rerank client, two dials apart: the answer is JSON, so
    thinking is off, but twelve triples need a larger cap than twenty numbers, and
    the retry stays at the full three because `graph/build_llm.py` has no retry of
    its own and a dropped chunk is a hole in the graph.

    The model is `model_name()`, the same one the answer step uses, because a
    graph and an answer being built by the same model is one fewer thing to
    explain when the two disagree.
    """
    project = getattr(settings, "vertex_project", None)
    if not project:
        raise RuntimeError(
            "Extracting triples needs a Vertex project. Set VERTEX_PROJECT (or "
            "GCP_PROJECT) in .env, or run `finrag graph` without --llm for the "
            "structural graph, which needs no model."
        )
    return VertexLLM(
        project=project,
        location=getattr(settings, "vertex_location", DEFAULT_VERTEX_LOCATION),
        model=model_name(),
        max_output_tokens=EXTRACTION_MAX_OUTPUT_TOKENS,
        thinking_budget=NO_THINKING_BUDGET,
    )


def build_rerank_llm(settings: Any) -> VertexLLM:
    """The client the `hybrid_rerank` mode reorders with, or a plain refusal.

    Three differences from `build_llm`, all of them because the caller is
    `retrieval/rerank.py` rather than the answer step:

      * no None. Reranking with no model is not a weaker answer, it is the hybrid
        order under a name that claims a model touched it. Refuse instead, and
        name the free mode in the message;
      * no `FINRAG_LLM` switch. That variable exists so the API can serve without
        credentials; a published rerank number must come from the real model, and
        the tests inject `FakeLLM` into `LLMReranker` directly;
      * its own model setting (`FINRAG_VERTEX_RERANK_MODEL`), thinking off, and a
        smaller token cap.
    """
    project = getattr(settings, "vertex_project", None)
    if not project:
        raise RuntimeError(
            "Reranking needs a Vertex project. Set VERTEX_PROJECT (or GCP_PROJECT) "
            "in .env, or run the `hybrid` mode, which needs no model."
        )
    return VertexLLM(
        project=project,
        location=getattr(settings, "vertex_location", DEFAULT_VERTEX_LOCATION),
        model=getattr(settings, "vertex_rerank_model", DEFAULT_VERTEX_MODEL),
        max_output_tokens=RERANK_MAX_OUTPUT_TOKENS,
        thinking_budget=NO_THINKING_BUDGET,
        retry_tries=RERANK_RETRY_TRIES,
    )


def build_llm(settings: Any) -> LLMBackend | None:
    """The generation backend named by the environment, or None for no model.

    None is a supported answer, not a failure: `generate.answer` falls back to
    quoting the retrieved text. A missing project id therefore degrades to
    extractive answers rather than refusing to serve, and the response says
    `extractive: true` so nobody mistakes a quote for a generated answer.
    """
    name = backend_name()
    if name == "off":
        return None
    if name == "fake":
        return FakeLLM()

    project = getattr(settings, "vertex_project", None)
    if not project:
        return None

    return VertexLLM(
        project=project,
        location=getattr(settings, "vertex_location", DEFAULT_VERTEX_LOCATION),
        model=model_name(),
    )
