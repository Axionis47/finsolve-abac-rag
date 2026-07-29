"""Gemini text generation on Vertex AI, the real side of `LLMBackend`.

`fake.py` has satisfied that protocol since R0 so the suite runs offline. R3 is
the first rung that needs the real one: the hybrid_rerank mode asks a model to
reorder twenty passages.

Shaped like `VertexEmbedder` in `retrieval/embed.py` on purpose. The client is
built on first use rather than in `__init__`, so importing this module and
constructing the object need no credentials, and one call is one call: retries
and any fallback belong to the caller that knows what a failure means. The
reranker is that caller, and it counts its own fallbacks.
"""

from __future__ import annotations

from typing import Any

# Reranking is a judgement about an ordering, not a piece of writing, so nothing
# here should vary between two runs of the same eval.
TEMPERATURE = 0.0

# A JSON list of twenty numbers is about 80 tokens. The cap is a guard against a
# model that decides to explain itself at length, not a real limit on the answer.
MAX_OUTPUT_TOKENS = 512


class VertexLLM:
    """One-shot Gemini generation through the google-genai SDK.

    Thinking is switched off. The reranker wants an ordering, and a thinking
    budget spends output tokens on reasoning we never read, which on this task
    turned a one second call into several.
    """

    name = "vertex"

    def __init__(
        self,
        project: str,
        location: str,
        model: str,
        client: Any | None = None,
    ) -> None:
        if not project:
            raise ValueError("VertexLLM needs a project id")
        self.project = project
        self.location = location
        self.model = model
        self._client = client

    def client(self) -> Any:
        """The google-genai client, built once and kept."""
        if self._client is None:
            from google import genai

            self._client = genai.Client(
                vertexai=True, project=self.project, location=self.location
            )
        return self._client

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Return the model's text reply, or raise. No retry here by design."""
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            system_instruction=system,
        )
        response = self.client().models.generate_content(
            model=self.model, contents=prompt, config=config
        )
        text = response.text
        if text is None:
            # A blocked or empty reply is not a string we can parse, and calling
            # it "" would look to the caller like a model that said nothing
            # rather than a call that failed.
            raise RuntimeError(f"Gemini ({self.model}) returned no text for this prompt.")
        return text


def build_llm(settings: Any) -> VertexLLM:
    """The Gemini backend named by the settings, or a plain refusal.

    A missing project id is refused here, with the fix in the message, rather
    than falling back to a fake and publishing its numbers as real ones. There is
    deliberately no "fake" branch: the offline path is to pick a mode that needs
    no model, and the tests inject `FakeLLM` directly.
    """
    if not settings.vertex_project:
        raise RuntimeError(
            "Reranking needs a Vertex project. Set VERTEX_PROJECT (or GCP_PROJECT) "
            "in .env, or run the `hybrid` mode, which needs no model."
        )
    return VertexLLM(
        project=settings.vertex_project,
        location=settings.vertex_location,
        model=settings.vertex_rerank_model,
    )
