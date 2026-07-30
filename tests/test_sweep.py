"""The leakage sweep, against a retriever that masks and one that does not.

The sweep's whole claim is that it would notice a leak. A test that only ran the
real retrievers could not tell "no leak happened" from "the check does nothing",
so the first thing here is `LeakyRetriever`, which returns every chunk to every
role. The sweep must catch it, count it exactly, and name the tuple behind it.

Then the real modes are swept over the same fixture corpus and must come back
clean. Policy semantics are not re-tested here; `test_policy.py` owns the role
matrix, and this file only checks that the sweep reports what the policy decides.

"Real modes" here means the offline ones. `graphrag` reads a Neo4j namespace built
from the real corpus, so it cannot be pointed at a six-chunk fixture; its mask is
proved in `tests/graph/test_traverse.py` against the test namespace, and on the
real corpus by `finrag sweep`.
"""

from __future__ import annotations

from typing import Any

import pytest

from finrag.eval import sweep
from finrag.eval.sweep import (
    breakdown_lines,
    offline_modes,
    run_leakage_sweep,
    sweep_passed,
)
from finrag.models import ROLES, Chunk, Hit
from finrag.policy.engine import PolicyEngine
from finrag.retrieval.embed import FakeEmbedder
from finrag.retrieval.registry import IndexBundle, available_modes, build_retriever

# Three questions is enough to prove the cell arithmetic and keeps the fake
# embedding of the fixture corpus fast.
QUESTIONS: list[dict[str, Any]] = [
    {"id": "Q-REVENUE", "role": "c_level", "question": "What was Q4 revenue?"},
    {"id": "Q-SPEND", "role": "marketing", "question": "What was the campaign spend?"},
    {"id": "Q-SALARY", "role": "hr", "question": "What is the average salary?"},
]


class LeakyRetriever:
    """Returns every chunk to every role. The bug the sweep exists to catch.

    No mask, no scoring, no `visible_ids` call. It is the exact shape of the
    mistake the ABAC invariant forbids: chunks occupying rank slots that the
    asking role was never permitted to see.
    """

    name = "leaky"

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = list(chunks)

    def search(self, query: str, role: str, k: int = 10) -> list[Hit]:
        hits: list[Hit] = []
        rank = 1
        for chunk in self.chunks[:k]:
            hits.append(
                Hit(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    section_slug=chunk.section_slug,
                    rank=rank,
                    score=1.0,
                    retriever=self.name,
                    owner_dept=chunk.owner_dept,
                    doc_type=chunk.doc_type,
                    sensitivity=chunk.sensitivity,
                    allowed_roles=chunk.allowed_roles,
                    text=chunk.text,
                )
            )
            rank += 1
        return hits


@pytest.fixture
def policy(policy_path) -> PolicyEngine:
    return PolicyEngine.from_yaml(policy_path)


@pytest.fixture
def real_retrievers(micro_corpus: list[Chunk]) -> dict[str, Any]:
    """Every registered mode, built over the fixture corpus, all offline.

    No artifacts on disk: BM25 builds its index in memory, dense embeds the six
    chunks with the hash embedder, and the reranker's backend always fails so the
    fused order is what comes out.
    """
    bundle = IndexBundle(
        chunks=micro_corpus,
        embedder=FakeEmbedder(dim=32),
        llm=sweep.FallbackOnlyLLM(),
    )
    retrievers: dict[str, Any] = {}
    for mode in offline_modes():
        retrievers[mode] = build_retriever(mode, bundle)
    return retrievers


def test_the_sweep_checks_every_question_role_and_mode_combination(
    policy: PolicyEngine, micro_corpus: list[Chunk]
) -> None:
    report = run_leakage_sweep({"leaky": LeakyRetriever(micro_corpus)}, policy, QUESTIONS)

    assert report["cells_checked"] == len(QUESTIONS) * len(ROLES) * 1
    assert report["roles"] == list(ROLES)
    assert report["modes"] == ["leaky"]


def test_the_sweep_catches_a_retriever_that_ignores_the_role_mask(
    policy: PolicyEngine, micro_corpus: list[Chunk]
) -> None:
    report = run_leakage_sweep({"leaky": LeakyRetriever(micro_corpus)}, policy, QUESTIONS)

    assert report["violations"] > 0
    assert sweep_passed(report) is False


def test_a_leak_is_reported_with_the_question_role_mode_and_chunk(
    policy: PolicyEngine, micro_corpus: list[Chunk]
) -> None:
    report = run_leakage_sweep({"leaky": LeakyRetriever(micro_corpus)}, policy, QUESTIONS)

    first = report["violation_details"][0]
    assert first["question_id"] in {"Q-REVENUE", "Q-SPEND", "Q-SALARY"}
    assert first["asked_as_role"] in ROLES
    assert first["mode"] == "leaky"
    assert first["chunk_id"] in {chunk.chunk_id for chunk in micro_corpus}


def test_the_sweep_finds_the_leak_in_a_role_no_question_was_written_for(
    policy: PolicyEngine, micro_corpus: list[Chunk]
) -> None:
    # `employee` is not the gold role of any question above, and the fixture
    # corpus has no `general` documents, so the policy lets employee see nothing
    # at all. The harness would never ask in that role; the sweep always does.
    report = run_leakage_sweep({"leaky": LeakyRetriever(micro_corpus)}, policy, QUESTIONS)

    assert report["per_role"]["employee"]["violations"] == len(QUESTIONS) * len(micro_corpus)


def test_c_level_sees_the_whole_fixture_corpus_without_a_violation(
    policy: PolicyEngine, micro_corpus: list[Chunk]
) -> None:
    # The same leaky retriever, read from the one role that is allowed
    # everything. Proves the sweep counts denials, not merely returned hits.
    report = run_leakage_sweep({"leaky": LeakyRetriever(micro_corpus)}, policy, QUESTIONS)

    assert report["per_role"]["c_level"]["hits"] == len(QUESTIONS) * len(micro_corpus)
    assert report["per_role"]["c_level"]["violations"] == 0


@pytest.mark.parametrize("mode", offline_modes())
def test_each_real_mode_sweeps_clean_on_the_fixture_corpus(
    mode: str, policy: PolicyEngine, real_retrievers: dict[str, Any]
) -> None:
    report = run_leakage_sweep({mode: real_retrievers[mode]}, policy, QUESTIONS)

    assert report["violations"] == 0
    assert report["violation_details"] == []


def test_the_real_modes_return_enough_hits_for_the_sweep_to_be_meaningful(
    policy: PolicyEngine, real_retrievers: dict[str, Any]
) -> None:
    # A sweep over a retriever that returned nothing would also report zero
    # violations, and would prove nothing. So the clean run has to have inspected
    # real hits.
    report = run_leakage_sweep(real_retrievers, policy, QUESTIONS)

    assert report["hits_inspected"] > 0
    for mode in report["modes"]:
        assert report["per_mode"][mode]["hits"] > 0


def test_a_role_that_may_see_nothing_counts_as_an_empty_cell_not_a_violation(
    policy: PolicyEngine, real_retrievers: dict[str, Any]
) -> None:
    report = run_leakage_sweep(real_retrievers, policy, QUESTIONS)

    employee = report["per_role"]["employee"]
    assert employee["hits"] == 0
    assert employee["violations"] == 0
    assert employee["empty_cells"] == employee["cells"]


def test_the_breakdown_table_has_a_row_per_role_and_one_total_row(
    policy: PolicyEngine, real_retrievers: dict[str, Any]
) -> None:
    report = run_leakage_sweep(real_retrievers, policy, QUESTIONS)
    lines = breakdown_lines(report)

    # One header, one row per role, one ALL row.
    assert len(lines) == 1 + len(ROLES) + 1
    assert lines[0].startswith("role")
    assert lines[-1].startswith("ALL")


def test_the_swept_agent_falls_back_to_one_plain_search_and_still_returns_hits(
    real_retrievers: dict[str, Any],
) -> None:
    # The mode is swept offline because the stub backend fails, the loop forces a
    # finish, and the fallback search puts real hits through the union and the
    # last lock. Empty hits here would mean the react column tests nothing.
    retriever = real_retrievers["react"]

    hits = retriever.search("What was the campaign spend?", "marketing", k=10)

    assert hits != []
    assert retriever.totals["fallback_searches"] == 1
    assert retriever.totals["hops"] == 0
    assert retriever.totals["call_failures"] == 2


def test_the_offline_rerank_backend_refuses_to_generate() -> None:
    # The sweep depends on this failing, not on it returning something bland: a
    # backend that answered would make the sweep call a model.
    with pytest.raises(RuntimeError):
        sweep.FallbackOnlyLLM().generate("rank these")


# Which modes a machine can sweep. graphrag needs the Neo4j container, and the
# rule is: dropped silently only when nobody asked for it by name.


def test_the_offline_set_is_every_mode_that_needs_no_container() -> None:
    assert "graphrag" not in offline_modes()
    for mode in available_modes():
        if mode != "graphrag":
            assert mode in offline_modes()


def test_the_offline_set_and_the_graph_modes_together_are_the_whole_registry() -> None:
    # A new mode must land in one of the two lists. Without this, a mode added to
    # the registry and forgotten here would never be swept at all.
    covered = set(offline_modes()) | set(sweep.MODES_NEEDING_A_GRAPH)
    assert covered == set(available_modes())


def test_asking_for_a_graph_mode_with_bolt_down_is_refused_by_name(
    monkeypatch: pytest.MonkeyPatch, settings
) -> None:
    monkeypatch.setattr(sweep, "bolt_is_open", lambda *args, **kwargs: False)
    with pytest.raises(RuntimeError, match="docker start finrag-neo4j"):
        sweep.build_sweep_retrievers(settings, ["graphrag"])


def test_the_default_mode_list_drops_graph_modes_when_bolt_is_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sweep, "bolt_is_open", lambda *args, **kwargs: False)
    assert sweep.sweepable_modes() == offline_modes()


def test_the_default_mode_list_is_the_whole_registry_when_bolt_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sweep, "bolt_is_open", lambda *args, **kwargs: True)
    assert sweep.sweepable_modes() == available_modes()


# The cold-cache guard. Before this, "the sweep is offline" meant "the questions
# happened to be cached"; one new gold question would have made the free gate
# start billing per query, silently.


def _vertex_settings(tmp_path) -> Any:
    """Settings that ask for the real embedder, over an empty artifacts folder."""
    from dataclasses import replace as replace_fields

    from finrag.config import Settings

    return replace_fields(
        Settings(),
        artifacts_dir=tmp_path,
        embedder="vertex",
        vertex_project="demo-project",
    )


def test_a_cold_cache_names_the_question_instead_of_calling_the_api(tmp_path) -> None:
    embedder = sweep.build_sweep_embedder(_vertex_settings(tmp_path))

    with pytest.raises(RuntimeError) as caught:
        embedder.embed_texts(["what was the campaign spend in Q3"])

    message = str(caught.value)
    assert "campaign spend in Q3" in message
    assert "finrag eval --mode dense" in message


def test_a_warm_cache_answers_without_the_backend(tmp_path) -> None:
    # The other half: a cached question must still sweep, with nothing called.
    from finrag.retrieval.embed import EMBED_DIM, FakeEmbedder
    from finrag.retrieval.embed_cache import CachingEmbedder, EmbeddingCache

    settings = _vertex_settings(tmp_path)
    cache = EmbeddingCache.load(settings.embed_cache_path, settings.vertex_embed_model, EMBED_DIM)
    CachingEmbedder(FakeEmbedder(), cache).embed_texts(["a cached question"])

    embedder = sweep.build_sweep_embedder(settings)
    assert embedder.embed_texts(["a cached question"]).shape == (1, EMBED_DIM)


def test_the_fake_embedder_is_handed_back_untouched(tmp_path) -> None:
    # No credentials, no cache, no hazard: the offline path must not be made to
    # depend on a cache file that only a paid run would fill.
    from dataclasses import replace as replace_fields

    from finrag.config import Settings

    settings = replace_fields(Settings(), artifacts_dir=tmp_path, embedder="fake")
    embedder = sweep.build_sweep_embedder(settings)

    assert embedder.embed_texts(["never seen before"]).shape[0] == 1
