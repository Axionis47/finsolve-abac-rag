"""The react mode: the parser, the loop, the ranking, and the two masks.

Nothing here calls a model. Two stand-ins do the work:

  * `ScriptedLLM` from `llm/fake.py`, which answers the first call with the first
    reply and so on. A ReAct prompt changes every hop, so keying replies on the
    prompt text (what `FakeLLM` does) would mean pasting a transcript into every
    test.
  * `PlannedRetriever` below, a well-behaved retriever that returns exactly the
    chunks a test says a query should find, and `LeakyRetriever`, which returns
    everything regardless of role. The second one is the compromised tool the
    mask tests are about.

The behaviours worth pinning are the ones a whole eval run would hide. A loop
that fell back on every question still produces a full report; only the counters
tell you. A loop that let a denied excerpt into a prompt would leak to the model
without leaking to the response, and no metric in this repo would notice.

Three modules, one test file. `react_prompt.py`, `react.py` and
`react_retriever.py` are one mode split for length, and splitting the tests the
same way would put the mask assertions in three files. Disclosed in DECISIONS.
"""

from __future__ import annotations

import json
import re

import pytest

from finrag.llm.fake import FakeLLM, ScriptedLLM
from finrag.models import ROLES, Chunk, Hit
from finrag.react import MAX_HOPS, AgentRun, ReactAgent, new_stats
from finrag.react_prompt import (
    SYSTEM_PROMPT,
    build_agent_prompt,
    normalise_query,
    parse_step,
)
from finrag.react_retriever import (
    ReactRetriever,
    cited_numbers,
    rank_observations,
    renumber_citations,
)
from finrag.retrieval.base import visible_ids
from finrag.retrieval.bm25 import BM25Retriever
from finrag.retrieval.dense import DenseRetriever
from finrag.retrieval.embed import FakeEmbedder
from finrag.retrieval.hybrid import HybridRetriever
from finrag.retrieval.registry import IndexBundle, available_modes, build_retriever

QUESTION = "which campaign mitigates the saturation risk"
SECRET_TEXT = "north america saturation is the leading risk this quarter"


def _chunk(chunk_id: str, text: str, allowed_roles: tuple[str, ...]) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=chunk_id.split("#")[0],
        section_path="Report > Section",
        section_slug=chunk_id.split("#")[1],
        text=text,
        owner_dept="finance" if chunk_id.startswith("finance/") else "marketing",
        doc_type="md",
        sensitivity="confidential" if chunk_id.startswith("finance/") else "internal",
        allowed_roles=allowed_roles,
    )


VISIBLE = ("marketing", "c_level")

A = _chunk("marketing/a#one", "the festive quarter campaign ran in three regions", VISIBLE)
B = _chunk("marketing/b#two", "brand colours and typeface", VISIBLE)
C = _chunk("marketing/c#three", "customer churn by region", VISIBLE)
SECRET = _chunk("finance/secret#four", SECRET_TEXT, ("c_level",))
CORPUS = [A, B, C, SECRET]


def hit_for(chunk: Chunk, rank: int) -> Hit:
    return Hit(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        section_slug=chunk.section_slug,
        rank=rank,
        score=1.0 / rank,
        retriever="planned",
        owner_dept=chunk.owner_dept,
        doc_type=chunk.doc_type,
        sensitivity=chunk.sensitivity,
        allowed_roles=chunk.allowed_roles,
        text=chunk.text,
    )


class PlannedRetriever:
    """Returns the chunks a test says a query should find, masked by role.

    A well-behaved retriever: it applies the mask before it ranks, the way every
    real one does. `default` is what an unplanned query returns, which is how the
    fallback search finds anything.
    """

    name = "planned"

    def __init__(
        self,
        chunks: list[Chunk],
        plan: dict[str, list[Chunk]] | None = None,
        default: list[Chunk] | None = None,
    ) -> None:
        self.chunks = list(chunks)
        self.plan = dict(plan or {})
        self.default = list(default if default is not None else chunks)
        self.queries: list[str] = []

    def search(self, query: str, role: str, k: int = 10) -> list[Hit]:
        self.queries.append(query)
        allowed = visible_ids(self.chunks, role)
        hits: list[Hit] = []
        rank = 1
        for chunk in self.plan.get(query, self.default):
            if chunk.chunk_id not in allowed:
                continue
            if rank > k:
                break
            hits.append(hit_for(chunk, rank))
            rank += 1
        return hits


class LeakyRetriever:
    """Returns every chunk it was given, role or no role.

    Stands in for a broken or future base retriever. The agent must not show its
    output to the model, let alone return it.
    """

    name = "leaky"

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = list(chunks)

    def search(self, query: str, role: str, k: int = 10) -> list[Hit]:
        hits: list[Hit] = []
        rank = 1
        for chunk in self.chunks[:k]:
            hits.append(hit_for(chunk, rank))
            rank += 1
        return hits


def reply(action: str, argument: str, thought: str = "next step") -> str:
    return json.dumps({"thought": thought, "action": action, "argument": argument})


def agent_for(replies: list[str], base=None, **kwargs) -> tuple[ReactAgent, ScriptedLLM]:
    llm = ScriptedLLM(replies)
    agent = ReactAgent(CORPUS, base or PlannedRetriever(CORPUS), llm, **kwargs)
    return agent, llm


def ids_of(hits: list[Hit]) -> list[str]:
    return [hit.chunk_id for hit in hits]


# --- the parser -------------------------------------------------------------


def test_a_plain_json_object_parses_into_a_step() -> None:
    step, problem = parse_step(reply("search", "festive quarter campaign", "I need the campaign"))

    assert problem is None
    assert step.action == "search"
    assert step.argument == "festive quarter campaign"
    assert step.thought == "I need the campaign"


def test_an_object_wrapped_in_prose_and_a_code_fence_still_parses() -> None:
    raw = 'Here you go:\n```json\n{"thought": "t", "action": "finish", "argument": "done"}\n```'

    step, problem = parse_step(raw)

    assert problem is None
    assert step.action == "finish"


def test_a_reply_with_no_json_object_is_a_parse_failure() -> None:
    step, problem = parse_step("I will now search for the campaign.")

    assert step is None
    assert problem.kind == "parse_failures"


def test_an_action_the_loop_does_not_know_is_refused_by_name() -> None:
    step, problem = parse_step(reply("read_document", "finance/secret"))

    assert step is None
    assert problem.kind == "unknown_actions"
    assert "read_document" in problem.message


def test_a_search_with_an_empty_query_is_refused() -> None:
    step, problem = parse_step(reply("search", "   "))

    assert step is None
    assert problem.kind == "empty_searches"


def test_a_finish_with_an_empty_argument_is_a_step_and_not_a_problem() -> None:
    # An empty answer is a refusal downstream (`prompting.is_refusal`), not a
    # protocol error, so the loop should end rather than retry.
    step, problem = parse_step(reply("finish", ""))

    assert problem is None
    assert step.action == "finish"
    assert step.argument == ""


def test_queries_differing_only_in_case_and_spacing_are_the_same_query() -> None:
    assert normalise_query("  Festive   Quarter ") == normalise_query("festive quarter")


# --- the prompt -------------------------------------------------------------


def test_the_prompt_never_names_the_role() -> None:
    """The function is not given a role, so this pins the shape, not a filter."""
    observations = [hit_for(A, 1), hit_for(C, 2)]
    hops = [{"thought": "t", "query": "festive campaign", "numbers": [1, 2]}]

    prompt = build_agent_prompt(QUESTION, hops, observations, 2)

    # Whole words: "three" contains "hr", and a substring match would fail on
    # corpus text that has nothing to do with a role.
    for role in ROLES:
        assert re.search(rf"\b{role}\b", prompt) is None


def test_observations_are_numbered_once_across_all_hops() -> None:
    observations = [hit_for(A, 1), hit_for(B, 2), hit_for(C, 3)]
    hops = [
        {"thought": "t1", "query": "one", "numbers": [1, 2]},
        {"thought": "t2", "query": "two", "numbers": [3]},
    ]

    prompt = build_agent_prompt(QUESTION, hops, observations, 1)

    assert "[1] section: one" in prompt
    assert "[2] section: two" in prompt
    assert "[3] section: three" in prompt


def test_the_retry_prompt_quotes_the_problem_back() -> None:
    prompt = build_agent_prompt(QUESTION, [], [], 4, "your reply was not valid JSON")

    assert "your reply was not valid JSON" in prompt
    assert "only retry" in prompt


def test_the_last_prompt_offers_no_search_and_says_so() -> None:
    prompt = build_agent_prompt(QUESTION, [], [], 0)

    assert "no searches left" in prompt
    assert '- action "search"' not in prompt


# --- the loop ---------------------------------------------------------------


def test_two_searches_then_a_finish_is_two_hops_and_three_calls() -> None:
    agent, llm = agent_for(
        [
            reply("search", "festive campaign"),
            reply("search", "regional churn"),
            reply("finish", "the campaign covered three regions [1]"),
        ],
        base=PlannedRetriever(CORPUS, {"festive campaign": [A], "regional churn": [C]}),
    )

    run = agent.run(QUESTION, "marketing")

    assert run.stats["hops"] == 2
    assert run.stats["llm_calls"] == 3
    assert run.hop_queries == ["festive campaign", "regional churn"]
    assert ids_of(run.observations) == [A.chunk_id, C.chunk_id]
    assert run.answer == "the campaign covered three regions [1]"
    assert len(llm.calls) == 3


def test_the_second_hops_prompt_carries_the_first_hops_observations() -> None:
    agent, llm = agent_for(
        [reply("search", "festive campaign"), reply("finish", "done [1]")],
        base=PlannedRetriever(CORPUS, {"festive campaign": [A]}),
    )

    agent.run(QUESTION, "marketing")

    second_prompt = llm.calls[1][0]
    assert "Hop 1" in second_prompt
    assert "festive quarter campaign" in second_prompt


def test_garbage_twice_in_a_row_forces_a_finish() -> None:
    agent, llm = agent_for(["I will search now.", "Still thinking about it."])

    run = agent.run(QUESTION, "marketing")

    assert run.stats["parse_failures"] == 2
    assert run.stats["forced_finishes"] == 1
    assert run.stats["hops"] == 0
    assert len(llm.calls) == 2
    assert run.answer is None


def test_one_bad_reply_followed_by_a_good_one_costs_a_retry_and_not_the_run() -> None:
    agent, _llm = agent_for(
        ["not json at all", reply("search", "festive campaign"), reply("finish", "done [1]")],
        base=PlannedRetriever(CORPUS, {"festive campaign": [A]}),
    )

    run = agent.run(QUESTION, "marketing")

    assert run.stats["parse_failures"] == 1
    assert run.stats["forced_finishes"] == 0
    assert run.stats["hops"] == 1
    assert run.answer == "done [1]"


def test_asking_the_same_query_twice_is_counted_and_does_not_buy_a_hop() -> None:
    agent, _llm = agent_for(
        [
            reply("search", "festive campaign"),
            reply("search", "Festive  Campaign"),
            reply("finish", "done [1]"),
        ],
        base=PlannedRetriever(CORPUS, {"festive campaign": [A]}),
    )

    run = agent.run(QUESTION, "marketing")

    assert run.stats["repeated_queries"] == 1
    assert run.stats["hops"] == 1
    assert run.hop_queries == ["festive campaign"]


def test_the_loop_stops_at_the_hop_cap_and_spends_one_last_call_on_an_answer() -> None:
    agent, llm = agent_for(
        [
            reply("search", "one"),
            reply("search", "two"),
            reply("search", "three"),
            reply("search", "four"),
            reply("finish", "out of road [1]"),
        ],
        base=PlannedRetriever(CORPUS, default=[A]),
    )

    run = agent.run(QUESTION, "marketing")

    assert run.stats["hops"] == MAX_HOPS
    assert run.stats["llm_calls"] == MAX_HOPS + 1
    assert run.stats["forced_finishes"] == 1
    assert run.answer == "out of road [1]"
    assert len(llm.calls) == MAX_HOPS + 1


def test_a_search_asked_for_after_the_cap_is_not_honoured() -> None:
    # The script never finishes, so the forced call gets a search back. We do not
    # run it and we do not pay for another call to ask again.
    agent, llm = agent_for(
        [
            reply("search", "one"),
            reply("search", "two"),
            reply("search", "three"),
            reply("search", "four"),
            reply("search", "five"),
        ],
        base=PlannedRetriever(CORPUS, default=[A]),
    )

    run = agent.run(QUESTION, "marketing")

    assert run.answer is None
    assert run.stats["unknown_actions"] == 1
    assert len(llm.calls) == MAX_HOPS + 1


def test_a_chunk_found_in_an_earlier_hop_is_not_observed_twice() -> None:
    agent, llm = agent_for(
        [
            reply("search", "one"),
            reply("search", "two"),
            reply("finish", "done [1][2]"),
        ],
        base=PlannedRetriever(CORPUS, {"one": [A, B], "two": [B, C]}),
    )

    run = agent.run(QUESTION, "marketing")

    assert ids_of(run.observations) == [A.chunk_id, B.chunk_id, C.chunk_id]
    # The second hop's prompt must not show B again either.
    assert llm.calls[2][0].count("brand colours and typeface") == 1


def test_a_hop_that_finds_nothing_new_says_so_in_the_next_prompt() -> None:
    agent, llm = agent_for(
        [reply("search", "one"), reply("search", "two"), reply("finish", "done [1]")],
        base=PlannedRetriever(CORPUS, {"one": [A], "two": [A]}),
    )

    agent.run(QUESTION, "marketing")

    assert "observations: nothing new" in llm.calls[2][0]


def test_a_backend_that_raises_ends_as_a_forced_finish_and_not_an_exception() -> None:
    class RaisingLLM:
        name = "raising"

        def generate(self, prompt: str, system: str | None = None) -> str:
            raise RuntimeError("no credentials")

    agent = ReactAgent(CORPUS, PlannedRetriever(CORPUS), RaisingLLM())

    run = agent.run(QUESTION, "marketing")

    assert run.stats["call_failures"] == 2
    assert run.stats["forced_finishes"] == 1
    assert agent.last_failure.startswith("call failed: RuntimeError")


def test_a_loop_that_observes_nothing_falls_back_to_one_plain_search() -> None:
    """A broken model must not turn retrieval off. This is the floor."""
    base = PlannedRetriever(CORPUS, default=[C, A])
    agent, _llm = agent_for(["garbage", "more garbage"], base=base)

    run = agent.run(QUESTION, "marketing")

    assert run.stats["fallback_searches"] == 1
    assert base.queries == [QUESTION]
    assert ids_of(run.observations) == [C.chunk_id, A.chunk_id]
    assert run.answer is None


def test_the_agent_asks_the_base_retriever_for_its_own_hop_size() -> None:
    base = PlannedRetriever(CORPUS, default=[A, B, C])
    agent, _llm = agent_for([reply("finish", "nothing to say")], base=base, hop_k=2)

    run = agent.run(QUESTION, "marketing")

    # The fallback search ran with hop_k, so two of the three visible came back.
    assert len(run.observations) == 2


# --- the ranking ------------------------------------------------------------


def test_only_the_numbers_the_answer_really_cites_are_counted() -> None:
    assert cited_numbers("first [2] then [9] and [2] again", 3) == [2]
    assert cited_numbers(None, 3) == []


def test_cited_observations_come_first_in_citation_order() -> None:
    observations = [hit_for(A, 1), hit_for(B, 2), hit_for(C, 3)]

    ordered = rank_observations(observations, [3, 1])

    assert ids_of(ordered) == [C.chunk_id, A.chunk_id, B.chunk_id]


def test_citation_numbers_are_rewritten_onto_the_ranks_the_caller_sees() -> None:
    observations = [hit_for(A, 1), hit_for(B, 2), hit_for(C, 3)]
    hits = [hit_for(C, 1), hit_for(A, 2)]

    rewritten = renumber_citations("churn [3] and campaign [1], see [2]", observations, hits)

    # [3] was C, now rank 1. [1] was A, now rank 2. [2] was B, which did not
    # survive the cut, so the bracket goes rather than pointing at the wrong hit.
    assert rewritten == "churn [1] and campaign [2], see "


# --- the retriever ----------------------------------------------------------


def react_retriever(replies: list[str], base=None, **kwargs) -> ReactRetriever:
    agent, _llm = agent_for(replies, base=base, **kwargs)
    return ReactRetriever(CORPUS, agent)


def test_the_returned_hits_are_the_union_of_every_hop_ranked_from_one() -> None:
    retriever = react_retriever(
        [
            reply("search", "one"),
            reply("search", "two"),
            reply("finish", "the campaign [3] answers the churn question [1]"),
        ],
        base=PlannedRetriever(CORPUS, {"one": [B, C], "two": [A]}),
    )

    hits, answer = retriever.search_with_answer(QUESTION, "marketing", k=10)

    # Observations were B, C, A. The answer cites [3] then [1], so A then B, and
    # C follows in the order it was found.
    assert ids_of(hits) == [A.chunk_id, B.chunk_id, C.chunk_id]
    assert [hit.rank for hit in hits] == [1, 2, 3]
    assert answer == "the campaign [1] answers the churn question [2]"


def test_the_union_is_cut_to_k_after_the_cited_chunks() -> None:
    retriever = react_retriever(
        [reply("search", "one"), reply("finish", "see [2]")],
        base=PlannedRetriever(CORPUS, {"one": [A, B, C]}),
    )

    hits = retriever.search(QUESTION, "marketing", k=2)

    assert ids_of(hits) == [B.chunk_id, A.chunk_id]


def test_the_hit_debug_carries_the_hop_queries_and_the_stats() -> None:
    retriever = react_retriever(
        [reply("search", "festive campaign"), reply("finish", "done [1]")],
        base=PlannedRetriever(CORPUS, {"festive campaign": [A]}),
    )

    hits = retriever.search(QUESTION, "marketing", k=5)

    assert hits[0].retriever == "react"
    assert hits[0].debug["hop_queries"] == ["festive campaign"]
    assert hits[0].debug["agent"]["llm_calls"] == 2
    assert hits[0].debug["cited"] is True


def test_the_totals_add_up_over_every_question_the_process_asked() -> None:
    # One script, two runs: the scripted backend keeps its place, so the second
    # question reads the third and fourth replies.
    retriever = react_retriever(
        [
            reply("search", "one"),
            reply("finish", "done [1]"),
            reply("search", "one"),
            reply("finish", "done [1]"),
        ],
        base=PlannedRetriever(CORPUS, {"one": [A]}),
    )

    retriever.search(QUESTION, "marketing", k=5)
    retriever.search(QUESTION, "marketing", k=5)

    assert retriever.totals["questions"] == 2
    assert retriever.totals["llm_calls"] == 4
    assert retriever.totals["hops"] == 2


def test_a_role_that_may_see_nothing_gets_nothing_and_costs_no_call() -> None:
    agent, llm = agent_for([reply("search", "anything")])
    retriever = ReactRetriever(CORPUS, agent)

    assert retriever.search(QUESTION, "engineering", k=10) == []
    assert llm.calls == []


# --- the invariant: the mask stands in front of the model -------------------


def test_a_compromised_base_retriever_cannot_get_a_denied_chunk_into_a_prompt() -> None:
    """The sharper half of the invariant: the model must not even read it."""
    agent, llm = agent_for(
        [reply("search", "the risk"), reply("finish", "done [1]")],
        base=LeakyRetriever(CORPUS),
    )

    agent.run(QUESTION, "marketing")

    assert llm.calls != []
    for prompt, system in llm.calls:
        assert SECRET_TEXT not in prompt
        assert SECRET.chunk_id not in prompt
        assert system == SYSTEM_PROMPT


def test_a_compromised_base_retriever_cannot_get_a_denied_chunk_into_the_hits() -> None:
    """k is 10, wider than the visible pool of three, so no cut-off hides it."""
    retriever = react_retriever(
        [reply("search", "the risk"), reply("finish", "done [1]")],
        base=LeakyRetriever(CORPUS),
    )

    hits = retriever.search(QUESTION, "marketing", k=10)

    assert SECRET.chunk_id not in ids_of(hits)
    assert [hit.rank for hit in hits] == [1, 2, 3]


def test_the_last_lock_alone_scrubs_a_denied_observation(monkeypatch) -> None:
    """Deleting the lock in `search_with_answer` must fail a test, not a comment.

    The observe mask normally makes the last lock unreachable, so every other
    mask test would still pass without it. This one forges a run whose
    observations already hold the denied chunk, the way a bug between `_observe`
    and the return would, and proves the lock alone scrubs it.
    """
    retriever = react_retriever([reply("finish", "done")])
    forged = AgentRun(
        question=QUESTION,
        answer="done [1]",
        observations=[hit_for(SECRET, 1), hit_for(A, 2)],
        hop_queries=["the risk"],
        stats=new_stats(),
    )
    monkeypatch.setattr(retriever.agent, "run", lambda question, role: forged)

    hits = retriever.search(QUESTION, "marketing", k=10)

    assert SECRET.chunk_id not in ids_of(hits)
    assert ids_of(hits) == [A.chunk_id]


def test_the_denied_chunk_is_counted_as_masked_rather_than_dropped_in_silence() -> None:
    agent, _llm = agent_for(
        [reply("search", "the risk"), reply("finish", "done [1]")],
        base=LeakyRetriever(CORPUS),
    )

    run = agent.run(QUESTION, "marketing")

    assert run.stats["masked_out"] == 1


def test_the_real_loop_on_the_micro_corpus_stays_inside_what_marketing_may_see(
    micro_corpus,
) -> None:
    """The second mask test: a real hybrid tool, a real loop, the real fixture."""
    allowed = visible_ids(micro_corpus, "marketing")
    base = HybridRetriever(
        micro_corpus,
        BM25Retriever(micro_corpus),
        DenseRetriever(micro_corpus, FakeEmbedder(dim=32)),
    )
    agent = ReactAgent(
        micro_corpus,
        base,
        ScriptedLLM(
            [
                reply("search", "campaign spend and impressions"),
                reply("search", "Q4 revenue and the leading risk"),
                reply("finish", "spend was 2.0 million dollars [1]"),
            ]
        ),
    )
    retriever = ReactRetriever(micro_corpus, agent)

    run = agent.run("what did the campaign cost", "marketing")
    hits = retriever.search("what did the campaign cost", "marketing", k=10)

    assert run.observations != []
    for hit in run.observations:
        assert hit.chunk_id in allowed
    assert hits != []
    for hit in hits:
        assert hit.chunk_id in allowed


# --- the registry -----------------------------------------------------------


def test_the_registry_builds_the_react_mode_from_a_bundle() -> None:
    bundle = IndexBundle(chunks=CORPUS, embedder=FakeEmbedder(dim=32), llm=FakeLLM())

    retriever = build_retriever("react", bundle)

    assert retriever.name == "react"
    assert SECRET.chunk_id not in ids_of(retriever.search(QUESTION, "marketing", k=10))


def test_the_react_mode_searches_with_the_hybrid_retriever() -> None:
    bundle = IndexBundle(chunks=CORPUS, embedder=FakeEmbedder(dim=32), llm=FakeLLM())

    retriever = build_retriever("react", bundle)

    assert retriever.agent.base.name == "hybrid"


def test_the_react_mode_refuses_a_bundle_with_no_model() -> None:
    bundle = IndexBundle(chunks=CORPUS, embedder=FakeEmbedder(dim=32))

    with pytest.raises(ValueError, match="llm"):
        build_retriever("react", bundle)


def test_the_full_list_of_modes_is_what_this_rung_left_behind() -> None:
    """Pinned in the newest mode's file, as R5 pinned it in the graph tests."""
    assert available_modes() == [
        "bm25",
        "dense",
        "graphrag",
        "hybrid",
        "hybrid_rerank",
        "naive",
        "react",
    ]
