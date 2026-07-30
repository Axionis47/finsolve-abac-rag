"""The leakage sweep: every gold question, asked in every role, in every mode.

`harness.py` asks each gold question in the ONE role that question was written
for. That is the right thing for scoring, and the wrong thing for proving the
role mask. A question written for `marketing` never gets asked as `employee`
there, so a retriever that leaked to `employee` alone would score a clean run.

This module closes that hole by brute force. It takes the question TEXT out of
every gold entry, throws away the role the entry names, and asks that text in all
six policy roles against every retrieval mode. Each returned hit is then replayed
through `PolicyEngine.replay`, which re-decides it from the rules rather than
trusting the `allowed_roles` stamped on the chunk at ingest. So a stale stamp is
caught here instead of hidden here.

Every mode but one runs offline. Dense modes read the embedding cache ingest
filled, and the reranker is built on `FallbackOnlyLLM`, a backend that always
fails, so `LLMReranker` falls back to the fused order and calls nothing. The
sweep is about the mask, not about ranking quality, and every mask in
`RerankRetriever.search` still runs on the fallback path.

`react` gets the same backend and the same treatment. Its two model calls fail,
the loop forces a finish, and the fallback search in `ReactAgent.run` puts one
plain search of the question through the union, the ranking and the last lock.
So the mode is swept for free, and what is swept is the plumbing rather than a
real multi-hop plan. The multi-hop path is proved instead by `tests/test_react.py`
(a real loop over the micro corpus, every observation checked against
`visible_ids`) and by the r7 eval, which ran the real model over fifty questions
with leakage 0. Recorded here because a reader owes it to themselves to know
which half of a mode a gate covers.

The exception is `graphrag`, whose mask lives inside Cypher and therefore needs
the Neo4j container to be proved at all. It is swept when bolt answers and
dropped with a printed note when it does not, so a laptop with no container still
gets the other five modes checked rather than an error. Asking for it by name
with the container down is a refusal, not a silent skip: `--modes graphrag` is a
person saying they want that mask tested.

One number matters: `violations` must be 0. Everything else in the report is
there so a reader can see that the check really ran, and how wide.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from finrag.config import Settings
from finrag.graph.store import bolt_is_open
from finrag.ingest.pipeline import read_chunks
from finrag.models import ROLES
from finrag.policy.engine import PolicyEngine
from finrag.retrieval.base import Retriever
from finrag.retrieval.embed import Embedder, build_embedder
from finrag.retrieval.registry import IndexBundle, available_modes, build_retriever

# Modes that need to embed the query text. They read the on-disk embedding cache
# first, so a sweep over questions a previous eval already embedded costs nothing.
MODES_NEEDING_AN_EMBEDDER: frozenset[str] = frozenset(
    {"dense", "hybrid", "hybrid_rerank", "react"}
)

# Modes that cannot run without the Neo4j container.
MODES_NEEDING_A_GRAPH: frozenset[str] = frozenset({"graphrag"})

# A broken retriever can leak on every one of the fifteen hundred cells, and a
# report with fifteen thousand violation records is unreadable and unwritable.
# The COUNT is always exact; only the recorded examples are capped.
MAX_RECORDED_VIOLATIONS = 100


class FallbackOnlyLLM:
    """A generation backend that always fails, so a model-using mode falls back.

    Not a mock for a test: this is what the sweep uses on purpose. `LLMReranker`
    counts a failure and returns its input order unchanged; `ReactAgent` counts a
    failure, forces a finish and runs its fallback search. Both keep the sweep
    free and offline while still running every mask in the mode.
    """

    name = "fallback-only"

    def generate(self, prompt: str, system: str | None = None) -> str:
        raise RuntimeError("the leakage sweep runs offline: no model is called")


class CacheOnlyEmbedder:
    """Answers from the on-disk embedding cache, and refuses to call an API.

    Why this exists rather than trusting `build_embedder`: the sweep's "offline"
    was luck, not structure. `build_embedder` defaults to the real Vertex backend
    behind the cache, so the 1,800-cell gate ran free only because the 50 gold
    questions happened to be in `embed_cache.npz` from an earlier eval. One new
    gold question, or a cleared artifacts folder, and the free gate would have
    quietly started billing per query.

    So a miss is an error naming the question, not a call. The fix it asks for is
    one eval run, which pays for each question once and leaves it cached.
    """

    def __init__(self, cached: Any) -> None:
        self.cache = cached.cache
        self.name = f"cache-only-{cached.name}"
        self.model = cached.model
        self.dim = cached.dim

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            vector = self.cache.get(text)
            if vector is None:
                raise RuntimeError(
                    "The leakage sweep is offline by design, and there is no cached "
                    f"'{self.model}' vector for this text:\n  {text[:120]!r}\n"
                    "Run `finrag eval --mode dense` once to fill the cache, or set "
                    "FINRAG_EMBEDDER=fake to sweep with offline vectors."
                )
            out[row] = vector
        return out


def build_sweep_embedder(settings: Settings) -> Embedder:
    """The query embedder for a sweep: the cache, with nothing behind it that bills.

    The fake backend is handed back as it is. It costs nothing, needs no cache,
    and a sweep on a machine with no credentials is exactly what it is for.
    """
    embedder = build_embedder(settings)
    if settings.embedder == "fake":
        return embedder
    return CacheOnlyEmbedder(embedder)


def offline_modes() -> list[str]:
    """Every registry mode that needs no container and no network."""
    modes: list[str] = []
    for mode in available_modes():
        if mode not in MODES_NEEDING_A_GRAPH:
            modes.append(mode)
    return modes


def sweepable_modes() -> list[str]:
    """Every mode this machine can sweep right now, graph modes included if bolt answers.

    One bolt check, here, rather than one per mode: the answer is the same for
    all of them and a connection attempt per mode would slow a laptop sweep down
    for nothing.
    """
    if bolt_is_open():
        return list(available_modes())
    return offline_modes()


def build_sweep_retrievers(
    settings: Settings,
    modes: Sequence[str] | None = None,
) -> dict[str, Retriever]:
    """One retriever per mode, ready to sweep.

    The artifacts are loaded once and shared, so every mode sees exactly the
    chunk set and the indexes `finrag eval` sees. That matters: a sweep over a
    different corpus than the eval would prove nothing about the eval.

    With no `modes` given this sweeps everything the machine can run. A mode named
    explicitly is always built, and a graph mode named with the container down
    raises rather than quietly dropping the one mask the caller asked about.
    """
    if modes is None:
        modes = sweepable_modes()
    modes = tuple(modes)
    _check_graph_modes(modes)

    bundle = IndexBundle(
        chunks=read_chunks(settings.chunks_path),
        bm25_dir=settings.bm25_dir,
        dense_path=settings.dense_path,
        graph_path=settings.graph_path,
    )
    for mode in modes:
        if mode in MODES_NEEDING_AN_EMBEDDER:
            bundle.embedder = build_sweep_embedder(settings)
            break
    bundle.llm = FallbackOnlyLLM()

    retrievers: dict[str, Retriever] = {}
    for mode in modes:
        retrievers[mode] = build_retriever(mode, bundle)
    return retrievers


def _check_graph_modes(modes: Sequence[str]) -> None:
    """Refuse a graph mode the container cannot serve, and say how to start it."""
    wanted: list[str] = []
    for mode in modes:
        if mode in MODES_NEEDING_A_GRAPH:
            wanted.append(mode)
    if not wanted or bolt_is_open():
        return
    raise RuntimeError(
        f"The {', '.join(wanted)} mode needs Neo4j, and bolt is not answering. "
        "Start it with `docker start finrag-neo4j`, or leave --modes off to sweep "
        "the modes that run without it."
    )


def _new_counts() -> dict[str, int]:
    """The four numbers we keep per role, per mode, and overall."""
    return {"cells": 0, "hits": 0, "violations": 0, "empty_cells": 0}


def _record(counts: dict[str, int], n_hits: int, n_violations: int) -> None:
    """Fold one cell's result into a counts block.

    An empty cell is not a failure. Most roles cannot see most of the corpus, so
    a role asked a question about a department it has no access to should return
    nothing. We count those separately to show the mask is doing work rather than
    the retriever quietly failing on every query.
    """
    counts["cells"] += 1
    counts["hits"] += n_hits
    counts["violations"] += n_violations
    if n_hits == 0:
        counts["empty_cells"] += 1


def _sum_counts(blocks: Sequence[dict[str, int]]) -> dict[str, int]:
    """Add several counts blocks together."""
    total = _new_counts()
    for block in blocks:
        for key in total:
            total[key] += block[key]
    return total


def run_leakage_sweep(
    retrievers: dict[str, Retriever],
    policy: PolicyEngine,
    gold: Sequence[dict[str, Any]],
    roles: Sequence[str] = ROLES,
    k: int = 10,
) -> dict[str, Any]:
    """Ask every question in every role in every mode, and replay every hit.

    Returns the report dict. `violations` is the headline and must be 0;
    `violation_details` names the (question, role, mode, chunk) tuples behind it,
    capped at `MAX_RECORDED_VIOLATIONS` records.
    """
    modes = sorted(retrievers)
    roles = tuple(roles)

    breakdown: dict[str, dict[str, dict[str, int]]] = {}
    for role in roles:
        breakdown[role] = {}
        for mode in modes:
            breakdown[role][mode] = _new_counts()

    details: list[dict[str, Any]] = []
    n_violations = 0

    for entry in gold:
        for role in roles:
            for mode in modes:
                hits = retrievers[mode].search(entry["question"], role, k=k)
                # Re-decide from the rules. Never trust the ingest-time stamp,
                # because a wrong stamp is one of the things this sweep is for.
                found = policy.replay(role, hits)

                _record(breakdown[role][mode], len(hits), len(found))
                n_violations += len(found)
                for violation in found:
                    if len(details) < MAX_RECORDED_VIOLATIONS:
                        details.append(
                            {
                                "question_id": entry["id"],
                                "gold_role": entry["role"],
                                "asked_as_role": role,
                                "mode": mode,
                                "chunk_id": violation.chunk_id,
                                "denied_by_rule": violation.rule_name,
                            }
                        )

    per_role: dict[str, dict[str, int]] = {}
    for role in roles:
        per_role[role] = _sum_counts(list(breakdown[role].values()))

    per_mode: dict[str, dict[str, int]] = {}
    for mode in modes:
        columns = []
        for role in roles:
            columns.append(breakdown[role][mode])
        per_mode[mode] = _sum_counts(columns)

    totals = _sum_counts(list(per_role.values()))

    return {
        "k": k,
        "roles": list(roles),
        "modes": modes,
        "n_questions": len(gold),
        "cells_checked": totals["cells"],
        "hits_inspected": totals["hits"],
        "violations": n_violations,
        "violations_recorded": len(details),
        "empty_cells": totals["empty_cells"],
        "rerank": "fallback only (offline stub backend, no api calls)",
        "react": "forced finish then one fallback search (offline stub backend, no api calls)",
        "per_role": per_role,
        "per_mode": per_mode,
        "breakdown": breakdown,
        "violation_details": details,
    }


def sweep_passed(report: dict[str, Any]) -> bool:
    """True when no denied chunk reached any result list anywhere."""
    return report["violations"] == 0


def breakdown_lines(report: dict[str, Any]) -> list[str]:
    """The per-role by per-mode table, as lines ready to print.

    Each cell is `hits/violations`, because a column of zero violations only
    means something next to the number of hits it was drawn from. A role and mode
    that returned nothing at all cannot leak and has not been tested.
    """
    modes = report["modes"]
    width = 16

    header = f"{'role':<12}"
    for mode in modes:
        header += f"{mode:>{width}}"
    header += f"{'total hits':>12}{'leaks':>8}"

    lines = [header]
    for role in report["roles"]:
        row = f"{role:<12}"
        for mode in modes:
            cell = report["breakdown"][role][mode]
            row += f"{str(cell['hits']) + '/' + str(cell['violations']):>{width}}"
        totals = report["per_role"][role]
        row += f"{totals['hits']:>12}{totals['violations']:>8}"
        lines.append(row)

    footer = f"{'ALL':<12}"
    for mode in modes:
        cell = report["per_mode"][mode]
        footer += f"{str(cell['hits']) + '/' + str(cell['violations']):>{width}}"
    footer += f"{report['hits_inspected']:>12}{report['violations']:>8}"
    lines.append(footer)
    return lines
