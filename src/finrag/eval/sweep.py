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

The whole sweep runs offline. Dense modes read the embedding cache ingest filled,
and the reranker is built on `FallbackOnlyLLM`, a backend that always fails, so
`LLMReranker` falls back to the fused order and calls nothing. The sweep is about
the mask, not about ranking quality, and every mask in `RerankRetriever.search`
still runs on the fallback path.

One number matters: `violations` must be 0. Everything else in the report is
there so a reader can see that the check really ran, and how wide.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from finrag.config import Settings
from finrag.ingest.pipeline import read_chunks
from finrag.models import ROLES
from finrag.policy.engine import PolicyEngine
from finrag.retrieval.base import Retriever
from finrag.retrieval.embed import build_embedder
from finrag.retrieval.registry import IndexBundle, available_modes, build_retriever

# Modes that need to embed the query text. They read the on-disk embedding cache
# first, so a sweep over questions a previous eval already embedded costs nothing.
MODES_NEEDING_AN_EMBEDDER: frozenset[str] = frozenset({"dense", "hybrid", "hybrid_rerank"})

# A broken retriever can leak on every one of the fifteen hundred cells, and a
# report with fifteen thousand violation records is unreadable and unwritable.
# The COUNT is always exact; only the recorded examples are capped.
MAX_RECORDED_VIOLATIONS = 100


class FallbackOnlyLLM:
    """A generation backend that always fails, so a reranker falls back.

    Not a mock for a test: this is what the sweep uses on purpose. `LLMReranker`
    counts a failure and returns its input order unchanged, which keeps the sweep
    free and offline while still running every mask in `RerankRetriever.search`.
    """

    name = "fallback-only"

    def generate(self, prompt: str, system: str | None = None) -> str:
        raise RuntimeError("the leakage sweep runs offline: no rerank model is called")


def build_offline_retrievers(
    settings: Settings,
    modes: Sequence[str] | None = None,
) -> dict[str, Retriever]:
    """One retriever per mode, all of them runnable with no network.

    The artifacts are loaded once and shared, so every mode sees exactly the
    chunk set and the indexes `finrag eval` sees. That matters: a sweep over a
    different corpus than the eval would prove nothing about the eval.
    """
    if modes is None:
        modes = available_modes()
    modes = tuple(modes)

    bundle = IndexBundle(
        chunks=read_chunks(settings.chunks_path),
        bm25_dir=settings.bm25_dir,
        dense_path=settings.dense_path,
        graph_path=settings.graph_path,
    )
    for mode in modes:
        if mode in MODES_NEEDING_AN_EMBEDDER:
            bundle.embedder = build_embedder(settings)
            break
    bundle.llm = FallbackOnlyLLM()

    retrievers: dict[str, Retriever] = {}
    for mode in modes:
        retrievers[mode] = build_retriever(mode, bundle)
    return retrievers


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
