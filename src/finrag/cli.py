"""The `finrag` command line: one subcommand per stage of the build.

This is the front door a person uses. `finrag ingest` runs the ingest pipeline
and prints what it wrote. `finrag eval` scores a retrieval mode against the gold
set and saves the report. `finrag sweep` runs the ABAC leakage sweep over every
question, role and mode. The remaining subcommands exist so the installed
console script has a complete shape from day one; each prints which rung of the
build will fill it in and exits non-zero, so nothing mistakes a stub for work.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from finrag.config import Settings, load_env
from finrag.eval.goldset import GoldSetError, load_gold, validate
from finrag.eval.harness import EvalHarness
from finrag.eval.report_io import write_report
from finrag.eval.sweep import (
    breakdown_lines,
    build_offline_retrievers,
    run_leakage_sweep,
    sweep_passed,
)
from finrag.ingest.pipeline import IngestStats, read_chunks, run_ingest
from finrag.llm.vertex import build_llm
from finrag.policy.engine import PolicyEngine
from finrag.retrieval.embed import build_embedder
from finrag.retrieval.registry import IndexBundle, available_modes, build_retriever

# Modes that need to embed a query, so the CLI knows when to build an embedder.
# Building one may open a Vertex client, and naive and bm25 should not pay for it.
MODES_NEEDING_AN_EMBEDDER: frozenset[str] = frozenset({"dense", "hybrid", "hybrid_rerank"})

# Modes that call a generation model. Only the reranker does, and it makes one
# real API call per question, so no other mode should build a client.
MODES_NEEDING_AN_LLM: frozenset[str] = frozenset({"hybrid_rerank"})

# Subcommands that are not built yet, and the rung that will build them.
PENDING_COMMANDS: dict[str, str] = {
    "graph": "R5 (graph schema and the structural or LLM builds)",
    "serve": "R6 (the FastAPI app and the UI)",
}


def build_parser() -> argparse.ArgumentParser:
    """Define the command line: `finrag <command>`, one command required."""
    parser = argparse.ArgumentParser(
        prog="finrag",
        description="FinSolve ABAC RAG: ingest the corpus, build the graph, eval, serve.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "ingest",
        help="chunk the corpus, write artifacts/chunks.jsonl, the BM25 index and dense.npz",
    )
    subparsers.add_parser("graph", help="build the knowledge graph (not wired up yet)")

    eval_parser = subparsers.add_parser("eval", help="score a retrieval mode on the gold set")
    eval_parser.add_argument(
        "--mode",
        default="naive",
        help=f"retrieval mode to score. One of: {', '.join(available_modes())}",
    )
    eval_parser.add_argument("--rung", default="r0", help="build rung, used in the report name")
    eval_parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[5, 10],
        help="cut-offs to score at (default: 5 10)",
    )

    sweep_parser = subparsers.add_parser(
        "sweep",
        help="ABAC leakage sweep: every gold question, in every role, in every mode",
    )
    sweep_parser.add_argument("--rung", default="r4", help="build rung, used in the report name")
    sweep_parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="how many hits to ask each mode for (default: 10)",
    )
    sweep_parser.add_argument(
        "--modes",
        nargs="+",
        default=None,
        help=f"modes to sweep (default: all of {', '.join(available_modes())})",
    )

    subparsers.add_parser("serve", help="run the API (not wired up yet)")

    return parser


def command_ingest() -> int:
    """Run ingest against the configured corpus and print the result."""
    settings = Settings.from_env()
    stats = run_ingest(settings)
    print_stats(stats)
    return 0


def command_eval(mode: str, rung: str, k_values: list[int]) -> int:
    """Score one retrieval mode on the gold set and save the report.

    Straight line: load the artifacts, build the retriever, run the harness,
    write the JSON, print the per-tier table.
    """
    settings = Settings.from_env()

    if not settings.chunks_path.exists():
        print(f"No chunks at {settings.chunks_path}. Run `finrag ingest` first.")
        return 1
    if not settings.gold_path.exists():
        print(f"No gold set at {settings.gold_path}.")
        return 1
    if mode not in available_modes():
        print(f"Unknown mode '{mode}'. Known modes: {', '.join(available_modes())}")
        return 1

    bundle = IndexBundle(
        chunks=read_chunks(settings.chunks_path),
        bm25_dir=settings.bm25_dir,
        dense_path=settings.dense_path,
        graph_path=settings.graph_path,
    )
    if mode in MODES_NEEDING_AN_EMBEDDER:
        # Queries go through the same cache ingest filled, so a repeated run of
        # this eval embeds nothing and makes no API calls.
        bundle.embedder = build_embedder(settings)
    if mode in MODES_NEEDING_AN_LLM:
        bundle.llm = build_llm(settings)
    retriever = build_retriever(mode, bundle)
    policy = PolicyEngine.from_yaml(settings.policy_path)

    # The gold set points at the corpus by name, so a renamed heading or a
    # dropped document turns every score into nonsense without failing. Check it
    # against the chunks we just loaded and the real policy before scoring
    # anything, and refuse to publish numbers from a gold set that has drifted.
    try:
        gold = load_gold(settings.gold_path)
        validate(gold, settings.chunks_path, policy)
    except GoldSetError as problem:
        print(problem)
        return 1

    report = EvalHarness(retriever, policy, gold.entries).run(k_values=tuple(k_values))
    add_rerank_counters(report, retriever)
    path = write_report(report, rung, mode, settings.results_dir)

    print_report(report, k_values, path)
    return 0


def command_sweep(rung: str, k: int, modes: list[str] | None = None) -> int:
    """Run the leakage sweep, print the table, and fail on any violation.

    Exits 1 when a denied chunk reached any result list, so a CI run or a shell
    `&&` chain stops there. That is the point of the command: it is a gate, not a
    report you read and shrug at.
    """
    settings = Settings.from_env()

    if not settings.chunks_path.exists():
        print(f"No chunks at {settings.chunks_path}. Run `finrag ingest` first.")
        return 1
    if not settings.gold_path.exists():
        print(f"No gold set at {settings.gold_path}.")
        return 1
    for mode in modes or []:
        if mode not in available_modes():
            print(f"Unknown mode '{mode}'. Known modes: {', '.join(available_modes())}")
            return 1

    policy = PolicyEngine.from_yaml(settings.policy_path)
    try:
        gold = load_gold(settings.gold_path)
        validate(gold, settings.chunks_path, policy)
    except GoldSetError as problem:
        print(problem)
        return 1

    retrievers = build_offline_retrievers(settings, modes)
    report = run_leakage_sweep(retrievers, policy, gold.entries, k=k)
    path = write_report(report, rung, "sweep", settings.results_dir)

    print_sweep(report, path)
    return 0 if sweep_passed(report) else 1


def print_sweep(report: dict[str, Any], path: Path) -> None:
    """Print the role-by-mode table, then the headline counts."""
    print(
        f"questions={report['n_questions']}  roles={len(report['roles'])}  "
        f"modes={len(report['modes'])}  cells={report['cells_checked']}  k={report['k']}"
    )
    print()
    for line in breakdown_lines(report):
        print(line)
    print()
    print(f"cells checked:   {report['cells_checked']}")
    print(f"hits inspected:  {report['hits_inspected']}")
    print(f"empty cells:     {report['empty_cells']} (a role that may see nothing gets nothing)")
    print(f"VIOLATIONS:      {report['violations']} (target 0)")
    for violation in report["violation_details"]:
        print(
            f"  LEAK {violation['question_id']} as {violation['asked_as_role']} "
            f"via {violation['mode']}: {violation['chunk_id']} "
            f"(denied by {violation['denied_by_rule']})"
        )
    print()
    print(f"Wrote {path}")


def add_rerank_counters(report: dict[str, Any], retriever: Any) -> None:
    """Record how often the reranker fell back, when the mode has a reranker.

    A rerank run where the model failed on half the questions scores like half a
    rerank and reads like a whole one, so the count belongs in the saved report
    rather than only in the console. Modes without a reranker add nothing.
    """
    reranker = getattr(retriever, "reranker", None)
    if reranker is None:
        return
    report["rerank"] = {
        "llm_calls": reranker.calls,
        "fallbacks": reranker.fallbacks,
        "last_failure": reranker.last_failure,
        "depth": reranker.depth,
    }


def print_report(report: dict[str, Any], k_values: list[int], path: Path) -> None:
    """Print the per-tier table, then the headline numbers.

    Section level comes first because it is the headline. After the role mask
    most roles see only a handful of documents, so doc recall is nearly free and
    tells us little; the section columns are where retrievers differ.
    """
    top_k = max(k_values)
    print(f"mode={report['mode']}  questions={report['n_questions']}")
    print()

    header = f"{'tier':<8}{'n':>4}{'scored':>8}{'sec hit':>10}{'sec mrr':>10}"
    header += f"{'recall@' + str(top_k):>12}{'doc mrr':>10}{'leak':>6}"
    print(header)
    for tier in report["per_tier"]:
        _print_summary_row(tier, report["per_tier"][tier], top_k)
    _print_summary_row("OVERALL", report["overall"], top_k)

    print()
    ambiguous = report["ambiguous"]
    print(
        f"ambiguous both-sources recall@{top_k}: "
        f"{ambiguous[f'both_sources_recall@{top_k}']:.3f} "
        f"({ambiguous['n_pass']}/{ambiguous['n']})"
    )
    probes = report["policy_probes"]
    print(f"policy probes passed: {probes['n_pass']}/{probes['n']}")
    print(f"total leakage: {report['overall']['leakage_count']} (target 0)")
    if "rerank" in report:
        rerank = report["rerank"]
        print(
            f"rerank: {rerank['llm_calls']} llm calls, "
            f"{rerank['fallbacks']} fallbacks to the hybrid order"
        )
        if rerank["last_failure"]:
            print(f"last rerank failure: {rerank['last_failure']}")
    print()
    print(f"Wrote {path}")


def _fixed(value: float | None, width: int) -> str:
    """A number to three places, or a dash when the metric was not measured."""
    if value is None:
        return f"{'-':>{width}}"
    return f"{value:>{width}.3f}"


def _print_summary_row(label: str, summary: dict[str, Any], top_k: int) -> None:
    row = f"{label:<8}{summary['n']:>4}{summary['n_scored']:>8}"
    row += _fixed(summary["section_hit_rate"], 10)
    row += _fixed(summary["section_mrr"], 10)
    row += _fixed(summary["recall@" + str(top_k)], 12)
    row += _fixed(summary["mrr"], 10)
    row += f"{summary['leakage_count']:>6}"
    print(row)


def command_pending(name: str) -> int:
    """Explain that a subcommand is a placeholder, and which rung fills it in."""
    print(f"finrag {name} is not wired up yet.")
    print(f"It arrives in rung {PENDING_COMMANDS[name]}.")
    return 1


def print_stats(stats: IngestStats) -> None:
    """Print the ingest summary as two small tables."""
    print(f"Wrote {stats.n_chunks} chunks to {stats.chunks_path}")
    print(f"Wrote a BM25 index of {stats.n_indexed} chunks to {stats.bm25_dir}")
    print(
        f"Wrote a dense matrix of {stats.n_embedded} chunks to {stats.dense_path} "
        f"(embedder: {stats.embedder_name})"
    )
    print()
    _print_counts("By doc type", stats.by_doc_type)
    print()
    _print_counts("By owner department", stats.by_owner_dept)


def _print_counts(title: str, counts: dict[str, int]) -> None:
    print(title)
    for name in counts:
        print(f"  {name:<14}{counts[name]:>6}")


def main(argv: list[str] | None = None) -> int:
    """Entry point for the installed `finrag` script."""
    # The Vertex project id and location live in .env. Read them here, once, at
    # the front door, so nothing deeper down has to know that file exists.
    # Anything already in the environment wins, so a shell override still works.
    load_env()

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "ingest":
        return command_ingest()
    if args.command == "eval":
        return command_eval(args.mode, args.rung, args.k)
    if args.command == "sweep":
        return command_sweep(args.rung, args.k, args.modes)
    return command_pending(args.command)


if __name__ == "__main__":
    sys.exit(main())
