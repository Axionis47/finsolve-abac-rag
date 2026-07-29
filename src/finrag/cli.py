"""The `finrag` command line: one subcommand per stage of the build.

This is the front door a person uses. `finrag ingest` runs the ingest pipeline
and prints what it wrote. `finrag eval` scores a retrieval mode against the gold
set and saves the report. The remaining subcommands exist so the installed
console script has a complete shape from day one; each prints which rung of the
build will fill it in and exits non-zero, so nothing mistakes a stub for work.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from finrag.config import Settings
from finrag.eval.goldset import GoldSetError, load_gold, validate
from finrag.eval.harness import EvalHarness
from finrag.eval.report_io import write_report
from finrag.ingest.pipeline import IngestStats, read_chunks, run_ingest
from finrag.policy.engine import PolicyEngine
from finrag.retrieval.registry import IndexBundle, available_modes, build_retriever

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

    subparsers.add_parser("ingest", help="chunk the corpus and write artifacts/chunks.jsonl")
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
    path = write_report(report, rung, mode, settings.results_dir)

    print_report(report, k_values, path)
    return 0


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
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "ingest":
        return command_ingest()
    if args.command == "eval":
        return command_eval(args.mode, args.rung, args.k)
    return command_pending(args.command)


if __name__ == "__main__":
    sys.exit(main())
