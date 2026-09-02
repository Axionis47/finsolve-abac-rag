"""The `finrag` command line: one subcommand per stage of the build.

This is the front door a person uses. `finrag ingest` runs the ingest pipeline
and prints what it wrote. `finrag graph` rebuilds the Neo4j graph from those
chunks. `finrag eval` scores one composed search pipeline against the gold set.
`finrag sweep` runs the ABAC leakage sweep over questions, roles and strategy
combinations. `finrag serve` runs the API and `finrag ui` runs the Streamlit pages
against it. Every subcommand is wired; none of them is a stub.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from finrag.config import Settings, load_env
from finrag.eval.goldset import GoldSetError, load_gold, validate
from finrag.eval.harness import EvalHarness
from finrag.eval.report_io import write_report
from finrag.eval.sweep import (
    breakdown_lines,
    build_sweep_pipelines,
    run_leakage_sweep,
    sweep_passed,
)
from finrag.graph import build_llm as graph_build_llm
from finrag.graph import build_structural
from finrag.graph.store import GraphStore
from finrag.ingest.pipeline import IngestStats, read_chunks, run_ingest
from finrag.ingest.registry import DEFAULT_CHUNKER, available_chunkers
from finrag.llm.vertex import build_extraction_llm
from finrag.pipeline import (
    SearchPipeline,
    StrategySelection,
    StrategyUnavailableError,
    load_runtime,
)
from finrag.policy.engine import PolicyEngine
from finrag.querying.registry import available_query_strategies
from finrag.reranking.registry import available_rerankers
from finrag.retrieval.registry import available_retrievers, retriever_requirements
from finrag.strategy import Requirement

# The Streamlit pages live outside the package, next to pyproject.toml.
REPO_ROOT_UI = Path(__file__).resolve().parents[2] / "ui"


def build_parser() -> argparse.ArgumentParser:
    """Define the command line: `finrag <command>`, one command required."""
    parser = argparse.ArgumentParser(
        prog="finrag",
        description="FinSolve ABAC RAG: ingest the corpus, build the graph, eval, serve.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="chunk the corpus, write artifacts/chunks.jsonl, the BM25 index and dense.npz",
    )
    ingest_parser.add_argument(
        "--chunker",
        default=None,
        help=f"chunking strategy to use. One of: {', '.join(available_chunkers())}",
    )
    graph_parser = subparsers.add_parser(
        "graph", help="rebuild the Neo4j knowledge graph from artifacts/chunks.jsonl"
    )
    graph_parser.add_argument(
        "--llm",
        action="store_true",
        help=(
            "also ask Gemini for entity triples, one call per chunk "
            "(costs money; the structural build alone is free)"
        ),
    )

    eval_parser = subparsers.add_parser("eval", help="score one strategy combination")
    eval_parser.add_argument(
        "--retriever",
        default="bm25",
        help=f"retrieval strategy. One of: {', '.join(available_retrievers())}",
    )
    eval_parser.add_argument(
        "--query-strategy",
        default="direct",
        help=f"query strategy. One of: {', '.join(available_query_strategies())}",
    )
    eval_parser.add_argument(
        "--reranker",
        default="none",
        help=f"reranking strategy. One of: {', '.join(available_rerankers())}",
    )
    eval_parser.add_argument("--rung", default="r0", help="build rung, used in the report name")
    eval_parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[5, 10],
        help="cut-offs to score at (default: 5 10)",
    )
    eval_parser.add_argument(
        "--chunker",
        default=None,
        help=f"chunking strategy the artifacts were built with. One of: "
        f"{', '.join(available_chunkers())}",
    )

    sweep_parser = subparsers.add_parser(
        "sweep",
        help="ABAC leakage sweep over composed strategy pipelines",
    )
    sweep_parser.add_argument("--rung", default="r4", help="build rung, used in the report name")
    sweep_parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="how many hits to ask each pipeline for (default: 10)",
    )
    sweep_parser.add_argument(
        "--retrievers",
        nargs="+",
        default=None,
        help=f"retrievers to sweep (default: all of {', '.join(available_retrievers())})",
    )
    sweep_parser.add_argument(
        "--query-strategies",
        nargs="+",
        default=None,
        help=f"query strategies to sweep (default: all of "
        f"{', '.join(available_query_strategies())})",
    )
    sweep_parser.add_argument(
        "--rerankers",
        nargs="+",
        default=None,
        help=f"rerankers to sweep (default: all of {', '.join(available_rerankers())})",
    )
    sweep_parser.add_argument(
        "--chunker",
        default=None,
        help=f"chunking strategy the artifacts were built with. One of: "
        f"{', '.join(available_chunkers())}",
    )

    serve_parser = subparsers.add_parser("serve", help="run the API on localhost")
    serve_parser.add_argument("--host", default="127.0.0.1", help="interface to bind")
    serve_parser.add_argument("--port", type=int, default=8000, help="port to bind")
    serve_parser.add_argument(
        "--reload", action="store_true", help="restart on a code change (development only)"
    )

    ui_parser = subparsers.add_parser("ui", help="run the Streamlit UI (needs the API running)")
    ui_parser.add_argument("--port", type=int, default=8501, help="port for the UI")
    ui_parser.add_argument(
        "--api",
        default="http://127.0.0.1:8000",
        help="base URL of the finrag API the pages call",
    )

    return parser


def _settings_for_chunker(chunker: str | None) -> Settings | None:
    """Settings with the chunker resolved, or None after printing why not.

    Checks `settings.chunker` after the `--chunker` replace, not before, so a
    bad name reaches the same refusal whether it came from the flag or from
    FINRAG_CHUNKER in the environment. Without this, an unset flag with a bad
    env var would sail past every caller's own check and die inside
    `build_chunker` (ingest) or behind a misleading "no chunks" message
    (eval, sweep).
    """
    settings = Settings.from_env()
    if chunker is not None:
        settings = replace(settings, chunker=chunker)
    if settings.chunker not in available_chunkers():
        known = ", ".join(available_chunkers())
        print(f"Unknown chunker '{settings.chunker}'. Known chunkers: {known}")
        return None
    return settings


def command_ingest(chunker: str | None) -> int:
    """Run ingest against the configured corpus and print the result."""
    settings = _settings_for_chunker(chunker)
    if settings is None:
        return 1
    stats = run_ingest(settings)
    print_stats(stats)
    return 0


def command_graph(with_llm: bool = False) -> int:
    """Rebuild the :Finrag graph from artifacts/chunks.jsonl. Wipes, then writes.

    The structural and temporal layers come from fixed parsing rules. The
    optional entity layer asks Gemini for triples. All lists go into one
    `write_graph` call because that call wipes the namespace first; separate
    writes would leave only the last layer.
    """
    settings = Settings.from_env()
    if settings.chunker != DEFAULT_CHUNKER:
        print(
            "The graph lives in a single Neo4j namespace, built from the default "
            f"chunker's chunks only. Run without FINRAG_CHUNKER (got '{settings.chunker}')."
        )
        return 1
    if not settings.chunks_path.exists():
        print(f"No chunks at {settings.chunks_path}. Run `finrag ingest` first.")
        return 1

    chunks = read_chunks(settings.chunks_path)
    nodes, edges = build_structural.extract(chunks)
    stats = None
    if with_llm:
        extractor = build_extraction_llm(settings)
        print(
            f"Asking {extractor.model} for triples over {len(chunks)} chunks, "
            "one call each..."
        )
        llm_nodes, llm_edges, stats = graph_build_llm.extract(
            extractor,
            chunks,
            cache_path=settings.graph_llm_cache_path,
            model=extractor.model,
        )
        nodes = nodes + llm_nodes
        edges = edges + llm_edges

    with GraphStore() as store:
        counts = store.write_graph(nodes, edges)
        layer_counts = store.layer_counts()

    print(f"Wrote {counts.n_nodes} nodes and {counts.n_edges} relationships to :Finrag")
    for layer, count in layer_counts.items():
        print(f"  {layer}: {count} relationships")
    if stats is not None:
        print(
            f"entity model: {stats.model}  prompt: {stats.prompt_version}  "
            f"model calls: {stats.n_model_calls}  cache hits: {stats.n_cache_hits}"
        )
        print(
            f"triples: {stats.n_triples} from {stats.n_parsed} chunks, "
            f"{stats.n_dropped_items} items dropped, "
            f"fallback rate: {stats.fallback_rate:.2f} ({stats.n_skipped} chunks)"
        )
    return 0


def command_eval(
    retriever: str,
    query_strategy: str,
    reranker: str,
    rung: str,
    k_values: list[int],
    chunker: str | None = None,
) -> int:
    """Score one composed strategy pipeline and save the report.

    Straight line: load the artifacts, build the retriever, run the harness,
    write the JSON, print the per-tier table.
    """
    settings = _settings_for_chunker(chunker)
    if settings is None:
        return 1

    if not settings.chunks_path.exists():
        print(f"No chunks at {settings.chunks_path}. Run `finrag ingest` first.")
        return 1
    if not settings.gold_path.exists():
        print(f"No gold set at {settings.gold_path}.")
        return 1
    selection = StrategySelection(retriever, query_strategy, reranker)
    try:
        pipeline = SearchPipeline(selection, load_runtime(settings))
    except (KeyError, ValueError, RuntimeError, StrategyUnavailableError) as problem:
        print(problem)
        return 1
    policy = PolicyEngine.from_yaml(settings.policy_path)

    # The gold set points at the corpus by name, so a renamed heading or a
    # dropped document turns every score into nonsense without failing. Check it
    # against the chunks we just loaded and the real policy before scoring
    # anything, and refuse to publish numbers from a gold set that has drifted.
    try:
        gold = load_gold(settings.gold_path)
        validate(gold, settings.chunks_path, policy, chunker=settings.chunker)
    except GoldSetError as problem:
        print(problem)
        return 1
    if settings.chunker != DEFAULT_CHUNKER:
        print(
            f"NOT CHECKED: gold_sections/answer_source (written for the "
            f"'{DEFAULT_CHUNKER}' chunker, not '{settings.chunker}')"
        )

    report = EvalHarness(pipeline, policy, gold.entries).run(k_values=tuple(k_values))
    add_strategy_counters(report, pipeline)
    path = write_report(report, rung, selection.pipeline_id, settings.results_dir)

    print_report(report, k_values, path)
    return 0


def command_sweep(
    rung: str,
    k: int,
    retrievers: list[str] | None = None,
    query_strategies: list[str] | None = None,
    rerankers: list[str] | None = None,
    chunker: str | None = None,
) -> int:
    """Run the leakage sweep, print the table, and fail on any violation.

    Exits 1 when a denied chunk reached any result list, so a CI run or a shell
    `&&` chain stops there. That is the point of the command: it is a gate, not a
    report you read and shrug at.
    """
    settings = _settings_for_chunker(chunker)
    if settings is None:
        return 1

    if not settings.chunks_path.exists():
        print(f"No chunks at {settings.chunks_path}. Run `finrag ingest` first.")
        return 1
    if not settings.gold_path.exists():
        print(f"No gold set at {settings.gold_path}.")
        return 1
    policy = PolicyEngine.from_yaml(settings.policy_path)
    try:
        gold = load_gold(settings.gold_path)
        validate(gold, settings.chunks_path, policy, chunker=settings.chunker)
    except GoldSetError as problem:
        print(problem)
        return 1
    if settings.chunker != DEFAULT_CHUNKER:
        print(
            f"NOT CHECKED: gold_sections/answer_source (written for the "
            f"'{DEFAULT_CHUNKER}' chunker, not '{settings.chunker}')"
        )

    try:
        pipelines = build_sweep_pipelines(
            settings,
            retrievers=retrievers,
            query_strategies=query_strategies,
            rerankers=rerankers,
        )
    except (KeyError, ValueError, RuntimeError) as problem:
        # A graph retriever was asked for with the container down. That is a setup
        # problem with a one-line fix, not a stack trace.
        print(problem)
        return 1

    # A sweep that quietly covered five modes when six exist would read as a
    # clean bill of health for the sixth. Name what is not being checked.
    requested_retrievers = retrievers or available_retrievers()
    for name in requested_retrievers:
        is_graph = Requirement.GRAPH in retriever_requirements(name)
        was_built = any(selection.retrieval_strategy == name for selection in pipelines)
        if is_graph and not was_built:
            print(f"NOT SWEPT: {name} (its backing service is not answering)")

    report = run_leakage_sweep(pipelines, policy, gold.entries, k=k)
    path = write_report(report, rung, "sweep", settings.results_dir)

    print_sweep(report, path)
    return 0 if sweep_passed(report) else 1


def print_sweep(report: dict[str, Any], path: Path) -> None:
    """Print the role-by-pipeline blocks, then the headline counts."""
    print(
        f"questions={report['n_questions']}  roles={len(report['roles'])}  "
        f"pipelines={len(report['pipelines'])}  cells={report['cells_checked']}  "
        f"k={report['k']}"
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
            f"via {violation['pipeline_id']}: {violation['chunk_id']} "
            f"(denied by {violation['denied_by_rule']})"
        )
    print()
    print(f"Wrote {path}")


def add_strategy_counters(report: dict[str, Any], pipeline: SearchPipeline) -> None:
    """Attach query and reranking costs without knowing concrete strategy classes."""
    configuration = getattr(pipeline.retriever, "configuration", None)
    if callable(configuration):
        report["retrieval_strategy_config"] = configuration()

    query = pipeline.query_strategy
    totals = getattr(query, "totals", None)
    if totals is not None:
        questions = max(totals.get("questions", 0), 1)
        block = dict(totals)
        block["mean_hops"] = round(totals.get("hops", 0) / questions, 3)
        block["mean_llm_calls"] = round(totals.get("llm_calls", 0) / questions, 3)
        block["last_failure"] = getattr(query, "last_failure", None)
        report["query_strategy_stats"] = block
    elif hasattr(query, "calls"):
        report["query_strategy_stats"] = {
            "llm_calls": query.calls,
            "fallbacks": query.fallbacks,
            "last_failure": query.last_failure,
        }

    reranker = pipeline.reranker
    if hasattr(reranker, "calls"):
        report["reranking_strategy_stats"] = {
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
    print(f"pipeline={report['pipeline_id']}  questions={report['n_questions']}")
    print()

    header = f"{'tier':<8}{'n':>4}{'scored':>8}{'sec hit':>10}{'sec mrr':>10}"
    header += f"{'sec rec@' + str(top_k):>12}{'all sec':>10}"
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
    if "reranking_strategy_stats" in report:
        rerank = report["reranking_strategy_stats"]
        print(
            f"rerank: {rerank['llm_calls']} llm calls, "
            f"{rerank['fallbacks']} fallbacks to the hybrid order"
        )
        if rerank["last_failure"]:
            print(f"last rerank failure: {rerank['last_failure']}")
    query_stats = report.get("query_strategy_stats")
    if query_stats and "questions" in query_stats:
        agent = query_stats
        print(
            f"agent: {agent['llm_calls']} llm calls over {agent['questions']} questions "
            f"({agent['mean_llm_calls']} each), {agent['mean_hops']} mean hops, "
            f"{agent['forced_finishes']} forced finishes"
        )
        print(
            f"agent failures: {agent['parse_failures']} parse, "
            f"{agent['repeated_queries']} repeated queries, "
            f"{agent['unknown_actions']} unknown actions, "
            f"{agent['call_failures']} call failures, "
            f"{agent['fallback_searches']} fallback searches"
        )
        if agent["last_failure"]:
            print(f"last agent failure: {agent['last_failure']}")
    elif query_stats:
        print(
            f"query strategy: {query_stats['llm_calls']} llm calls, "
            f"{query_stats['fallbacks']} direct fallbacks"
        )
        if query_stats["last_failure"]:
            print(f"last query failure: {query_stats['last_failure']}")
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
    row += _fixed(summary["section_recall@" + str(top_k)], 12)
    row += _fixed(summary["all_sections@" + str(top_k)], 10)
    row += _fixed(summary["recall@" + str(top_k)], 12)
    row += _fixed(summary["mrr"], 10)
    row += f"{summary['leakage_count']:>6}"
    print(row)


def command_serve(host: str, port: int, reload: bool) -> int:
    """Run the FastAPI app with uvicorn.

    Passed as an import string with `factory=True` so uvicorn builds the app in
    the worker process; `--reload` needs that anyway, and it keeps this function
    from loading 318 chunks in a process that is about to be replaced.
    """
    import uvicorn

    settings = Settings.from_env()
    if not settings.chunks_path.exists():
        print(f"No chunks at {settings.chunks_path}. Run `finrag ingest` first.")
        return 1

    print(f"Serving finrag on http://{host}:{port}  (docs at /docs)")
    uvicorn.run(
        "finrag.api.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
    )
    return 0


def command_ui(port: int, api: str) -> int:
    """Run the Streamlit UI, pointed at a running API.

    Streamlit has no supported in-process entry point, so this shells out to the
    module the same way `streamlit run` does. FINRAG_API is how the pages find
    the API; it is set here so nobody has to export it by hand.
    """
    import os
    import subprocess

    entry = REPO_ROOT_UI / "app.py"
    if not entry.exists():
        print(f"No UI entry script at {entry}.")
        return 1

    env = dict(os.environ)
    env["FINRAG_API"] = api
    print(f"Starting the UI on http://127.0.0.1:{port}, talking to {api}")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(entry),
            "--server.port",
            str(port),
        ],
        env=env,
        check=False,
    )
    return completed.returncode


def print_stats(stats: IngestStats) -> None:
    """Print the ingest summary as two small tables."""
    print(f"chunker: {stats.chunker}")
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
        return command_ingest(args.chunker)
    if args.command == "graph":
        return command_graph(args.llm)
    if args.command == "eval":
        return command_eval(
            args.retriever,
            args.query_strategy,
            args.reranker,
            args.rung,
            args.k,
            args.chunker,
        )
    if args.command == "sweep":
        return command_sweep(
            args.rung,
            args.k,
            args.retrievers,
            args.query_strategies,
            args.rerankers,
            args.chunker,
        )
    if args.command == "serve":
        return command_serve(args.host, args.port, args.reload)
    if args.command == "ui":
        return command_ui(args.port, args.api)

    # Unreachable: argparse rejects any name that has no subparser. It is here so
    # a new subparser with no branch above fails loudly instead of exiting 0.
    print(f"finrag {args.command} has no handler. This is a bug in cli.py.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
