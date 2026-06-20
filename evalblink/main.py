"""Benchmark LLM prompts × models on your own data via OpenRouter."""

from __future__ import annotations

import argparse
import json
import os
import sys

import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from evalblink import cache, compare, reporter, runner

DEFAULT_CONFIG = "benchmarks/exact_match_classification.yaml"
RESULTS_DIR = "results"


def load_config(filepath):
    """Load a benchmark YAML, exiting with a clear message on failure."""
    try:
        with open(filepath) as stream:
            return yaml.safe_load(stream)
    except FileNotFoundError:
        sys.exit(f"Config not found: {filepath}")
    except yaml.YAMLError as exc:
        sys.exit(f"Invalid YAML in {filepath}: {exc}")


def cmd_run(args):
    """Run a benchmark, write the report, and exit on the CI quality gate."""
    config = load_config(args.config)
    use_cache = not args.no_cache
    results, timestamp = runner.run(config, verbose=args.verbose, use_cache=use_cache)
    result = reporter.write(config, results, timestamp)

    # Top-level CI gate (0-100 %); separate from the per-case scorer thresholds
    # under `evaluation:`.
    threshold = config.get("quality_threshold")
    if threshold is None:
        print("\nGate: no quality_threshold set — not enforcing pass/fail.")
        sys.exit(0)
    if result["passed"]:
        print(f"\nPASS: best score meets quality_threshold ({threshold}).")
        sys.exit(0)
    print(f"\nFAIL: best score is below quality_threshold ({threshold}).")
    sys.exit(1)


def cmd_compare(args):
    """Print the quality/cost delta between two persisted run records."""
    try:
        record_a = compare.load_record(args.file_a)
        record_b = compare.load_record(args.file_b)
    except FileNotFoundError as exc:
        sys.exit(f"Run record not found: {exc.filename}")
    if args.detailed:
        reporter.render_detailed(compare.detailed_diff(record_a, record_b))
    else:
        reporter.render_comparison(compare.diff(record_a, record_b))
    sys.exit(0)


def cmd_report(args):
    """Regenerate the Markdown report from an existing JSON result file."""
    try:
        record = compare.load_record(args.file)
    except FileNotFoundError as exc:
        sys.exit(f"Run record not found: {exc.filename}")
    reporter.write_from_record(record)
    sys.exit(0)


def cmd_history(args):
    """List all past runs in the results/ directory."""
    if not os.path.isdir(RESULTS_DIR):
        print("No results directory found. Run a benchmark first.")
        sys.exit(0)

    records = []
    for fname in os.listdir(RESULTS_DIR):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(RESULTS_DIR, fname)
        try:
            with open(path) as f:
                data = json.load(f)
            total_cost = sum(r.get("total_cost", 0) for r in data.get("results", []))
            insights = data.get("insights") or {}
            best = insights.get("best_quality", {})
            best_score = best.get("score")
            errors = insights.get("errors", 0)
            records.append(
                {
                    "run_id": data.get("run_id", os.path.basename(fname)),
                    "benchmark": data.get("benchmark", "—"),
                    "timestamp": data.get("timestamp", "—"),
                    "best_score": best_score,
                    "total_cost": total_cost,
                    "errors": errors,
                }
            )
        except (json.JSONDecodeError, OSError):
            continue

    if not records:
        print("No run records found in results/.")
        sys.exit(0)

    records.sort(key=lambda r: r["timestamp"], reverse=True)

    console = Console()
    table = Table(title="Run History")
    table.add_column("Run ID")
    table.add_column("Benchmark")
    table.add_column("Timestamp")
    table.add_column("Best Score", justify="right")
    table.add_column("Total Cost", justify="right")
    table.add_column("Errors", justify="right")
    for r in records:
        score = "—" if r["best_score"] is None else f"{r['best_score']:.1f}%"
        table.add_row(
            r["run_id"],
            r["benchmark"],
            r["timestamp"],
            score,
            f"${r['total_cost']:.6f}",
            str(r["errors"]) if r["errors"] else "—",
        )
    console.print(table)
    sys.exit(0)


def cmd_cache_stats(args):
    """Print cache entry count and total size."""
    s = cache.stats()
    size_kb = s["size_bytes"] / 1024
    print(f"Cache entries : {s['entries']}")
    print(f"Cache size    : {size_kb:.1f} KB")
    sys.exit(0)


def cmd_cache_clear(args):
    """Delete all cached responses."""
    if not args.yes:
        s = cache.stats()
        print(f"{s['entries']} cache entries. Pass --yes to confirm deletion.")
        sys.exit(0)
    removed = cache.clear()
    print(f"Cleared {removed} cache entries.")
    sys.exit(0)


_RUN_DESCRIPTION = "Run a benchmark YAML against all model × prompt combinations."

_RUN_EPILOG = """\
YAML CONFIG KEYS
  name                          required
  models                        list of OpenRouter model IDs  (required)
  prompts[].id                  prompt identifier             (required)
  prompts[].template            Jinja2 — use {{ var }}        (required)
  prompts[].system              optional system message
  inference.temperature         default: 0
  inference.max_tokens          default: 100
  variables                     global key/value pairs injected into templates
  quality_threshold             CI gate 0–100; exit 1 if best score is below
  evaluation.judge_model        required for llm_judge
  evaluation.judge_threshold    default: 0.70
  test_cases[].id               required
  test_cases[].evaluation       exact_match | llm_judge | weighted_match  (required)
  test_cases[].expected_output  required for exact_match
  test_cases[].criteria         required for llm_judge
  test_cases[].reference        optional gold answer injected into the judge prompt
  test_cases[].variables        per-case variables
  test_cases[].tags             list of strings for per-category breakdown

EXAMPLES
  evalblink run benchmarks/classification.yaml
  evalblink run benchmarks/classification.yaml -v
  evalblink run benchmarks/classification.yaml --no-cache
"""

_COMPARE_DESCRIPTION = "Diff two run records — quality and cost deltas, no API calls."

_COMPARE_EPILOG = """\
OUTPUT COLUMNS
  Quality A / B   pass-rate % per run
  Quality Δ       ↑ / ↓ / stable (threshold: 5 pp)
  Cost A / B      total cost per combo (models + judge)
  Cost Δ          cost change
  Notes           a_only / b_only when a combo is missing from one run

--detailed also shows per-combo case transitions and a global summary:
  regressed   pass → fail
  improved    fail → pass
  new_error   scored → None  (pipeline error, not counted as a regression)
  recovered   None → scored

EXAMPLES
  evalblink compare results/run_a.json results/run_b.json
  evalblink compare results/run_a.json results/run_b.json --detailed
"""


def build_parser():
    parser = argparse.ArgumentParser(
        prog="evalblink",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    run_p = subparsers.add_parser(
        "run",
        help="run a benchmark config against all model × prompt combinations",
        description=_RUN_DESCRIPTION,
        epilog=_RUN_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    run_p.add_argument(
        "config",
        nargs="?",
        default=DEFAULT_CONFIG,
        help=(f"path to a benchmark YAML (default: {DEFAULT_CONFIG})"),
    )
    run_p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print per-test-case detail (response, score, status) during the run",
    )
    run_p.add_argument(
        "--no-cache",
        action="store_true",
        help="bypass the local cache and force fresh API calls for all candidate requests",
    )
    run_p.set_defaults(func=cmd_run)

    # --- compare ---
    compare_p = subparsers.add_parser(
        "compare",
        help="diff two run records — quality and cost deltas, no API calls",
        description=_COMPARE_DESCRIPTION,
        epilog=_COMPARE_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    compare_p.add_argument(
        "file_a",
        metavar="FILE_A",
        help="baseline run JSON (the earlier / reference run)",
    )
    compare_p.add_argument(
        "file_b",
        metavar="FILE_B",
        help="candidate run JSON (the run being evaluated)",
    )
    compare_p.add_argument(
        "-d",
        "--detailed",
        action="store_true",
        help=(
            "drill into per-test-case transitions and show a global summary "
            "(worst-regressed cases, per-tag net change)"
        ),
    )
    compare_p.set_defaults(func=cmd_compare)

    # --- report ---
    report_p = subparsers.add_parser(
        "report",
        help="regenerate the Markdown report from an existing JSON result file",
    )
    report_p.add_argument(
        "file",
        metavar="FILE",
        help="path to a result JSON file (from results/)",
    )
    report_p.set_defaults(func=cmd_report)

    # --- history ---
    history_p = subparsers.add_parser(
        "history",
        help="list all past runs in the results/ directory",
    )
    history_p.set_defaults(func=cmd_history)

    # --- cache ---
    cache_p = subparsers.add_parser(
        "cache",
        help="manage the local response cache",
    )
    cache_sub = cache_p.add_subparsers(dest="cache_command", required=True)

    stats_p = cache_sub.add_parser("stats", help="show entry count and total size")
    stats_p.set_defaults(func=cmd_cache_stats)

    clear_p = cache_sub.add_parser("clear", help="delete all cached responses")
    clear_p.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="confirm deletion without prompting",
    )
    clear_p.set_defaults(func=cmd_cache_clear)

    return parser


def main():
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
