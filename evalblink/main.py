"""Benchmark LLM prompts × models on your own data via OpenRouter."""

from __future__ import annotations

import argparse
import sys

import yaml
from dotenv import load_dotenv

from evalblink import compare, reporter, runner

DEFAULT_CONFIG = "benchmarks/exact_match_classification.yaml"


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
    results, timestamp = runner.run(config, verbose=args.verbose)
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
  test_cases[].variables        per-case variables
  test_cases[].tags             list of strings for per-category breakdown

EXAMPLES
  evalblink run benchmarks/classification.yaml
  evalblink run benchmarks/classification.yaml -v
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
    run_p.set_defaults(func=cmd_run)

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

    return parser


def main():
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
