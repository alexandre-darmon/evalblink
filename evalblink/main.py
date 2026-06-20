"""Entrypoint: dispatch the ``run`` and ``compare`` subcommands.

python -m evalblink.main run [path/to/benchmark.yaml]
python -m evalblink.main compare results/a.json results/b.json
"""

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


def build_parser():
    parser = argparse.ArgumentParser(prog="evalblink", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_p = subparsers.add_parser("run", help="run a benchmark config")
    run_p.add_argument(
        "config", nargs="?", default=DEFAULT_CONFIG, help="path to a benchmark YAML"
    )
    run_p.add_argument(
        "-v", "--verbose", action="store_true", help="print per-test-case detail"
    )
    run_p.set_defaults(func=cmd_run)

    compare_p = subparsers.add_parser(
        "compare", help="diff two run records (quality + cost)"
    )
    compare_p.add_argument("file_a", help="earlier run JSON (results/*.json)")
    compare_p.add_argument("file_b", help="later run JSON (results/*.json)")
    compare_p.add_argument(
        "-d",
        "--detailed",
        action="store_true",
        help="drill into per-test-case changes and a global summary",
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
