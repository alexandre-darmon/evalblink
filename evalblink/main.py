"""Entrypoint: load config, run the benchmark, write the report.

Run with ``python -m evalblink.main [path/to/benchmark.yaml]``.
"""

from __future__ import annotations

import argparse
import sys

import yaml
from dotenv import load_dotenv

from evalblink import reporter, runner

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


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(prog="evalblink", description=__doc__)
    parser.add_argument(
        "config", nargs="?", default=DEFAULT_CONFIG, help="path to a benchmark YAML"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="print per-test-case detail"
    )
    args = parser.parse_args()
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


if __name__ == "__main__":
    main()
