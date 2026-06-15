"""Entrypoint: load config, run the benchmark, write the report.

Run with ``python -m evalblink.main [path/to/benchmark.yaml]``.
"""

from __future__ import annotations

import sys

import yaml
from dotenv import load_dotenv

from evalblink import reporter, runner

DEFAULT_CONFIG = "benchmarks/llm_as_judge.yaml"


def load_config(filepath):
    with open(filepath) as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None


def main():
    load_dotenv()
    filepath = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    config = load_config(filepath)
    results, timestamp = runner.run(config)
    reporter.write(config, results, timestamp)


if __name__ == "__main__":
    main()
