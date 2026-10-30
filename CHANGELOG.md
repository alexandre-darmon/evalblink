# Changelog

All notable changes to evalblink. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.2] — 2026-10-30

### Added
- `run --model / --prompt / --tag / --limit` run part of a benchmark without editing the YAML.
- A top-level `gate: {model, prompt}` makes the CI gate check one combo instead of the best one.
- `compare` with no arguments diffs the two latest runs of a benchmark, and with one run uses it as the baseline for the latest run; `report` with none uses the latest run. Both accept a run ID or a unique prefix of one. Only runs with the same `run` filters (now recorded in the JSON) are paired.
- `compare --fail-on-regression POINTS` exits 1 when a combo's pass rate, with unscored cases counted as failures, drops by more than `POINTS`, and when no combo could be compared at all — for CI against a kept baseline.
- `python -m evalblink`.
- A bare `evalblink` prints the help with a quickstart and the environment variables; every subcommand's help ends with examples.
- `cache clear --all` also drops the cached model catalog.

### Changed
- `validate` rejects unknown top-level keys and warns about unknown nested ones, with a "did you mean" suggestion (`qualty_threshold` used to switch the CI gate off silently). Keys starting with `_` or `x-` are free.
- `run --dry-run` exits 1 on a model ID that isn't in the OpenRouter catalog and suggests the closest one.
- Table cells wrap instead of ending in `…`, so model and run IDs can always be copied.
- `cache stats` and `cache clear` count cached responses apart from the model catalog, which `clear` keeps unless `--all` is given.
- A rate limit (HTTP 429) backs off 5 s / 10 s / 20 s with jitter, or the server's `Retry-After`; when retries run out the error says what to do. The `init` starter sets `concurrency: 2`.
- Jinja2 ≥ 3.1.6 is required (sandbox escape fixes).

## [0.1.1] — 2026-10-30

### Added
- `init` asks three questions and writes a commented starter that validates and runs as-is; `init [path]` and `init --yes`.
- `models --text` (text in, text-only out); `init` suggests free text models and a small paid option.
- `run` and `validate` with no path use the only YAML in `benchmarks/`.

### Changed
- `compare` notes each combo's pipeline-error count and leaves combos whose count changed out of the verdict.
- `--dry-run` caps budgets at each model's output limit and warns when `max_tokens` or `judge_max_tokens` exceeds it.
- `insights.passed` is `false` whenever nothing could be scored, matching exit code 1.

## [0.1.0] — 2026-10-29

First release: `run`, `validate`, `compare` (`--detailed`), `report`, `history`, `models`, `init` and `cache`, with exact-match, weighted-match and LLM-as-judge scoring through OpenRouter.

[Unreleased]: https://github.com/alexandre-darmon/evalblink/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/alexandre-darmon/evalblink/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/alexandre-darmon/evalblink/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/alexandre-darmon/evalblink/releases/tag/v0.1.0
