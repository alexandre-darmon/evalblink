# Changelog

All notable changes to evalblink. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Latency.** Every candidate call's latency is measured: the wall time of the attempt that returned the reply, without retries, backoff or judge calls. It shows on each progress line (`pass  0.62s`), as p50 in the results table, p50 and p95 in the Markdown report, per case and per combo in the JSON (`latency_s`, `latency_p50_s`, `latency_p95_s`, `latency_mean_s`), as a **Fastest** line in the recommendation, and as a `p50 latency Δ` column in `compare`. A cached reply keeps the latency measured when it was fetched.
- **Confidence intervals and noise.** Each score comes with its 95 % interval (Wilson, over the number of test cases): `96.0% (87–99)` in the table, `score_ci` in the JSON. The recommendation lists the combos **within noise** of the best one (not distinguishable at 95 %), and `compare` marks quality changes within noise and says how many gains are beyond it.
- **`repeats: N`** runs every model × prompt × test case N times (each repeat with its own cache entry). The score covers every repeat, cases whose repeats disagree are reported as **flaky**, and `--dry-run` multiplies the estimate by N.
- **`json_match`**: the reply's JSON must hold the values in `expected_output` (extra keys ignored, strings ignore case, partial credit shown). **`json_schema`**: the reply's JSON must satisfy a JSON Schema. **`python`**: your own scorer, `scorer: "scorers.py:my_check"`, returning `True`/`False` or `{passed, score, reason}`; a scorer that raises is a pipeline error (`scorer error`), not a failed case. Example: `benchmarks/structured_output.yaml`.
- **GitHub Action** (`action.yml`): `uses: alexandre-darmon/evalblink@<tag>` runs a benchmark, puts the report in the job summary, and fails the job on the quality gate or, with `baseline`, on a regression.
- **Local and self-hosted models.** A model id `ollama:llama3.2` (or `lmstudio:…`, `llamacpp:…`, `vllm:…`) runs on that server at its default local address, next to OpenRouter models in the same run; a `providers:` block moves a built-in server or adds any OpenAI-compatible one (`base_url`, optional `api_key_env`, `concurrency`, `timeout`). The data stays on your machine, local calls cost $0, and a run of local models needs no OpenRouter key (`--dry-run` then makes no network call). Before the first call, `run` checks each server and model and says how to fix a problem (`Start it: ollama serve`, `Get it: ollama pull llama3.2`). Each server gets one request at a time by default, so its queue never counts as latency. The judge can be local too. Example: `benchmarks/local_models.yaml`.
- `docs/reference.md`: the full reference (every config key, validation rule, mode, statistic and output file).

### Changed
- The README is an overview (about a third of its former length), opens with a banner, and its terminal samples are visuals in the same style (`docs/assets/`), each with the full text output in a collapsible block. The details moved to `docs/reference.md`.
- The terminal results table fits an 80-column terminal: `Score (95% CI)`, `Tokens in/out` and `p50 latency` columns. The Markdown report keeps separate columns.
- The comparison with promptfoo was rechecked against its documentation and corrected.
- The roadmap adds evaluating your own application (an HTTP endpoint or a command) for V1.1.
- Run records are `schema_version` 2 (additive: version 1 records still load, compare and report, with no latency).
- New dependency: `jsonschema`.

## [0.1.5] — 2026-11-02

### Changed
- `poolside/laguna-xs-2.1:free` was retired by OpenRouter, so every shipped example failed with a 404. The new default (`init` and the example benchmarks) is the free `cohere/north-mini-code:free`, picked by running the examples on every free text model plus the cheapest paid ones: it tied the best (34/38 cases over two rounds) with no errors, and isn't from the judge's vendor. The suggested paid option is now `openai/gpt-oss-20b`: as accurate here, and cheaper than `google/gemma-4-31b-it` ($0.03 / $0.13 per 1M tokens in / out, against $0.20 / $0.40).

### Added
- Three free, deterministic scorers for free-text replies, using `expected_output`: `contains` (every text appears, ignoring case; one string or a list), `not_contains` (none appears) and `regex` (Python `re.search`; `(?i)` to ignore case). `contains` / `not_contains` compare text as a reader sees it: Unicode spaces, line breaks and dashes count as ` ` and `-`, and invisible characters are ignored. `regex` sees the reply as written (surrounding whitespace ignored). `validate` rejects a missing or empty text and a pattern that doesn't compile. Example: `benchmarks/text_checks.yaml`.

## [0.1.4] — 2026-10-31

### Changed
- `run` shows one progress line per finished case, `[3/12] model / prompt / case  pass`, with the 0–1 score for `llm_judge` and `weighted_match` and the reason for a pipeline error (`⚠ API error: …`), after a `Running 12 cases (2 models × 2 prompts × 3 test cases), 5 at a time` line. It replaces the `Test case: <id>` lines and the separate per-error warnings.
- Progress and warnings (retry notices, which now name the model, cache warnings, the judge self-preference warning, the pipeline-error count) go to stderr; the results table, saved paths and PASS/FAIL stay on stdout, so `run … > results.txt` keeps only the results.
- The per-combo totals printed before the table (a second copy of it) are gone.
- The `init` starter's comment now holds a ready-to-save `cases.csv` with its own test cases, and the one line to switch to it.
- CI builds the release artifact with the same workflow as a release (`.github/workflows/build.yml`), downloads it the way `publish.yml` does, checks it holds exactly this version's wheel and sdist, installs both and runs the test suite from the sdist. A broken artifact hand-off now fails on a pull request instead of mid-release.
- GitHub Actions upgraded to Node 24: `checkout` v7, `setup-python` v7, `upload-artifact` v7, `download-artifact` v8.

### Added
- `evalblink init --csv` writes the starter with its test cases in a spreadsheet next to it (`benchmarks/my-benchmark.yaml` + `benchmarks/my-benchmark.csv`), ready to edit in any spreadsheet app. Shown in `evalblink --help`, `init --help` and the README Quickstart.

## [0.1.3] — 2026-10-30

### Added
- Test cases from a CSV, TSV or JSONL file: `test_cases: {file: cases.csv, evaluation: exact_match}`, relative to the YAML. Case-field columns keep their meaning (headers match whatever their case), every other column is a template variable, Excel exports (BOM, `;`-separated) work, and errors — a short row, an unclosed quote — name the file and line. Example: `benchmarks/csv_classification.yaml`.
- The `init` starter shows how to switch to a file.

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

[Unreleased]: https://github.com/alexandre-darmon/evalblink/compare/v0.1.5...HEAD
[0.1.5]: https://github.com/alexandre-darmon/evalblink/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/alexandre-darmon/evalblink/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/alexandre-darmon/evalblink/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/alexandre-darmon/evalblink/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/alexandre-darmon/evalblink/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/alexandre-darmon/evalblink/releases/tag/v0.1.0
