# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

evalblink benchmarks LLM prompts × models on your own data, routing every call through [OpenRouter](https://openrouter.ai). A run takes a YAML benchmark config, executes `model × prompt × test_case`, scores each response, and writes a results matrix (Rich table), a JSON record, and a Markdown report.

## Commands

```bash
# Setup
pip install -e ".[dev]"                  # installs evalblink + dev tools (ruff, pytest)
export OPENROUTER_API_KEY=...             # or put it in .env (loaded via python-dotenv)

# Run a benchmark
evalblink run benchmarks/classification.yaml
evalblink run                            # defaults to benchmarks/exact_match_classification.yaml
evalblink run benchmarks/classification.yaml -v          # verbose: per-test-case detail
evalblink run benchmarks/classification.yaml --no-cache  # bypass local cache
evalblink run benchmarks/classification.yaml --dry-run   # estimate cost, no API calls

# Validate a config (errors and warnings, no API calls)
evalblink validate benchmarks/classification.yaml

# Compare two finished runs (quality + cost deltas; offline, no API calls)
evalblink compare results/<run_a>.json results/<run_b>.json
evalblink compare results/<run_a>.json results/<run_b>.json --detailed  # per-test-case transitions + global summary

# Other commands
evalblink report results/<run>.json      # regenerate Markdown from existing JSON
evalblink history                        # list all past runs
evalblink models                         # list OpenRouter models + pricing
evalblink models --provider anthropic --free --min-context 100k
evalblink init                           # interactively scaffold a new benchmark YAML
evalblink cache stats
evalblink cache clear --yes

# Tests
pytest -q

# Lint (ruff is the only dev tool; no config file, uses defaults)
ruff check .
ruff format .
```

Tests live in `tests/` and run offline (a fake `httpx.Client`, no API key). Beyond unit tests, verify end-to-end by running configs and inspecting the generated `results/<timestamp>_<slug>.{json,md}` files. Runs hit the live OpenRouter API (free models cost $0 but are rate-limited).

## Architecture

A run flows through one linear pipeline; module boundaries are deliberate, not incidental:

- **`main.py`** — `argparse` subcommand dispatcher (`load_dotenv()` then `args.func`). Subcommands: `run`, `validate`, `compare`, `report`, `history`, `models`, `init`, `cache stats`, `cache clear`. `run` validates the config via `validator.validate` before touching the API — exits 1 with error details on bad config. `load_config` exits with a clear message on missing/invalid YAML.
- **`runner.py`** — owns the single `httpx.Client` and the `model × prompt × test_case` triple loop. Renders Jinja2 templates, dispatches work to `ThreadPoolExecutor` (default 5 workers, config key `concurrency`), routes each response to the right evaluator, accumulates per-prompt token/cost totals. Per-case detail prints are gated behind `verbose`.
- **`validator.py`** — static validation of benchmark YAML configs; no API calls, no side effects. Returns `(errors, warnings)`. Called by `cmd_run` and `cmd_validate`. Checks required fields, eval-mode-specific constraints, Jinja2 template variable coverage, and heuristic warnings (single-brace syntax, low max_tokens for llm_judge, long expected_output).
- **`openrouter.py`** — the *single API choke point*. Every network call (candidate responses AND the LLM judge) goes through `openrouter_request`. It lives in its own leaf module specifically to avoid a `runner ↔ evaluator` import cycle. Responsibilities: build the request, consult the SHA256 cache, **retry transient failures** (`RETRYABLE_CODES = {408,429,502,503,504}` + local `httpx.TimeoutException`, `MAX_RETRIES` attempts, exponential backoff), raise `RuntimeError` on non-transient API errors or null content.
- **`evaluator.py`** — three scorers: `exact_match`, `weighted_match`, `evaluate_llm_judge`, plus `_extract_json` (strips markdown code fences before parsing). Pure scoring; `evaluate_llm_judge` makes an API call but receives the `httpx.Client` as a parameter (never creates one) so it stays free of any runner dependency. Judge failures (API error, malformed JSON) return a result with `score=None` and a `status` flag — a pipeline failure is distinguished from a real score of 0. `weighted_match` returns `0.0` on unparseable candidate output (a task failure, not a pipeline error).
- **`estimate.py`** — offline cost estimation for `--dry-run`. Pure logic: given a config and pricing catalog from `openrouter.fetch_models`, estimates total cost without making any completion calls. Returns per-combo rows plus the `over_budget` verdict against `max_cost_usd`.
- **`cache.py`** — SHA256 file cache under `.evalblink_cache/<key>.json`. The cache key is the full request payload (model, messages, temperature, max_tokens). `temperature: 0` + cache = reproducible runs.
- **`reporter.py`** — all output (terminal Rich table, JSON, Markdown to `results/`). Formats only; makes no decisions. `write` is the single entry point for a run; `render_comparison` formats the `compare` delta table; `render_estimate` formats the `--dry-run` output. Filenames are slugified. JSON records carry a `schema_version` (from `schemas.SCHEMA_VERSION`).
- **`compare.py`** — pure run-to-run diff (no I/O beyond `load_record`). `diff` keys combos by `(model, prompt_id)`, emits per-combo quality/cost deltas (latency is not tracked, so it's out of scope), and tolerates schema-version mismatch rather than hard-blocking. `detailed_diff` (the `compare --detailed` mode) drills into per-test-case transitions — `case_diff` classifies each case as regressed/improved/new_error/recovered/etc. (reusing the `match_score is None` = pipeline-error rule, never a real fail) and aggregates a global summary (change counts, worst-regressed cases across combos, per-tag net).
- **`schemas.py`** — `TypedDict`s documenting the config and result dict shapes. **Documentation/typing only, zero runtime effect** — the pipeline passes plain dicts straight from the YAML. Runtime validation is `validator.py`'s job.

## Benchmark config schema (as actually consumed)

Configs live in `benchmarks/*.yaml`. The structure the code reads:

- **Templates use Jinja2** — `{{ variable }}`, NOT `{variable}`. Single braces render literally and silently produce wrong prompts (a real bug class). `evalblink validate` warns on this pattern. Variables come from the global `variables:` block merged with each `test_case.variables`.
- **`inference:`** `temperature`, `max_tokens`. Reasoning models (e.g. `poolside/laguna`) consume the budget on reasoning tokens and return null content if `max_tokens` is too low — give headroom.
- **`evaluation:` block is per-mode:**
  - `exact_match` cases need no `evaluation:` block at all.
  - `llm_judge` requires `evaluation.judge_model` (+ optional `judge_threshold`, default 0.70). Missing `judge_model` raises `ValueError`.
  - `weighted_match` requires `evaluation.quality_threshold` and `evaluation.variables` (list of `{name, weight, tolerance?}`); expected/actual outputs are JSON arrays of `{use_case, percent, order}`.
- **`concurrency`** — optional top-level int (default 5); controls `ThreadPoolExecutor` worker count.
- **`max_cost_usd`** — optional top-level float; `--dry-run` exits 1 if estimated cost exceeds it.
- **`test_cases[].evaluation`** selects the scorer: `"exact_match"` | `"weighted_match"` | `"llm_judge"`. A single config may mix modes. `exact_match` needs `expected_output`; `llm_judge` needs `criteria`; `tags` drive per-category breakdown.

When fixing a config, keep changes in the YAML and don't alter scoring semantics in `evaluator.py`/`runner.py` unless a run reveals a genuine code defect.
