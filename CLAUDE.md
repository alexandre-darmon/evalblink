# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

evalblink benchmarks LLM prompts × models on your own data, routing every call through [OpenRouter](https://openrouter.ai). A run takes a YAML benchmark config, executes `model × prompt × test_case`, scores each response, and writes a results matrix (Rich table), a JSON record, and a Markdown report.

## ⚠️ README is aspirational, not the implementation

`README.md` and `PRD.md` describe a planned product (`evalblink run`, `compare`, `history`, `--dry-run`, concurrency, `max_cost_usd`, top-level `judge_model`/`cache` keys). **Most of that does not exist yet.** Trust the code and the working configs in `benchmarks/`, not the README, for current behavior. Notably the README's YAML examples use the wrong template syntax (`{var}`) and a top-level `judge_model` — neither matches the implementation (see below).

## Commands

```bash
# Setup
pip install -r requirements.txt          # or use the existing .venv/
export OPENROUTER_API_KEY=...             # or put it in .env (loaded via python-dotenv)

# Run a benchmark (this IS the CLI — there is no `evalblink` console script)
python -m evalblink.main run benchmarks/classification.yaml
python -m evalblink.main run             # defaults to benchmarks/exact_match_classification.yaml
python -m evalblink.main run benchmarks/classification.yaml -v   # verbose: per-test-case detail

# Compare two finished runs (quality + cost deltas; offline, no API calls)
python -m evalblink.main compare results/<run_a>.json results/<run_b>.json
python -m evalblink.main compare results/<run_a>.json results/<run_b>.json --detailed  # per-test-case changes + global summary

# Tests
pytest -q

# Lint (ruff is the only dev tool; no config file, uses defaults)
ruff check .
ruff format .
```

Tests live in `tests/` and run offline (a fake `httpx.Client`, no API key). Beyond unit tests, verify end-to-end by running configs and inspecting the generated `results/<timestamp>_<slug>.{json,md}` files. Runs hit the live OpenRouter API (free models cost $0 but are rate-limited) and `runner.run` sleeps 5s after every *uncached* call.

## Architecture

A run flows through one linear pipeline; module boundaries are deliberate, not incidental:

- **`main.py`** — `argparse` subcommand dispatcher (`load_dotenv()` then `args.func`). `run <config> [-v]` → `runner.run` → `reporter.write` → CI gate `sys.exit(0|1)`; `compare <a.json> <b.json>` → `compare.load_record` ×2 → `compare.diff` → `reporter.render_comparison`. `load_config` exits with a clear message on missing/invalid YAML.
- **`runner.py`** — owns the single `httpx.Client` and the `model × prompt × test_case` triple loop. Renders templates, calls OpenRouter for each candidate, routes the response to the right evaluator, accumulates per-prompt token/cost totals. Sequential. Rate-limits with `time.sleep(5)` only on non-cached calls. Per-case detail prints are gated behind `verbose`.
- **`openrouter.py`** — the *single API choke point*. Every network call (candidate responses AND the LLM judge) goes through `openrouter_request`. It lives in its own leaf module specifically to avoid a `runner ↔ evaluator` import cycle. Responsibilities: build the request, consult the SHA256 cache, **retry transient failures** (`RETRYABLE_CODES = {408,429,502,503,504}` + local `httpx.TimeoutException`, `MAX_RETRIES` attempts, exponential backoff), raise `RuntimeError` on non-transient API errors or null content.
- **`evaluator.py`** — three scorers: `exact_match`, `weighted_match`, `evaluate_llm_judge`, plus `_extract_json` (strips markdown code fences before parsing). Pure scoring; `evaluate_llm_judge` makes an API call but receives the `httpx.Client` as a parameter (never creates one) so it stays free of any runner dependency. Judge failures (API error, malformed JSON) return a result with `score=None` and a `status` flag — a pipeline failure is distinguished from a real score of 0. `weighted_match` returns `0.0` on unparseable candidate output (a task failure, not a pipeline error).
- **`cache.py`** — SHA256 file cache under `.evalblink_cache/<key>.json`. The cache key is the full request payload (model, messages, temperature, max_tokens). `temperature: 0` + cache = reproducible runs.
- **`reporter.py`** — all output (terminal Rich table, JSON, Markdown to `results/`). Formats only; makes no decisions. `write` is the single entry point for a run; `render_comparison` formats the `compare` delta table. Filenames are slugified. JSON records carry a `schema_version` (from `schemas.SCHEMA_VERSION`).
- **`compare.py`** — pure run-to-run diff (no I/O beyond `load_record`). `diff` keys combos by `(model, prompt_id)`, emits per-combo quality/cost deltas (latency is not tracked, so it's out of scope), and tolerates schema-version mismatch rather than hard-blocking. `detailed_diff` (the `compare --detailed` mode) drills into per-test-case transitions — `case_diff` classifies each case as regressed/improved/new_error/recovered/etc. (reusing the `match_score is None` = pipeline-error rule, never a real fail) and aggregates a global summary (change counts, worst-regressed cases across combos, per-tag net). Mirrors the `analysis`/`reporter` decisions-vs-formatting split.
- **`schemas.py`** — `TypedDict`s documenting the config and result dict shapes. **Documentation/typing only, zero runtime effect** — the pipeline passes plain dicts straight from the YAML.

## Benchmark config schema (as actually consumed)

Configs live in `benchmarks/*.yaml`. The structure the code reads:

- **Templates use Jinja2** — `{{ variable }}`, NOT `{variable}`. Single braces render literally and silently produce wrong prompts (a real bug class — see `benchmarks/exact_match_classification.yaml` for the correct form). Variables come from the global `variables:` block merged with each `test_case.variables`.
- **`inference:`** `temperature`, `max_tokens`. Reasoning models (e.g. `poolside/laguna`) consume the budget on reasoning tokens and return null content if `max_tokens` is too low — give headroom.
- **`evaluation:` block is per-mode** (the README's top-level `judge_model` is wrong):
  - `exact_match` cases need no `evaluation:` block at all.
  - `llm_judge` requires `evaluation.judge_model` (+ optional `judge_threshold`, default 0.70). Missing `judge_model` raises `ValueError`.
  - `weighted_match` requires `evaluation.quality_threshold` and `evaluation.variables` (list of `{name, weight, tolerance?}`); expected/actual outputs are JSON arrays of `{use_case, percent, order}`.
- **`test_cases[].evaluation`** selects the scorer: `"exact_match"` | `"weighted_match"` | `"llm_judge"`. A single config may mix modes. `exact_match` needs `expected_output`; `llm_judge` needs `criteria`; `tags` drive per-category breakdown.

When fixing a config, keep changes in the YAML and don't alter scoring semantics in `evaluator.py`/`runner.py` unless a run reveals a genuine code defect.
