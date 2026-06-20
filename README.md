# evalblink

**Benchmark LLM prompts × models — quality, cost, and latency in one command.**

Vendor-neutral Python CLI to benchmark LLM prompts and models on your own data. No JavaScript, no vendor lock-in.

```bash
pip install evalblink
evalblink run benchmarks/classification.yaml
```

---

## Why evalblink

Most LLM eval tools are built for JavaScript developers, locked to a single vendor, or require you to wrap your pipeline in their SDK. evalblink is different.

- **Python-native.** `pip install` and a YAML file. No Node.js, no boilerplate.
- **Vendor-neutral.** Routes through [OpenRouter](https://openrouter.ai) — one API key, every major model. Not owned by any vendor you're evaluating.
- **Your data, not public benchmarks.** Public benchmarks are contaminated — models are trained on the same corpora that include the benchmarks themselves. evalblink runs on your prompts, your inputs, your labels.
- **Cost-aware.** Every run reports token cost per combination, including judge calls. Dry-run estimates before you spend a dollar.

> PromptFoo was the reference standard for structured LLM evaluation. It was acquired by OpenAI in March 2026. Enterprise teams in regulated industries need a neutral alternative. evalblink is that alternative.

---

## What it does

Define a benchmark in YAML. Run one command. Get a results matrix, a versioned Markdown report, and a local cache so prompt iteration costs almost nothing.

```
evalblink — Customer Support Classification
Run: 2026-06-10 14:32 | ID: a3f2b1 | Cache hits: 102/300

              | gpt-4o-mini         | claude-sonnet       | mistral-small-3
prompt_v1     | 87% · $0.06 · 1.1s  | 94% · $0.45 · 2.3s  | 82% · $0.04 · 0.9s
prompt_v2     | 91% · $0.07 · 1.2s  | 96% · $0.47 · 2.1s  | 88% · $0.04 · 0.8s
              | +4% ↑               | +2% ↑               | +6% ↑

Judge cost (llm_judge cases): $0.18
Total run cost: $1.05

QUALITY BY TAG — best combination (claude-sonnet / prompt_v2)
tag             | cases | quality
----------------|-------|--------
easy            | 30    | 99%
medium          | 15    | 91%
edge_case       | 5     | 60%   ⚠️
billing         | 12    | 63%   ⚠️
non_english     | 4     | 50%   ⚠️

RECOMMENDATION
Best quality : claude-sonnet / prompt_v2 (96% overall)
Best value   : mistral-small-3 / prompt_v2 (88% quality · $0.04/run)
⚠️  Warning   : edge_case quality is 60% — review before shipping.
```

---

## Quickstart

**1. Install**

```bash
pip install evalblink
```

> **Local development (from the repo root):**
> ```bash
> pip install -e ".[dev]"
> ```

**2. Set your API key**

```bash
export OPENROUTER_API_KEY=your_key_here
```

**3. Write a benchmark**

```yaml
# benchmarks/classification.yaml
name: "Customer Support Classification"
description: "Compare 2 prompt versions across 3 models on 50 labelled conversations"
judge_model: "anthropic/claude-sonnet-4-6"
max_cost_usd: 10.00
quality_threshold: 85
cache: true

inference:
  temperature: 0
  max_tokens: 50

prompts:
  - id: "v1"
    template: >
      Classify this conversation. Return only valid JSON: {"label": "<label>"}.
      Choose from: {labels}.
      Conversation: {conversation}

  - id: "v2"
    system: "You are an expert classification assistant."
    template: >
      Analyze the following conversation and return only valid JSON: {"label": "<label>"}.
      Choose exactly one label from: {labels}.
      Conversation: {conversation}

models:
  - "openai/gpt-4o-mini"
  - "anthropic/claude-sonnet-4-6"
  - "mistralai/mistral-small-3"

variables:
  labels: "order_issue, billing, product_question, other"

test_cases:
  - id: "conv_001"
    variables:
      conversation: "I cannot find my last order."
    expected_output: "order_issue"
    evaluation: "exact_match"
    tags: ["order", "easy"]

  - id: "conv_002"
    variables:
      conversation: "What is included in my subscription plan?"
    evaluation: "llm_judge"
    criteria: >
      Should identify a billing or subscription inquiry.
      Must return a single label, no explanation.
    reference: "billing"
    tags: ["billing", "medium"]

  - id: "conv_003"
    variables:
      conversation: "I was charged twice but the app shows one order."
    expected_output: "billing"
    evaluation: "exact_match"
    tags: ["billing", "edge_case"]

  - id: "conv_004"
    variables:
      conversation: "Bonjour, je ne trouve pas ma commande."
    expected_output: "order_issue"
    evaluation: "exact_match"
    tags: ["order", "edge_case", "non_english"]
```

**4. Run**

```bash
evalblink run benchmarks/classification.yaml
```

---

## Configuration Reference

| Field | Required | Description |
|---|---|---|
| `name` | ✅ | Benchmark display name |
| `description` | — | Optional context |
| `judge_model` | ✅ if `llm_judge` used | OpenRouter model ID for evaluation |
| `max_cost_usd` | — | Hard stop if total estimated cost (models + judge) exceeds limit |
| `quality_threshold` | — | Minimum quality % for CI/CD pass |
| `cache` | — | Enable local response caching (default: `true`) |
| `concurrency` | — | Max parallel API requests (default: `5`) |
| `inference.temperature` | — | Sampling temperature (default: `0` for reproducibility) |
| `inference.max_tokens` | — | Max output tokens per call (default: `100`) |
| `prompts[].id` | ✅ | Unique identifier used in reports |
| `prompts[].template` | ✅ | Prompt template using `{variable}` placeholders |
| `prompts[].system` | — | Optional system message |
| `models` | ✅ | List of OpenRouter model IDs |
| `variables` | — | Global variables injected into all templates |
| `test_cases[].evaluation` | ✅ | `exact_match` or `llm_judge` |
| `test_cases[].expected_output` | ✅ if `exact_match` | Ground truth label |
| `test_cases[].criteria` | ✅ if `llm_judge` | Evaluation rubric for the judge |
| `test_cases[].reference` | — | Optional gold reference answer injected into the judge prompt |
| `test_cases[].tags` | — | Free-form strings for per-category result breakdown |
| `test_cases[].human_score` | — | Integer 1–5 from a human reviewer, used by `evalblink calibrate` (V1.1) |

---

## CLI Reference

```bash
# Run a benchmark
evalblink run benchmarks/classification.yaml

# Run all benchmarks in a folder
evalblink run benchmarks/

# Estimate cost without calling APIs (includes judge cost)
evalblink run benchmarks/classification.yaml --dry-run

# Force fresh API calls, bypass cache
evalblink run benchmarks/classification.yaml --no-cache

# Compare two runs (delta view)
evalblink compare results/run_001.json results/run_002.json

# Regenerate a Markdown report from an existing JSON result
evalblink report results/run_001.json

# List past runs
evalblink history

# Browse available models from OpenRouter
evalblink available-models
evalblink available-models --provider mistral --min-context 100k

# Scaffold a new benchmark interactively
evalblink init

# Cache management
evalblink cache stats
evalblink cache clear

# [V1.1] Validate judge calibration against human-labelled cases
evalblink calibrate results/run_001.json
```

---

## Evaluation Modes

### Exact match

For classification and structured output tasks. The model is instructed to return a specific JSON structure — evalblink parses it deterministically. No regex, no fragile text extraction.

```yaml
evaluation: "exact_match"
expected_output: "billing"
```

If the model returns malformed JSON, the case is scored 0 and logged under "Parse Errors" — separate from correct/incorrect results.

### LLM-as-judge

For open-ended tasks (summarization, generation, explanation) where no single correct output exists.

```yaml
evaluation: "llm_judge"
criteria: "Should identify a billing inquiry. Must return a single label."
reference: "billing"   # optional — grounds the judge in a known-good answer
```

The judge reasons step by step before committing to a score — this chain-of-thought-first approach (G-Eval, MT-Bench) produces more calibrated results than scoring first and justifying retroactively. Judge reasoning surfaces in the report so you know *why* a response scored poorly.

**Built-in bias mitigations:**
- Judge prompt instructs against rewarding length (verbosity bias: 10–20% magnitude)
- Judge prompt instructs against rewarding confident tone over accuracy (style bias)
- If `judge_model` shares a vendor with any candidate model, evalblink warns you at run start (self-preference bias: 10–25% magnitude)

---

## Test Case Stratification

Add optional `tags` to any test case. evalblink groups results by tag and surfaces per-category quality — because a 91% aggregate can hide 55% on your edge cases.

```yaml
test_cases:
  - id: "conv_001"
    tags: ["order", "easy"]

  - id: "conv_003"
    tags: ["billing", "edge_case"]

  - id: "conv_004"
    tags: ["order", "edge_case", "non_english"]
```

Recommended distribution: ~50% easy, ~30% medium, ~20% edge cases. Any tag scoring below 70% in the best combination triggers an automatic warning in the CLI and report.

---

## Concurrency

API calls run concurrently using `ThreadPoolExecutor` from Python's standard library — no async complexity required.

```yaml
concurrency: 5    # max parallel requests (default: 5)
```

| Mode | Estimated duration (50 cases × 3 models × 2 prompts) |
|---|---|
| Sequential | ~8–12 minutes |
| Concurrent (default: 5) | ~45–90 seconds |

Failed API calls are retried with exponential backoff (up to 3 attempts). After 3 failures, the test case is scored `null` and listed under "API Errors" in the report.

---

## Caching

Every API response is cached locally by SHA256 key including model, rendered system prompt, rendered prompt, temperature, and max_tokens. When you iterate on prompt v2, all v1 responses are served from cache at zero cost.

```bash
evalblink cache stats    # hit rate from last run
evalblink cache clear    # wipe cache
```

> Cache + `temperature: 0` = fully reproducible benchmarks. evalblink warns you if you enable cache with temperature > 0.

---

## Results & History

Every run produces two files:

```
results/
  2026-06-10_14-32_classification_a3f2b1.json   # full data, git-friendly
  2026-06-10_14-32_classification_a3f2b1.md     # human-readable report
```

The JSON includes schema versioning so `evalblink compare` handles old files gracefully — it never hard-blocks on a version mismatch. Regenerate the Markdown report at any time:

```bash
evalblink report results/run_001.json
```

```bash
evalblink compare results/run_001.json results/run_002.json
```

```
DELTA: run_001 → run_002

Model             | Quality    | Cost       | Latency
------------------|------------|------------|----------
gpt-4o-mini       | +4% ↑      | +$0.01     | +0.1s
claude-sonnet     | +2% ↑      | stable     | stable
mistral-small-3   | +6% ↑      | stable     | -0.1s ↓

Verdict: prompt_v2 improves quality across all models.
         mistral-small-3: +6% quality at no additional cost.
```

---

## CI/CD Integration

evalblink exits `0` (pass) or `1` (fail) based on `quality_threshold`.

```yaml
# .github/workflows/llm-eval.yml
- run: evalblink run benchmarks/classification.yaml
  env:
    OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
```

---

## Evaluation Methodology — `/docs/skills/`

evalblink ships four practitioner guides covering the full evaluation workflow. No code required.

| Guide | What it covers |
|---|---|
| `01_design_your_test_set.md` | Golden set design, 50/30/20 distribution, contamination risk, tag taxonomy |
| `02_write_a_testable_prompt.md` | Deterministic output formats, variable isolation, common anti-patterns |
| `03_read_your_results.md` | Interpreting the results matrix, parse errors, aggregate vs subgroup scores |
| `04_refine_your_prompt.md` | Using `evalblink compare`, one-variable iteration, when to expand your test set |

---

## vs PromptFoo

| | evalblink | PromptFoo |
|---|---|---|
| Language | Python | Node.js |
| Vendor neutrality | Independent | Acquired by OpenAI (2026) |
| Live model metadata | ✅ OpenRouter API | ❌ |
| Cost tracking incl. judge | ✅ | ❌ |
| Run comparison (delta) | ✅ | ❌ |
| Markdown report per run | ✅ | ❌ |
| JSON-enforced output | ✅ | ❌ |
| Self-preference detection | ✅ warning at run start | ❌ |
| Verbosity + style bias guardrails | ✅ baked into judge prompt | ❌ |
| Reference-guided grading | ✅ optional `reference` field | ❌ |
| Per-tag result breakdown | ✅ with ⚠️ auto-warnings | ❌ |
| Judge calibration vs human labels | ✅ V1.1 `evalblink calibrate` | ❌ |
| Pairwise judge mode | ✅ V1.1 with order-swap tie detection | ❌ |
| Caching | ✅ | ✅ |
| Concurrency | ✅ | ✅ |
| CI/CD | ✅ | ✅ |
| Red teaming | ❌ | ✅ |
| Web UI | V2 | ✅ |

---

## Roadmap

**V1 — CLI (current)**
- Exact match + LLM-as-judge evaluation
- Per-tag result stratification with auto-warnings
- Self-preference, verbosity, and style bias mitigations
- Reference-guided grading
- Concurrent API calls, exponential backoff retry
- Local caching, schema versioning, run compare
- `/docs/skills/` methodology guides

**V1.1**
- `evalblink calibrate` — validate judge scores against human labels (Pearson + Cohen's Kappa)
- `evalblink init` — interactive YAML scaffolding with correct tag distribution
- Pairwise judge mode with double-pass order-swap tie detection

**V2**
- Streamlit dashboard
- Multi-turn conversation test cases

**V3**
- `evalblink suggest` — hypothesis engine that diagnoses subgroup failures and proposes targeted prompt changes (human stays in the decision loop)

---

## Installation

```bash
pip install evalblink
```

Requires Python 3.10+. One environment variable: `OPENROUTER_API_KEY`.

---

## License

MIT
