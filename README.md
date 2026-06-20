# evalblink

**Benchmark LLM prompts × models — quality and cost in one command.**

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
                  Customer Support Classification — 2026-06-10_14-32_customer-support-classification
 Model                         Prompt  Score   Pass/Scored  Errors  Prompt tok  Completion tok  Cost
 anthropic/claude-sonnet-4-6   v2      96.0%   48/50        0       5120        344             $0.000471
 anthropic/claude-sonnet-4-6   v1      94.0%   47/50        0       4200        312             $0.000445
 openai/gpt-4o-mini            v2      91.0%   45/50        0       5010        321             $0.000071
 openai/gpt-4o-mini            v1      87.0%   43/50        1       4100        298             $0.000063
 mistralai/mistral-small-3     v2      88.0%   44/50        0       4900        308             $0.000042
 mistralai/mistral-small-3     v1      82.0%   41/50        2       4000        289             $0.000038

       QUALITY BY TAG — best combo (anthropic/claude-sonnet-4-6 / v2)
 Tag          Cases  Quality  Errors
 easy         30     99%      0
 medium       15     91%      0
 edge_case    5      60%      0     ⚠️
 billing      12     63%      1     ⚠️
 non_english  4      50%      0     ⚠️

RECOMMENDATION
Best quality : anthropic/claude-sonnet-4-6 / v2 (96.0%)
Best value   : mistralai/mistral-small-3 / v2 (88.0% · $0.0000/run)
⚠️  Warning  : edge_case quality is below 70%.
⚠️  Warning  : billing quality is below 70%.
⚠️  Warning  : non_english quality is below 70%.

Results saved to results/2026-06-10_14-32_customer-support-classification.json
Markdown report saved to results/2026-06-10_14-32_customer-support-classification.md
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
quality_threshold: 85        # CI/CD gate: exits 1 if best combo score < 85%
max_cost_usd: 10.00          # --dry-run exits 1 if estimated cost exceeds this

models:
  - "openai/gpt-4o-mini"
  - "anthropic/claude-sonnet-4-6"
  - "mistralai/mistral-small-3"

inference:
  temperature: 0
  max_tokens: 50

evaluation:
  judge_model: "openai/gpt-4o"
  judge_threshold: 0.70      # per-case pass threshold for llm_judge (default: 0.70)

prompts:
  - id: "v1"
    template: >
      Classify this conversation.
      Return only the label — no JSON, no explanation.
      Choose from: {{ labels }}.
      Conversation: {{ conversation }}

  - id: "v2"
    system: "You are an expert classification assistant."
    template: >
      Classify the following conversation.
      Return exactly one label — no JSON, no explanation.
      Choose from: {{ labels }}.
      Conversation: {{ conversation }}

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
| `models` | ✅ | List of OpenRouter model IDs |
| `quality_threshold` | — | Top-level CI/CD gate: `evalblink run` exits 1 if the best combo score falls below this percentage (0–100) |
| `max_cost_usd` | — | `--dry-run` exits 1 if the estimated cost exceeds this amount |
| `concurrency` | — | Max parallel API requests (default: `5`) |
| `inference.temperature` | — | Sampling temperature (default: `0` for reproducibility) |
| `inference.max_tokens` | — | Max output tokens per call (default: `4096`) |
| `evaluation.judge_model` | ✅ if `llm_judge` used | OpenRouter model ID used as the judge |
| `evaluation.judge_threshold` | — | Per-case pass threshold for `llm_judge` — normalized score 0–1 (default: `0.70`) |
| `evaluation.quality_threshold` | ✅ if `weighted_match` used | Per-case pass threshold for `weighted_match` — weighted score 0–1 |
| `evaluation.variables` | ✅ if `weighted_match` used | Weighted dimension definitions (`name`, `weight`, optional `tolerance`) |
| `prompts[].id` | ✅ | Unique identifier used in reports |
| `prompts[].template` | ✅ | Jinja2 template — use `{{ variable }}` (double braces) |
| `prompts[].system` | — | Optional system message (also a Jinja2 template) |
| `variables` | — | Global key/value pairs injected into all templates |
| `test_cases[].id` | ✅ | Unique identifier |
| `test_cases[].evaluation` | ✅ | `exact_match` \| `llm_judge` \| `weighted_match` |
| `test_cases[].expected_output` | ✅ if `exact_match` or `weighted_match` | Ground truth (string for `exact_match`, JSON array for `weighted_match`) |
| `test_cases[].criteria` | ✅ if `llm_judge` | Evaluation rubric shown to the judge |
| `test_cases[].reference` | — | Optional gold answer injected into the judge prompt |
| `test_cases[].variables` | — | Per-case variables that override or extend global `variables` |
| `test_cases[].tags` | — | Free-form strings for per-category result breakdown |

> **Threshold naming:** `quality_threshold` at the top level is the CI/CD gate (0–100 %). `evaluation.quality_threshold` is the per-case pass threshold for `weighted_match` (0–1). `evaluation.judge_threshold` is the per-case pass threshold for `llm_judge` (0–1, default 0.70). These are three distinct concepts.

---

## CLI Reference

```bash
# Run a benchmark
evalblink run benchmarks/classification.yaml
evalblink run benchmarks/classification.yaml -v          # verbose: per-case detail
evalblink run benchmarks/classification.yaml --no-cache  # bypass local cache
evalblink run benchmarks/classification.yaml --dry-run   # estimate cost, no API calls

# Validate a config file (errors and warnings, no API calls)
evalblink validate benchmarks/classification.yaml

# Compare two runs (quality + cost deltas)
evalblink compare results/run_001.json results/run_002.json
evalblink compare results/run_001.json results/run_002.json --detailed  # per-case transitions

# Regenerate a Markdown report from an existing JSON result
evalblink report results/run_001.json

# List past runs
evalblink history

# Browse available models from OpenRouter
evalblink models
evalblink models --provider anthropic
evalblink models --free
evalblink models --min-context 100k

# Scaffold a new benchmark interactively
evalblink init

# Cache management
evalblink cache stats
evalblink cache clear --yes
```

---

## Evaluation Modes

### Exact match

For classification tasks where the model must return a specific label. Prompt the model explicitly to return no JSON and no extra text.

```yaml
# Full example: benchmarks/exact_match_classification.yaml
name: "Customer Support Classification"
inference:
  temperature: 0
  max_tokens: 50

models:
  - "google/gemma-4-31b-it:free"

variables:
  labels: "order_issue, billing, product_question, other"

prompts:
  - id: "v1"
    template: >
      Classify this conversation.
      Return only a valid label from the list. No JSON. No explanation.
      Choose from: {{ labels }}.
      Conversation: {{ conversation }}

test_cases:
  - id: "conv_001"
    variables:
      conversation: "I cannot find my last order."
    expected_output: "order_issue"
    evaluation: "exact_match"
    tags: ["order", "easy"]
```

Case-insensitive string comparison after whitespace trimming. The model must return just the label text. If it wraps the output in JSON or adds any explanation, the match fails.

### LLM-as-judge

For open-ended tasks (summarization, generation, explanation) where no single correct output exists.

```yaml
# Full example: benchmarks/llm_as_judge.yaml
name: "Customer Support Quality"
inference:
  temperature: 0
  max_tokens: 1024

evaluation:
  judge_model: "openai/gpt-4o"
  judge_threshold: 0.70        # optional, default 0.70

models:
  - "google/gemma-4-31b-it:free"

prompts:
  - id: "v1"
    system: >
      You are Paul, a senior customer support agent.
      Acknowledge the customer's emotion before proposing a solution.
      Keep your response under 100 words.
    template: "Customer message: {{ customer_message }}"

test_cases:
  - id: "conv_001"
    variables:
      customer_message: "This is the third time my order arrived damaged."
    evaluation: "llm_judge"
    criteria: >
      The assistant must acknowledge frustration before proposing resolution.
      A concrete next step must be proposed (refund, replacement, escalation).
    tags: ["frustration", "edge_case"]
```

The judge reasons step by step before committing to a score (chain-of-thought-first, G-Eval / MT-Bench style). The judge returns a score of 1–5, normalized to 0–1. Cases scoring below `judge_threshold` count as failures. Judge reasoning surfaces in the report so you know *why* a response scored poorly.

**Built-in bias mitigations:**
- Judge prompt instructs against rewarding length (verbosity bias: 10–20% magnitude)
- Judge prompt instructs against rewarding confident tone over accuracy (style bias)
- If `evaluation.judge_model` shares a vendor with any candidate model, evalblink warns you at run start (self-preference bias: 10–25% magnitude)

### Weighted match

For structured outputs where the model must return a JSON array of `{"use_case", "percent", "order"}` objects. Three dimensions are scored independently and combined by weight.

```yaml
# Full example: benchmarks/weighted_match_config.yaml
name: "ChatGPT Usage Classification"
inference:
  temperature: 0
  max_tokens: 4096

evaluation:
  quality_threshold: 0.80      # per-case pass threshold (0–1)
  variables:
    - name: "use_case"
      weight: 0.50             # F1 score on matched labels
    - name: "percent"
      weight: 0.25
      tolerance: 0.20          # percent within ±0.20 of expected counts as match
    - name: "order"
      weight: 0.25             # exact order match

models:
  - "poolside/laguna-xs.2:free"

variables:
  use_cases: "Translate content, Analyze content, Summarize content, Write content, Other"

prompts:
  - id: "v1"
    system: >
      Analyse this conversation and identify the use cases present.
      Return only valid JSON, no explanation, no markdown.
      Format: [{"use_case": "<label>", "percent": <float>, "order": <int>}]
      Percentages must sum to 1.0.
      Available use cases: {{ use_cases }}.
    template: "Conversation: {{ conversation }}"

test_cases:
  - id: "conv_001"
    variables:
      conversation: |
        User: Summarize this document.
        User: Translate it to English.
    expected_output:
      - use_case: "Summarize content"
        percent: 0.70
        order: 1
      - use_case: "Translate content"
        percent: 0.30
        order: 2
    evaluation: "weighted_match"
    tags: ["summarize", "translate"]
```

Returns a weighted score 0.0–1.0. Cases scoring below `evaluation.quality_threshold` are counted as failures.

**Scoring per dimension:**
- `use_case`: F1 score (precision × recall) on matched labels
- `percent`: fraction of expected items within `tolerance` of the expected value
- `order`: fraction of expected items with an exact order match

---

## Cost Estimation

Before running a benchmark, estimate the cost without making any API calls:

```bash
evalblink run benchmarks/classification.yaml --dry-run
```

```
DRY RUN — estimated cost (no API calls)
 Model                         Prompt  Est prompt tok  Est completion tok  Est cost
 anthropic/claude-sonnet-4-6   v1      3842            200                 $0.001230
 anthropic/claude-sonnet-4-6   v2      4610            200                 $0.001476
 openai/gpt-4o-mini            v1      3842            200                 $0.000210
 openai/gpt-4o-mini            v2      4610            200                 $0.000252

Estimated total cost: $0.003168
Estimate only — completion tokens assume max_tokens; prompt tokens ≈ chars/4.
✓ Within budget: estimate $0.003168 ≤ max_cost_usd $10.000000.
```

Set `max_cost_usd` in the YAML to enforce a budget: `--dry-run` exits 1 if the estimate exceeds it, which makes it safe to gate in CI before committing to a full run.

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
evalblink cache stats    # entry count and total size on disk
evalblink cache clear    # wipe cache (--yes to skip confirmation)
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
 DELTA: 2026-06-08_run_001 → 2026-06-10_run_002
 Model                         Prompt  Quality Δ    Cost Δ       Notes
 anthropic/claude-sonnet-4-6   v1      +4.0% ↑      +0.0002 ↑
 anthropic/claude-sonnet-4-6   v2      stable        stable
 openai/gpt-4o-mini            v1      +6.0% ↑      stable

Verdict: B improved on A across 2/3 combos.
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
| Self-preference detection | ✅ warning at run start | ❌ |
| Verbosity + style bias guardrails | ✅ baked into judge prompt | ❌ |
| Reference-guided grading | ✅ optional `reference` field | ❌ |
| Per-tag result breakdown | ✅ with ⚠️ auto-warnings | ❌ |
| Judge calibration vs human labels | V1.1 `evalblink calibrate` | ❌ |
| Pairwise judge mode | V1.1 with order-swap tie detection | ❌ |
| Caching | ✅ | ✅ |
| Concurrency | ✅ | ✅ |
| CI/CD | ✅ | ✅ |
| Red teaming | ❌ | ✅ |
| Web UI | V2 | ✅ |

---

## Roadmap

**V1 — CLI (current)**
- Exact match + LLM-as-judge + weighted match evaluation
- Per-tag result stratification with auto-warnings
- Self-preference, verbosity, and style bias mitigations
- Reference-guided grading
- Concurrent API calls (`ThreadPoolExecutor`), exponential backoff retry
- Local caching, schema versioning, run compare (`--detailed` per-case transitions)
- `evalblink validate` — static config validation, no API calls
- `evalblink init` — interactive YAML scaffolding
- `evalblink models` — live OpenRouter model catalog with filtering
- `/docs/skills/` methodology guides

**V1.1**
- `evalblink calibrate` — validate judge scores against human labels (Pearson + Cohen's Kappa)
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
