# PRD — evalblink
## Multi-LLM Evaluation & Benchmark Kit

**Author:** Alexandre Darmon  
**Version:** 7.0  
**Status:** Living document  
**Last updated:** June 2026

> **Reading this document:** sections tagged `[V1 — shipped]` describe the current implementation. Sections tagged `[V1.1]`, `[V2]`, and `[V3]` describe planned future work. Nothing in future sections is implemented today.

---

## 1. Problem Statement

Enterprise teams building LLM-powered products face the same recurring challenge: making evidence-based decisions on a stack that changes constantly.

Three problems compound each other.

**Model selection.** Which model performs best on this specific use case — not on academic benchmarks, but on the team's actual prompts and data?

**Prompt iteration.** Which version of a prompt delivers the best quality/cost tradeoff across a given model? How does changing one sentence in a system prompt affect results at scale?

**Progress tracking.** How do you prove that the LLM product improved between sprint 3 and sprint 7? Without structured history, every evaluation starts from scratch.

The current process is manual, slow, and non-reproducible: copy-pasting prompts across interfaces, comparing outputs visually, making decisions based on intuition rather than data.

Tools like PromptFoo pioneered structured LLM evaluation but were built for JavaScript developers and acquired by OpenAI in March 2026. Enterprise data teams — working in Python, routing across vendors via OpenRouter, and operating in regulated industries — need a vendor-neutral, Python-native alternative with built-in cost estimation, live model metadata, and a versioned results workflow.

**evalblink** is that workflow.

> **On contamination:** public LLM benchmarks are increasingly unreliable — models are trained on web corpora that include the benchmarks themselves, inflating scores. evalblink is designed around the opposite principle: you bring your own data, your own prompts, your own labels. Evaluation runs on inputs the model has never seen in training, which is the only contamination-free evaluation that exists.

---

## 2. Target Users

**Primary:** AI Product Managers and Data Scientists in enterprise teams integrating LLMs — evaluating model selection, prompt iteration, or cost/quality tradeoffs.

**Secondary:** Developers building LLM-powered features who need evidence before committing to a model in production.

**Context:** Users are comfortable with a CLI and YAML files. They do not need to write Python code to use the tool.

---

## 3. Goals

### `[V1 — shipped]`

- Run a structured benchmark across N prompts × N models × N test cases in one command
- Evaluate responses via exact match, LLM-as-judge, or weighted match (three shipped modes)
- Measure quality and cost per combination — including judge costs; latency is intentionally not tracked
- Cache results locally to avoid redundant API calls during prompt iteration
- Run API calls concurrently using `ThreadPoolExecutor`
- Retry failed API calls with exponential backoff (manual retry loop, no external library)
- Estimate cost before a run with `--dry-run` (no API calls)
- Validate configs statically with `evalblink validate` (no API calls)
- Generate a versioned Markdown report and JSON record per run
- Store results history locally and compare runs with `evalblink compare` (quality + cost deltas; per-case transitions with `--detailed`)
- Browse live OpenRouter model catalog with pricing and context length via `evalblink models`
- Scaffold a new benchmark YAML interactively via `evalblink init`

### `[V1.1 — planned]`

- `evalblink calibrate` — validate judge scores against human-labelled cases (Pearson + Cohen's Kappa)
- Pairwise judge mode with double-pass order-swap tie detection

### `[V2 — planned]`

- Streamlit dashboard
- Multi-turn conversation test cases

### `[V3 — vision]`

- `evalblink suggest` — hypothesis engine that diagnoses subgroup failures and proposes targeted prompt changes (human stays in the decision loop)

---

## 4. Out of Scope (V1)

- Web UI / dashboard (→ V2)
- Latency measurement (intentionally excluded — varies with network conditions and adds noise to quality/cost comparisons)
- Red teaming or adversarial testing
- Real-time guardrails or production monitoring
- Authentication or multi-user support
- Cloud storage or remote result sync
- Fine-tuned model support

---

## 5. `[V1 — shipped]` Benchmark Configuration

Every benchmark is defined in a single YAML file.

### Minimal example (exact match)

```yaml
name: "Customer Support Classification"
quality_threshold: 85         # CI/CD gate: exit 1 if best combo score < 85%

inference:
  temperature: 0
  max_tokens: 50

models:
  - "openai/gpt-4o-mini"
  - "anthropic/claude-sonnet-4-6"

variables:
  labels: "order_issue, billing, product_question, other"

prompts:
  - id: "v1"
    template: >
      Classify this conversation.
      Return only the label — no JSON, no explanation.
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

### Mixed-mode example (exact match + llm_judge)

```yaml
name: "Customer Support Classification"
quality_threshold: 85
max_cost_usd: 10.00

inference:
  temperature: 0
  max_tokens: 200

evaluation:
  judge_model: "openai/gpt-4o"
  judge_threshold: 0.70

models:
  - "openai/gpt-4o-mini"
  - "anthropic/claude-sonnet-4-6"

variables:
  labels: "order_issue, billing, product_question, other"

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
      conversation: "Bonjour, je ne trouve pas ma commande."
    expected_output: "order_issue"
    evaluation: "exact_match"
    tags: ["order", "non_english"]
```

### Configuration reference

| Field | Required | Description |
|---|---|---|
| `name` | ✅ | Benchmark display name |
| `models` | ✅ | List of OpenRouter model IDs |
| `quality_threshold` | — | Top-level CI/CD gate: exits 1 if best combo score falls below this percentage (0–100) |
| `max_cost_usd` | — | `--dry-run` exits 1 if the estimated cost exceeds this amount |
| `concurrency` | — | Max parallel API requests (default: `5`) |
| `inference.temperature` | — | Sampling temperature (default: `0` for reproducibility) |
| `inference.max_tokens` | — | Max output tokens per call (default: `4096`) |
| `evaluation.judge_model` | ✅ if `llm_judge` used | OpenRouter model ID used as the judge |
| `evaluation.judge_threshold` | — | Per-case pass threshold for `llm_judge`, normalized 0–1 (default: `0.70`) |
| `evaluation.quality_threshold` | ✅ if `weighted_match` used | Per-case pass threshold for `weighted_match`, 0–1 |
| `evaluation.variables` | ✅ if `weighted_match` used | Weighted dimension definitions (`name`, `weight`, optional `tolerance`) |
| `prompts[].id` | ✅ | Unique identifier used in reports |
| `prompts[].template` | ✅ | Jinja2 template — use `{{ variable }}` (double braces) |
| `prompts[].system` | — | Optional system message (also a Jinja2 template) |
| `variables` | — | Global key/value pairs injected into all templates |
| `test_cases[].id` | ✅ | Unique identifier |
| `test_cases[].evaluation` | ✅ | `exact_match` \| `llm_judge` \| `weighted_match` |
| `test_cases[].expected_output` | ✅ if `exact_match` or `weighted_match` | Ground truth — string for `exact_match`, JSON array for `weighted_match` |
| `test_cases[].criteria` | ✅ if `llm_judge` | Evaluation rubric shown to the judge |
| `test_cases[].reference` | — | Optional gold answer injected into the judge prompt |
| `test_cases[].variables` | — | Per-case variables that override or extend global `variables` |
| `test_cases[].tags` | — | Free-form strings for per-category result breakdown |
| `test_cases[].human_score` | — | Integer 1–5 from a human reviewer — available for collection now, used by `evalblink calibrate` in V1.1 |

> **Three threshold fields, three distinct concepts.** `quality_threshold` at the top level (0–100 %) is the CI/CD gate on the best combo's aggregate score. `evaluation.quality_threshold` (0–1) is the per-case pass threshold for `weighted_match`. `evaluation.judge_threshold` (0–1) is the per-case pass threshold for `llm_judge`. All three can coexist in one config.

> **Self-preference warning:** if `evaluation.judge_model` shares a vendor prefix with any model in `models` (e.g. `anthropic/claude-*` judging `anthropic/claude-*`), evalblink emits a warning at run start. Self-preference bias runs 10–25% in magnitude — use a judge from a different model family when possible.

---

## 6. `[V1 — shipped]` Template Injection

Variables are injected into prompt templates using Jinja2.

```python
from jinja2 import Template
rendered = Template(template).render(**{**global_variables, **test_case_variables})
```

**Templates use double-brace syntax: `{{ variable }}`**. Single braces `{variable}` are treated as literal text and silently produce wrong prompts — `evalblink validate` warns on this pattern.

**Why Jinja2:** filter expressions (`{{ label | upper }}`, `{{ items | join(", ") }}`) are common in LLM prompt authoring and require a real template engine. `str.format_map()` cannot support them.

**Literal braces in prompt text:** wrap in a Jinja2 raw block — `{% raw %}{...}{% endraw %}` — or use backtick code blocks in the prompt instead.

---

## 7. `[V1 — shipped]` Evaluation Modes

A single config can mix all three modes across its test cases.

### 7.1 Exact Match

For classification tasks where the model must return a specific label.

**How it works:** case-insensitive string comparison after trimming whitespace.

```python
response.strip().lower() == expected_output.strip().lower()
```

No JSON parsing. The model must return just the label text. **Prompt it explicitly to return no JSON and no explanation** — if the response wraps the label in `{"label": "..."}` or adds any surrounding text, the match fails.

```yaml
test_cases:
  - id: "conv_001"
    evaluation: "exact_match"
    expected_output: "order_issue"
```

### 7.2 LLM-as-Judge

For open-ended tasks (summarization, generation, explanation) where no single correct output exists.

The judge receives the original task, the evaluation criteria, the model response, and optionally a reference answer:

```
You are an expert evaluator. Your goal is to assess quality, not style.
Original task: {prompt}
Evaluation criteria: {criteria}
Model response: {model_output}
[Reference answer: {reference}]   ← only included when test_cases[].reference is set

Think step by step about whether the response meets the criteria.

Important:
- Judge on criteria satisfaction only — do not reward length.
- Do not reward confident tone over accuracy.
- The criteria were not shown to the model being judged.

Then return ONLY valid JSON:
{"reasoning": "<your analysis>", "score": <integer 1-5>}
```

**Score normalization:** `(score - 1) / 4` → 0.0–1.0. A case passes if `normalized_score >= judge_threshold` (default 0.70, meaning a raw score ≥ 3.8 out of 5).

**Judge failures** (API error, malformed JSON) return `score=None` and a `status` flag — a pipeline failure is never conflated with a real score of 0.

```yaml
evaluation:
  judge_model: "openai/gpt-4o"
  judge_threshold: 0.70        # optional, default 0.70

test_cases:
  - id: "conv_002"
    evaluation: "llm_judge"
    criteria: >
      Should identify a billing inquiry.
      Must return a single label, no explanation.
    reference: "billing"       # optional
```

**Why reasoning first:** The judge writes its analysis before committing to a number — the chain-of-thought-first approach established by G-Eval (Liu et al., 2023) and MT-Bench produces more calibrated scores because the model reasons its way to a position rather than anchoring on a number and justifying retroactively. The `reasoning` field surfaces in the Markdown report, giving actionable signal on why a response scored poorly.

### 7.3 Weighted Match

For structured outputs where the model must return a JSON array of `{"use_case", "percent", "order"}` objects. Three dimensions are scored independently and combined by weight.

**Fixed output schema:** the expected and actual outputs must conform to this structure — `evaluation.variables` defines the weights and tolerances for `use_case`, `percent`, and `order`, not arbitrary field names.

```yaml
evaluation:
  quality_threshold: 0.80      # per-case pass threshold (0–1)
  variables:
    - name: "use_case"
      weight: 0.50             # F1 score on matched labels
    - name: "percent"
      weight: 0.25
      tolerance: 0.20          # |actual - expected| ≤ tolerance counts as match
    - name: "order"
      weight: 0.25             # exact order match

test_cases:
  - id: "conv_001"
    evaluation: "weighted_match"
    expected_output:
      - use_case: "Summarize content"
        percent: 0.70
        order: 1
      - use_case: "Translate content"
        percent: 0.30
        order: 2
```

**Scoring per dimension:**
- `use_case`: F1 score (2 × precision × recall / (precision + recall)) on matched label sets
- `percent`: fraction of expected items where the model's percent is within `tolerance` of the expected value
- `order`: fraction of expected items with an exact order match

**On parse failure:** if the model returns malformed JSON or the wrong shape, the case scores 0.0 — a task failure, not a pipeline error; the run continues.

---

## 8. `[V1 — shipped]` Judge Reliability & Bias Mitigation

LLM judges exhibit documented, quantified biases. evalblink addresses them in V1 with three built-in mitigations.

### Known biases

| Bias | Magnitude | Description |
|---|---|---|
| Self-preference | 10–25% | Judge favours responses from its own model family |
| Verbosity | 10–20% | Longer responses rated higher regardless of quality |
| Style | variable | Confident tone rewarded even when content is wrong |
| Position | 5–15% | Judge favours the response appearing first in pairwise prompts |

*Sources: MT-Bench (Zheng et al., 2024), Judging the Judges (Gu et al., 2025).*

### `[V1 — shipped]` Self-preference detection

At run start, evalblink compares the vendor prefix of `evaluation.judge_model` against every model in `models`. If any candidate shares the same vendor as the judge, a warning is emitted before the first API call:

```
⚠️  Self-preference risk: judge 'anthropic/claude-sonnet-4-6' shares vendor
    'anthropic' with candidate(s): anthropic/claude-haiku-4-5.
    Judge scores for these may be inflated.
```

The run is not blocked — sometimes no better judge is available — but the warning is always shown.

### `[V1 — shipped]` Verbosity and style bias — judge prompt guardrails

The judge prompt instructs the model to:
- Judge on criteria satisfaction only, not on length
- Not reward confident tone over accuracy

These instructions are embedded in the judge prompt and are not user-configurable, ensuring they are always active.

### `[V1 — shipped]` Rubric isolation

`criteria` is a test-case-level field, never injected into the candidate's `template`. This prevents rubric-hacking — where a model learns to match rubric keywords superficially rather than meeting the actual quality bar.

### `[V1 — shipped]` Reference-guided grading

When `test_cases[].reference` is provided, the gold reference answer is injected into the judge prompt. This grounds evaluation in truth rather than plausibility, reducing the risk of hallucination blind spots where a fluent but incorrect answer scores well.

### `[V1.1 — planned]` Judge calibration — `evalblink calibrate`

LLM judge scores are only meaningful if they correlate with human judgment. `evalblink calibrate` computes Pearson correlation and Cohen's Kappa between judge scores and human-labelled `human_score` fields. Recommended thresholds: Pearson ≥ 0.7, Cohen's Kappa ≥ 0.6. Below 0.4, the rubric is likely ambiguous and needs revision.

The `human_score` field is available in the schema from V1 so teams can start collecting labels immediately.

```bash
evalblink calibrate results/run_001.json
```

### `[V1.1 — planned]` Pairwise judge mode

Pairwise evaluation — "which of these two responses better satisfies the criteria?" — produces higher inter-rater agreement than pointwise 1–5 scoring because it simplifies the cognitive task to relative comparison. This maps naturally onto evalblink's core use case: comparing prompt v1 vs v2 on the same model.

Position bias (5–15%) is mitigated with a double-pass: every comparison runs twice with A/B order swapped. A preference flip is recorded as a tie.

---

## 9. `[V1 — shipped]` Test Case Stratification

Tags are free-form strings attached to test cases. evalblink groups results by tag and reports quality per group.

**Why it matters:** aggregate scores hide subgroup failures. A 91% overall score can mask 55% on billing edge cases.

**Recommended taxonomy:**

| Tag | Purpose |
|---|---|
| `easy` | Unambiguous inputs any model should handle |
| `medium` | Require nuanced understanding |
| `edge_case` | Boundary conditions, ambiguous phrasing |
| `non_english` | Inputs in non-primary languages |
| `<category>` | Domain label matching your output taxonomy (e.g. `billing`, `order`) |

Aim for roughly 50% easy, 30% medium, 20% edge cases.

**Auto-warning:** any tag scoring below 70% in the best-performing combination is flagged in both the CLI output and the Markdown report:

```
⚠️  Warning  : edge_case quality is below 70%.
```

---

## 10. `[V1 — shipped]` Cost Estimation — `--dry-run`

Estimates the full run cost without making any completion API calls:

```bash
evalblink run benchmarks/classification.yaml --dry-run
```

```
DRY RUN — estimated cost (no API calls)
 Model                         Prompt  Est prompt tok  Est completion tok  Est cost
 anthropic/claude-sonnet-4-6   v1      3842            200                 $0.001230
 openai/gpt-4o-mini            v1      3842            200                 $0.000210

Estimated total cost: $0.001440
Estimate only — completion tokens assume max_tokens; prompt tokens ≈ chars/4.
✓ Within budget: estimate $0.001440 ≤ max_cost_usd $10.000000.
```

- Prompt tokens estimated as `len(rendered_text) / 4`
- Completion tokens use the configured `max_tokens` as a ceiling
- Judge tokens estimated separately per `llm_judge` case
- Exits 1 if the estimate exceeds `max_cost_usd`; exits 0 otherwise

Pricing data is fetched from OpenRouter once per day and cached locally at `.evalblink_cache/models.json`.

---

## 11. `[V1 — shipped]` Concurrency

API calls run concurrently using `ThreadPoolExecutor` from Python's standard library.

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=concurrency) as executor:
    futures = [executor.submit(call_model, ...) for ...]
    results = [f.result() for f in futures]
```

```yaml
concurrency: 5    # max parallel requests (default: 5)
```

**Why `ThreadPoolExecutor` over asyncio:** for a CLI tool making a finite batch of API calls to one host, threading is simpler, easier to debug, and produces identical throughput to async without the complexity.

**Estimated impact on 50 cases × 3 models × 2 prompts:**

| Mode | Estimated duration |
|---|---|
| Sequential | ~8–12 minutes |
| Concurrent (default: 5) | ~45–90 seconds |

---

## 12. `[V1 — shipped]` Retry Logic

Manual retry loop in `openrouter.py`. No external retry library.

```python
RETRYABLE_CODES = {408, 429, 502, 503, 504}
MAX_RETRIES = 4
```

Retries on: HTTP status codes in `RETRYABLE_CODES` and `httpx.TimeoutException`. Each retry waits with exponential backoff. After `MAX_RETRIES` failures, the test case is scored `None` (pipeline error, not a score of 0) and listed under "API Errors" in the report; the run continues.

---

## 13. `[V1 — shipped]` Caching

Every API response is cached locally by SHA256 key.

**Cache key:** `SHA256(model_id + rendered_system_prompt + rendered_prompt + temperature + max_tokens)`

Both `system` and `template` are rendered (variables injected) before hashing — changing a variable value invalidates the cache correctly.

```
.evalblink_cache/
  a3f2b1c4d5e6...json    ← candidate responses
  b7c8d9e0f1a2...json
  models.json            ← OpenRouter model catalog (TTL: 24h)
```

**Cache is always on** — bypass it per-run with `--no-cache`. There is no `cache:` config key.

> Cache + `temperature: 0` = fully reproducible benchmarks. evalblink warns if cache is active with `temperature > 0`.

```bash
evalblink cache stats    # entry count and total size on disk
evalblink cache clear    # wipe all cached responses (--yes to skip confirmation)
```

---

## 14. `[V1 — shipped]` CLI Reference

```bash
# Run a benchmark
evalblink run benchmarks/classification.yaml
evalblink run                               # defaults to benchmarks/exact_match_classification.yaml
evalblink run benchmarks/classification.yaml -v          # verbose: per-case detail
evalblink run benchmarks/classification.yaml --no-cache  # bypass local cache
evalblink run benchmarks/classification.yaml --dry-run   # estimate cost, no API calls

# Validate a config (errors and warnings, no API calls)
evalblink validate benchmarks/classification.yaml

# Compare two finished runs (quality + cost deltas; offline, no API calls)
evalblink compare results/<run_a>.json results/<run_b>.json
evalblink compare results/<run_a>.json results/<run_b>.json --detailed  # per-case transitions

# Regenerate Markdown report from existing JSON
evalblink report results/<run>.json

# List past runs
evalblink history

# Browse OpenRouter model catalog
evalblink models
evalblink models --provider anthropic
evalblink models --free
evalblink models --min-context 100k

# Scaffold a new benchmark YAML interactively
evalblink init

# Cache management
evalblink cache stats
evalblink cache clear --yes
```

---

## 15. `[V1 — shipped]` Results Matrix

### Terminal output

```
              Customer Support Classification — 2026-06-10_14-32_customer-support-classification
 Model                         Prompt  Score   Pass/Scored  Errors  Prompt tok  Completion tok  Cost
 anthropic/claude-sonnet-4-6   v2      96.0%   48/50        0       5120        344             $0.000471
 anthropic/claude-sonnet-4-6   v1      94.0%   47/50        0       4200        312             $0.000445
 openai/gpt-4o-mini            v2      91.0%   45/50        0       5010        321             $0.000071
 openai/gpt-4o-mini            v1      87.0%   43/50        1       4100        298             $0.000063

       QUALITY BY TAG — best combo (anthropic/claude-sonnet-4-6 / v2)
 Tag          Cases  Quality  Errors
 easy         30     99%      0
 medium       15     91%      0
 edge_case    5      60%      0     ⚠️
 billing      12     63%      1     ⚠️

RECOMMENDATION
Best quality : anthropic/claude-sonnet-4-6 / v2 (96.0%)
Best value   : openai/gpt-4o-mini / v2 (91.0% · $0.0001/run)
⚠️  Warning  : edge_case quality is below 70%.
⚠️  Warning  : billing quality is below 70%.
```

**Columns tracked:** Model, Prompt, Score (%), Pass/Scored, Errors (pipeline errors), Prompt tokens, Completion tokens, Cost. **Latency is not tracked.**

### Markdown report structure

Auto-generated at `results/YYYY-MM-DD_HH-MM_<name>.md`:

```markdown
# {benchmark name}
- Run ID: ...
- Timestamp: ...
- Judge model: ...
- Temperature: ...
- Max tokens: ...
- Quality threshold: ...

## Summary
| Model | Prompt | Score | Pass/Scored | Errors | Prompt tok | Completion tok | Cost |
...

## {model} — prompt `{id}`
| Test case | Evaluation | Match | Score | Cost |
...

## Quality by tag — best combo ({model} / {prompt})
| Tag | Cases | Quality | Errors | |
...

## Recommendation
- Best quality: ...
- Best value: ...
- ⚠️ Warning: ...
```

---

## 16. `[V1 — shipped]` Results History & Compare

### Storage

Every run writes two files:

```
results/
  2026-06-10_14-32_customer-support-classification.json   # full record, git-friendly
  2026-06-10_14-32_customer-support-classification.md     # human-readable report
```

JSON includes a top-level `schema_version` field. `evalblink compare` handles mismatched versions gracefully rather than hard-blocking — old records remain usable.

### Compare output

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

`--detailed` adds a per-combo test-case transition table (regressed / improved / new_error / recovered) and a global summary (change counts, worst-regressed cases, per-tag net).

**Latency is intentionally absent from compare output** — the runner does not capture it.

---

## 17. `[V1 — shipped]` OpenRouter API Integration

Called at run start to fetch live model metadata.

**Endpoint:** `GET https://openrouter.ai/api/v1/models`  
**Auth:** `OPENROUTER_API_KEY` environment variable  
**Local cache:** TTL 24h at `.evalblink_cache/models.json`

Data used per model:
- `pricing.prompt` and `pricing.completion` — cost per token for dry-run estimates and post-run cost reporting
- `context_length` — available for future context-overflow warnings

---

## 18. `[V1 — shipped]` CI/CD Integration

evalblink exits `0` (pass) or `1` (fail) based on `quality_threshold`.

```yaml
# .github/workflows/llm-eval.yml
- run: evalblink run benchmarks/classification.yaml
  env:
    OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
```

Best quality score across all combinations ≥ `quality_threshold` → exit 0.  
Below threshold, or `--dry-run` over `max_cost_usd` → exit 1, pipeline blocked.

---

## 19. `[V1 — shipped]` Technical Stack

| Component | Tool | Notes |
|---|---|---|
| CLI | `argparse` (stdlib) | Subcommand dispatcher in `main.py` |
| Terminal output | `rich` | Tables, colors, progress |
| Config | `PyYAML` (`safe_load`) | Standard, readable |
| Template injection | `Jinja2` | Double-brace syntax; supports filters and raw blocks |
| HTTP client | `httpx` (sync) | `httpx.Client` as context manager reuses the TCP connection across all calls to the same host |
| Concurrency | `ThreadPoolExecutor` (stdlib) | Sufficient for batch API calls; no async complexity |
| Retry logic | Manual loop in `openrouter.py` | `RETRYABLE_CODES`, `MAX_RETRIES`, exponential backoff; no external retry library |
| Data validation | `TypedDict` (typing) + `validator.py` | TypedDicts for IDE/mypy only; `validator.validate()` does runtime checking at the CLI boundary |
| LLM routing | OpenRouter | Multi-vendor, single API key |
| Caching | Local JSON files (SHA256 key) | Zero infra, reproducible |
| Result storage | Local JSON + Markdown | Human-readable, git-friendly |

**Why `httpx` over `requests`:** `httpx.Client()` used as a context manager keeps the TCP connection alive across the entire benchmark run — connection reuse that `requests` does not provide. On a 300-call benchmark, this difference is measurable. The async migration path is also explicit: swap `httpx.Client` for `httpx.AsyncClient`.

**Environment variables:**

```bash
OPENROUTER_API_KEY=     # required
```

---

## 20. `[V1 — shipped]` Repo Structure

```
evalblink/
├── evalblink/
│   ├── main.py             # argparse CLI entrypoint + subcommand handlers
│   ├── runner.py           # Benchmark execution (ThreadPoolExecutor)
│   ├── evaluator.py        # exact_match + weighted_match + llm_judge + bias detection
│   ├── reporter.py         # Terminal (Rich) + JSON + Markdown output
│   ├── openrouter.py       # httpx client, model metadata, retry logic, SHA256 cache
│   ├── cache.py            # SHA256 cache read/write/stats/clear
│   ├── compare.py          # Delta between two run JSON records
│   ├── estimate.py         # Offline cost estimation for --dry-run
│   ├── validator.py        # Static config validation (no API calls)
│   ├── analysis.py         # Insights: best combo, tag breakdown, recommendations
│   └── schemas.py          # TypedDicts for IDE/mypy only — no runtime effect
├── benchmarks/
│   ├── exact_match_classification.yaml
│   ├── llm_as_judge.yaml
│   ├── weighted_match_config.yaml
│   └── classification.yaml          # mixed-mode: exact_match + llm_judge
├── docs/
│   └── skills/              # practitioner guides (content planned for V1.1)
├── tests/                   # offline unit tests (fake httpx.Client, no API key needed)
├── results/                 # git-ignored
├── .evalblink_cache/        # git-ignored
├── pyproject.toml
├── README.md
├── CLAUDE.md
├── PRD.md
└── .env.example
```

---

## 21. `[V1 — shipped]` Differentiation vs PromptFoo

| | evalblink | PromptFoo |
|---|---|---|
| Language | Python (`pip install`) | Node.js (`npx`) |
| Vendor neutrality | Independent | Acquired by OpenAI (March 2026) |
| Live model metadata | ✅ OpenRouter API | ❌ |
| Cost estimation incl. judge | ✅ | ❌ |
| Run comparison (delta) | ✅ `evalblink compare` | ❌ |
| Markdown report per run | ✅ | ❌ |
| Self-preference detection | ✅ warning at run start | ❌ |
| Verbosity + style bias guardrails | ✅ baked into judge prompt | ❌ |
| Reference-guided grading | ✅ optional `reference` field | ❌ |
| Per-tag result breakdown | ✅ with ⚠️ auto-warnings | ❌ |
| Weighted match evaluation | ✅ multi-dimension scoring | ❌ |
| Judge calibration vs human labels | V1.1 `evalblink calibrate` | ❌ |
| Pairwise judge mode | V1.1 with order-swap tie detection | ❌ |
| Caching | ✅ | ✅ |
| Concurrency | ✅ `ThreadPoolExecutor` | ✅ |
| CI/CD integration | ✅ | ✅ |
| Red teaming | ❌ Out of scope | ✅ |
| Web UI | V2 | ✅ |

---

## 22. `[V1 — shipped]` Evaluation Methodology — `/docs/skills/`

evalblink ships with four practitioner guides covering the full evaluation workflow. **Content is planned for V1.1** — the directory structure (`docs/skills/`) is in place.

| Guide | What it will cover |
|---|---|
| `01_design_your_test_set.md` | Golden set design, 50/30/20 distribution, contamination risk, tag taxonomy |
| `02_write_a_testable_prompt.md` | Deterministic output formats, variable isolation, common anti-patterns |
| `03_read_your_results.md` | Interpreting the results matrix, parse errors, aggregate vs subgroup scores |
| `04_refine_your_prompt.md` | Using `evalblink compare`, one-variable iteration, when to expand the test set |

---

## 23. `[V1.1 — planned]` Interactive Scaffolding — `evalblink init`

> **Status:** `evalblink init` shipped in V1 as a basic interactive YAML scaffolder. V1.1 will extend it to generate pre-structured test case distributions (50/30/20 easy/medium/edge_case) with inline references to the `/docs/skills/` guides.

---

## 24. `[V2 — planned]` Streamlit Dashboard

Visual interface for running benchmarks and sharing results with non-technical stakeholders.

**Planned screens:**
- **Configure** — Upload YAML or build from form. Model selector with live OpenRouter metadata. Cost preview.
- **Run** — Select prompt × model combinations. Dry-run toggle. Progress bar per combination. Live cost counter.
- **Results** — Interactive heatmap (prompts × models). Drill down per cell → individual test case results. Tag breakdown. Export Markdown or PDF.
- **Compare** — Two-run delta view. Highlight improved/regressed cells.
- **History** — List of all runs: name, date, best score, total cost. Search, filter, delete.
- **Models** — Full OpenRouter model table. Filter by provider, context size, price.

### `[V2 — planned]` Multi-turn Conversation Tests

Multi-turn test cases specify the full conversation history and the expected behaviour at each turn:

```yaml
test_cases:
  - id: "multi_001"
    type: "multi_turn"
    tags: ["context_retention", "edge_case"]
    turns:
      - role: "user"
        content: "I can't find my last order."
      - role: "assistant"
        evaluation: "exact_match"
        expected_output: "order_issue"
      - role: "user"
        content: "Actually, I think I was also charged twice."
      - role: "assistant"
        evaluation: "llm_judge"
        criteria: >
          Should acknowledge both the missing order and the billing issue.
          Must not contradict the previous turn.
```

**Why V2 and not V1:** multi-turn test cases are more complex to author — this complexity lands on the user's YAML and increases the learning curve. V1 targets zero-to-benchmark in under 5 minutes.

---

## 25. `[V3 — vision]`  Assisted Optimisation

> **Vision, not a commitment.** Nothing here ships before V2 is validated.

`evalblink suggest` reads a benchmark result and proposes hypotheses — not a rewritten prompt, but diagnostic observations grounded in the data:

```
evalblink suggest results/run_002.json

DIAGNOSIS — anthropic/claude-sonnet-4-6 / v2
Overall quality: 96%

⚠️  Subgroup failures detected:
    edge_case   60% — model struggles with ambiguous phrasing
    non_english 50% — model defaults to English label names

Hypotheses to test:
  1. Add explicit handling for ambiguous cases in the system prompt
  2. Add a non-English example to the few-shot block
  3. Increase max_tokens — truncated outputs may explain the parse error rate on edge_case

→ Run: evalblink run benchmarks/classification_v3.yaml to validate hypothesis 1
```

**The human stays in the decision loop.** evalblink diagnoses and hypothesises. The human decides what to test next and writes the new prompt. This preserves the core value proposition: understanding *why* one prompt is better, not just optimising a metric.

**Why not a fully autonomous agent:** a fully autonomous agent that generates test cases and optimises against them in the same loop produces overfitting by design — a system that scores well on its own benchmark and fails in production. The human checkpoint between hypothesis and validation is not a limitation; it is the architecture.

---

## 26. Success Metrics

| Metric | V1 Target |
|---|---|
| GitHub stars | 100+ in first month |
| Time to first benchmark | < 5 minutes from `pip install` |
| README clarity | A non-engineer runs the example without help |
| Personal use | Validated on a real production benchmark |

---

*evalblink V7.0 — PRD updated to reflect V1 shipped state. All V1 sections describe the current implementation exactly. Sources: G-Eval (Liu et al., 2023), MT-Bench (Zheng et al., 2024), Judging the Judges (Gu et al., 2025).*
