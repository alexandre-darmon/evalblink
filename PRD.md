# PRD — evalblink
## Multi-LLM Evaluation & Benchmark Kit

**Author:** Alexandre Darmon  
**Version:** 6.0  
**Status:** Draft  
**Last updated:** June 2026

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

- Run a structured benchmark across N prompts × N models × N test cases in one command
- Evaluate responses via exact match (JSON-enforced output) or LLM-as-judge
- Measure quality, cost, and latency per combination — including judge costs
- Cache results locally to avoid redundant API calls during prompt iteration
- Run API calls concurrently using `ThreadPoolExecutor`
- Retry failed API calls with exponential backoff
- Generate a versioned markdown report per run
- Store results history locally and compare runs over time
- Fetch live model metadata from OpenRouter API

---

## 4. Out of Scope (V1)

- Web UI / dashboard (→ V2)
- Red teaming or adversarial testing
- Real-time guardrails or production monitoring
- Authentication or multi-user support
- Cloud storage or remote result sync
- Fine-tuned model support

---

## 5. Benchmark Configuration

Every benchmark is defined in a single YAML file.

```yaml
name: "Customer Support - Conversation Classification"
description: "Compare 2 prompt versions across 3 models on 50 labelled conversations"
judge_model: "anthropic/claude-sonnet-4-6"
max_cost_usd: 10.00
quality_threshold: 85         # CI/CD: exit 1 if best score < 85%
cache: true                   # Skip API calls for already-cached prompt+model combos

inference:
  temperature: 0              # Required for reproducibility
  max_tokens: 50              # Keeps cost low for classification tasks

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
  labels: "order_issue, authentication_issue, billing, product_question, other"

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
    reference: "billing"   # optional — grounds the judge in a known-good answer
    tags: ["billing", "medium"]

  - id: "conv_003"
    variables:
      conversation: "I was charged twice last month but the app shows one order."
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

### Configuration reference

| Field | Required | Description |
|---|---|---|
| `name` | ✅ | Benchmark display name |
| `description` | — | Optional context |
| `judge_model` | ✅ if llm_judge used | OpenRouter model ID for evaluation |
| `max_cost_usd` | — | Hard stop if total estimated cost (models + judge) exceeds limit |
| `quality_threshold` | — | Minimum quality % for CI/CD pass |
| `cache` | — | Enable local response caching (default: true) |
| `inference.temperature` | — | Sampling temperature (default: 0 for reproducibility) |
| `inference.max_tokens` | — | Max output tokens per call (default: 100) |
| `prompts[].id` | ✅ | Unique identifier used in reports |
| `prompts[].template` | ✅ | Prompt template using `{variable}` placeholders |
| `prompts[].system` | — | Optional system message |
| `models` | ✅ | List of OpenRouter model IDs |
| `variables` | — | Global variables injected into all templates |
| `test_cases[].evaluation` | ✅ | `exact_match` or `llm_judge` |
| `test_cases[].expected_output` | ✅ if exact_match | Ground truth label |
| `test_cases[].criteria` | ✅ if llm_judge | Evaluation rubric for the judge |
| `test_cases[].reference` | — | Optional gold reference answer injected into the judge prompt to ground scoring in truth rather than plausibility (V1) |
| `test_cases[].tags` | — | Optional list of free-form strings for per-category result breakdown (e.g. `["billing", "edge_case"]`) |
| `test_cases[].human_score` | — | Optional integer 1–5 from a human reviewer, used by `evalblink calibrate` (V1.1) |

> **Self-preference warning:** if `judge_model` shares a vendor prefix with any model in `models` (e.g. `anthropic/claude-*` judging `anthropic/claude-*`), evalblink emits a warning at run start. Self-preference bias runs 10–25% in magnitude — always use a judge from a different model family than the candidates being evaluated.

---

## 6. Template Injection

Variables are injected into prompt templates using Jinja2.

```python
from jinja2 import Template
rendered_prompt = Template(template).render(**{**global_variables, **test_case_variables})
```

Templates use double-brace syntax: `{{ variable }}`. Single braces `{variable}` are treated as literal text and silently produce wrong prompts — `evalblink validate` warns on this pattern.

**Why Jinja2:** filter expressions (`{{ label | upper }}`, `{{ items | join(", ") }}`) are common in LLM prompt authoring and require a real template engine. `str.format_map()` cannot support them, and its error messages on missing keys are harder to diagnose than Jinja2's `UndefinedError`.

**Literal braces in prompt text:** wrap in Jinja2 raw blocks — `{% raw %}{...}{% endraw %}` — or avoid bare braces in few-shot JSON examples by using backtick code blocks in the prompt instead.

---

## 7. Evaluation Modes

### 7.1 Exact Match (JSON-enforced)

Used for classification and structured output tasks.

The prompt template instructs the model to return a specific JSON structure:

```
Return only valid JSON: {"label": "<label>"}
```

evalblink parses the response:

```python
import json
parsed = json.loads(response)
predicted = parsed.get("label", "").strip().lower()
expected = expected_output.strip().lower()
score = 1 if predicted == expected else 0
```

**Why JSON over text parsing:** Extracting "the last meaningful word" from a free-text LLM response is fragile and breaks on model updates, language changes, or unexpected formatting. Enforcing a JSON contract is deterministic and requires zero regex.

**Fallback:** If the model returns malformed JSON, evalblink scores the test case 0, logs the raw response, and records it as a parse error. No regex fallback is attempted — doing so would undermine the JSON contract that makes exact match reliable in the first place.

```json
{
  "test_case_id": "ticket_001",
  "status": "parse_error",
  "raw_response": "The answer is billing.",
  "score": 0,
  "note": "Model returned malformed JSON. Expected {\"label\": \"...\"}."
}
```

Parse errors are grouped under a dedicated "Parse Errors" section in the Markdown report, separate from correct/incorrect results.

### 7.2 LLM-as-Judge

Used for open-ended tasks (summarization, generation, explanation) where no single correct output exists.

The judge model receives:

```
You are an expert evaluator. Your goal is to assess quality, not style.

Original task: {prompt}
Evaluation criteria: {criteria}
Model response: {model_output}
{reference_block}
Think step by step about whether the response meets the criteria.

Important:
- Judge on criteria satisfaction only — do not reward length. A concise correct answer
  scores higher than a verbose partially-correct one.
- Do not reward confident tone. An accurate hedged answer scores higher than a
  confidently wrong one.
- The criteria above are for evaluation only and were not shown to the model being judged.

Then return ONLY valid JSON:
{"reasoning": "<your analysis>", "score": <integer 1-5>}

Scoring scale:
1 — Completely wrong or off-topic
2 — Partially relevant but missing key elements
3 — Acceptable but imprecise
4 — Good, minor issues
5 — Perfect, meets all criteria
```

When `reference` is provided in the test case, `{reference_block}` is injected as:

```
Reference answer: {reference}
Use this as a ground truth anchor when assessing correctness.
```

When no reference is provided, `{reference_block}` is omitted entirely.

**Why reasoning first:** The judge writes its analysis before committing to a number. This chain-of-thought approach — established by G-Eval (Liu et al., 2023) and MT-Bench — produces more calibrated scores because the model reasons its way to a position rather than anchoring on a number and justifying it retroactively. The `reasoning` field also surfaces in the report, giving users actionable signal on why a response scored poorly.

**Response structure:**

```json
{
  "reasoning": "The response correctly identifies a billing issue and returns a single label without explanation, meeting both criteria.",
  "score": 4
}
```

Score normalized to 0–100%: `(score - 1) / 4 * 100`.

**Judge parse error:** If the judge returns malformed JSON, the test case is scored `null`, logged under "Judge Errors" in the report, and excluded from quality calculations. Score 0 is reserved for valid judge responses rating the output as completely wrong — it must not be conflated with a pipeline failure.

**Judge cost:** Each llm_judge test case triggers one additional API call to the judge model. This cost is included in the dry-run estimate and the final cost report. On a 50 test case benchmark with `llm_judge`, expect judge costs to equal or exceed model call costs when using a powerful judge model.

---

### 7.3 Judge Reliability & Bias Mitigation

LLM judges exhibit documented, quantified biases. evalblink addresses them systematically, with V1 mitigations built in and V1.1 mitigations designed from the start.

#### Known biases and their magnitude

| Bias | Magnitude | Description |
|---|---|---|
| Self-preference | 10–25% | Judge favours responses from its own model family |
| Verbosity | 10–20% | Longer responses rated higher regardless of quality |
| Style | variable | Confident tone rewarded even when content is wrong |
| Position | 5–15% | Judge favours the response appearing first in pairwise prompts |

*Sources: MT-Bench (Zheng et al., 2024), Judging the Judges (Gu et al., 2025), LLM Evaluation Guide (2025).*

#### `[V1]` Self-preference detection

At config validation time, `schemas.py` extracts the vendor prefix from `judge_model` and each entry in `models`. If any candidate model shares the same vendor as the judge, evalblink emits a warning before the run starts:

```
⚠️  Self-preference risk detected.
    judge_model: anthropic/claude-sonnet-4-6
    candidate:   anthropic/claude-haiku-3-5

    LLM judges show 10–25% preference for outputs from their own model family.
    Recommendation: use a judge from a different vendor (e.g. openai/gpt-4o).
```

The run is not blocked — sometimes no better judge is available — but the warning is always shown and recorded in the result JSON under `"warnings"`.

#### `[V1]` Verbosity and style bias — judge prompt guardrails

The judge prompt explicitly instructs the model to:
- Judge on criteria satisfaction only, not on length
- Prefer accurate hedged answers over confidently wrong ones

These two lines are embedded in the system-level instruction block and are not user-configurable, to ensure they are always active.

#### `[V1]` Rubric isolation — design principle

The `criteria` field is passed exclusively to the judge model. It is never injected into the candidate's `template`. This prevents rubric-hacking — where a model learns to match rubric keywords superficially rather than meeting the actual quality bar. The YAML schema enforces this separation: `criteria` is a test-case-level field, not a prompt-level field.

#### `[V1]` Reference-guided grading

When `test_cases[].reference` is provided, the gold reference answer is injected into the judge prompt. This grounds evaluation in truth rather than plausibility, reducing the risk of hallucination blind spots where a fluent but incorrect answer scores well. Reference is optional — if absent, the judge evaluates against criteria alone.

#### `[V1.1]` Judge calibration — `evalblink calibrate`

> **Designed in V1, built in V1.1.** The `human_score` field is available in the schema from V1 so teams can start collecting labels immediately. The `calibrate` command ships in V1.1.

LLM judge scores are only meaningful if they correlate with human judgment. The recommended threshold is a correlation coefficient ≥ 0.7 or Cohen's Kappa ≥ 0.6 (substantial agreement). Below 0.4, the rubric is likely ambiguous and needs revision.

**Schema addition (V1 — collect now, analyse in V1.1):**

```yaml
test_cases:
  - id: "conv_002"
    evaluation: "llm_judge"
    criteria: "Should identify a billing inquiry."
    human_score: 4    # optional — integer 1-5, collected offline by a human reviewer
```

**CLI command (V1.1):**

```bash
evalblink calibrate results/run_001.json
```

**Output:**

```
JUDGE CALIBRATION REPORT
Judge model: anthropic/claude-sonnet-4-6
Human-labelled cases: 24 / 50

Pearson correlation : 0.81  ✅ (threshold: 0.70)
Cohen's Kappa       : 0.74  ✅ (threshold: 0.60)

Verdict: Judge is well-calibrated. Scores are reliable.

Cases with largest disagreement (judge vs human):
  conv_018  judge=5  human=2  Δ=3  → review criteria
  conv_031  judge=2  human=5  Δ=3  → review criteria
```

Re-validate calibration after any judge model update — provider updates can shift judge behaviour without notice.

#### `[V1.1]` Pairwise judge mode

> **Designed in V1, built in V1.1.**

Pointwise scoring (V1's default) asks the judge to score each response independently on a 1–5 scale. Pairwise evaluation — asking "which of these two responses better satisfies the criteria?" — produces higher inter-rater agreement because it simplifies the cognitive task from absolute rating to relative comparison.

This maps naturally onto evalblink's core use case: comparing prompt v1 vs v2 on the same model.

**Pairwise is immune to verbosity bias by design:** if both responses are verbose, neither benefits. The judge compares *relative* quality.

**Position bias mitigation (double-pass):** pairwise evaluation introduces position bias (5–15% magnitude) — the judge systematically favours the response appearing first. evalblink mitigates this with a double-pass: every comparison is run twice with A/B order swapped. If the judge's preference flips, the result is recorded as a tie. This catches approximately 90% of position-bias sensitivity errors.

```yaml
# V1.1 YAML addition
judge_mode: "pairwise"    # default: "pointwise"
```

**Cost implication:** pairwise mode doubles judge API calls (two passes per comparison). This is included in dry-run cost estimates.

---

### 7.4 `[V1]` Test Case Stratification

Tags are free-form strings attached to test cases. They have no semantic meaning to evalblink — the user defines their own taxonomy. evalblink groups results by tag and reports quality per group.

**Why it matters:** aggregate scores hide subgroup failures. A 91% overall score can mask 55% on billing edge cases. The literature is consistent on this: high overall scores masking poor performance on critical subsets is the single most common cause of production failures in LLM applications (LLM Evaluation Guide, 2025).

**Recommended tag taxonomy for classification tasks:**

| Tag | Purpose |
|---|---|
| `easy` | Unambiguous inputs any model should handle |
| `medium` | Require nuanced understanding |
| `edge_case` | Boundary conditions, ambiguous phrasing, unusual formats |
| `non_english` | Inputs in languages other than the primary training language |
| `<category>` | Domain-specific label matching your output taxonomy (e.g. `billing`, `order`) |

Aim for roughly 50% easy, 30% medium, 20% edge cases. The research-recommended golden set size is 50–200 cases total, version-controlled alongside your prompt templates.

**Implementation:** one optional `List[str]` field on `TestCase` in `schemas.py`. The reporter does a `groupby(tag)` over results — no new API calls, no new dependencies. Tags with fewer than 3 cases are reported but flagged as statistically thin.

**⚠️ threshold trigger:** if any tag scores below 70% in the best-performing combination, the recommendation block in the CLI and the report both flag it explicitly. This surfaces before the aggregate threshold check so subgroup failures are never buried.

---

API calls run concurrently using `ThreadPoolExecutor` from Python's standard library.

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=concurrency) as executor:
    futures = [executor.submit(call_model, prompt, model) for ...]
    results = [f.result() for f in futures]
```

**Why ThreadPoolExecutor over asyncio:** For a CLI tool making network API calls, `ThreadPoolExecutor` is simpler, easier to debug, and produces identical throughput. `asyncio` + `httpx` async is the right architecture for a server handling concurrent users — not for a script executing a finite batch of API calls. Threading is sufficient and removes a significant layer of complexity.

**Concurrency config:**

```yaml
concurrency: 5    # max parallel requests (default: 5)
```

**Estimated impact on a 50 test case × 3 models × 2 prompts benchmark:**

| Mode | Estimated duration |
|---|---|
| Sequential | ~8–12 minutes |
| Concurrent (default: 5) | ~45–90 seconds |

---

## 9. Retry Logic

All API calls are wrapped with exponential backoff using `tenacity`.

```python
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

@retry(
    wait=wait_exponential(min=1, max=30),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((TimeoutError, RateLimitError))
)
def call_model(prompt: str, model: str) -> str:
    ...
```

**Behavior:**
- Attempt 1: immediate
- Attempt 2: wait 1–2s
- Attempt 3: wait 2–4s
- After 3 failures: log the error, score the test case as `None`, continue benchmark

Failed calls are listed in the report under "API Errors" and excluded from quality calculations.

---

## 10. Caching

evalblink caches every API response locally to avoid redundant calls during prompt iteration.

**Cache key:** `SHA256(model_id + rendered_system_prompt + rendered_prompt + temperature + max_tokens)`

The system prompt is included in the key because changing `prompts[].system` alone must invalidate the cache. Both `system` and `template` are rendered (variables injected) before hashing — an unrendered template with different variables could produce the same raw string but a different effective prompt.

```python
import hashlib, json

def make_cache_key(model_id, rendered_system, rendered_prompt, temperature, max_tokens):
    payload = json.dumps({
        "model": model_id,
        "system": rendered_system or "",
        "prompt": rendered_prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()
```

`json.dumps` with `sort_keys=True` ensures dict key ordering never produces a different hash for identical inputs.

```
.evalblink_cache/
  a3f2b1c4d5e6...json
  b7c8d9e0f1a2...json
```

**Important — temperature and reproducibility:**

> Caching is only meaningful when `temperature: 0`. At temperature > 0, two identical calls return different outputs. Caching non-deterministic responses defeats the purpose of evaluation.

evalblink enforces this: if `cache: true` and `temperature > 0`, a warning is displayed at run start:

```
⚠️  Cache enabled with temperature=0.7. Cached responses may not reflect
    current model behavior. Set temperature: 0 for reproducible benchmarks.
```

**Cache commands:**

```bash
evalblink cache clear       # delete all cached responses
evalblink cache stats       # show entry count, size, hit rate from last run
```

**Impact:** On a prompt iteration run (changing only `v1` → `v2`), prompt `v1` responses are served from cache at zero cost. Only `v2` triggers new API calls.

---

## 11. CLI Reference

```bash
# Run a single benchmark
evalblink run benchmarks/classification.yaml

# Run all benchmarks in a folder
evalblink run benchmarks/

# Estimate cost without calling APIs (includes judge cost)
evalblink run benchmarks/classification.yaml --dry-run

# Force fresh API calls, bypass cache
evalblink run benchmarks/classification.yaml --no-cache

# Compare two historical runs (delta view)
evalblink compare results/run_001.json results/run_002.json

# List past runs
evalblink history

# List available models from OpenRouter (context window + pricing)
evalblink available-models
evalblink available-models --min-context 100k --provider mistral

# Scaffold a new benchmark file interactively
evalblink init

# Cache management
evalblink cache clear
evalblink cache stats

# [V1.1] Validate judge calibration against human-labelled cases
evalblink calibrate results/run_001.json
```

---

## 12. Results Matrix

### CLI output

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
⚠️  Warning   : edge_case quality is 60% — review these test cases before shipping.
```

### Markdown report — three tables per metric

```markdown
## Results Matrix

### Quality (%)
              | gpt-4o-mini | claude-sonnet | mistral-small-3
prompt_v1     | 87%         | 94%           | 82%
prompt_v2     | 91% ↑       | 96% ↑         | 88% ↑

### Cost ($/run — models + judge)
              | gpt-4o-mini | claude-sonnet | mistral-small-3
prompt_v1     | $0.06       | $0.45         | $0.04
prompt_v2     | $0.07       | $0.46         | $0.04

### Latency (avg per call)
              | gpt-4o-mini | claude-sonnet | mistral-small-3
prompt_v1     | 1.1s        | 2.3s          | 0.9s
prompt_v2     | 1.2s        | 2.1s ↓        | 0.8s ↓
```

> Compact single-cell format (`quality · cost · latency`) for CLI. Separate tables per metric for Markdown. Interactive heatmap for V2 Streamlit.

---

## 13. Markdown Report Structure

Auto-generated at `/results/YYYY-MM-DD_HH-MM-SS_<name>.md`

```markdown
# Benchmark Report: {name}
Run ID: {uuid} | {timestamp}
Cache: {hit_count}/{total} hits | Total cost: ${total}

## Warnings
[self-preference alerts, cache+temperature conflicts, context window overflows]

## Results Matrix
[quality / cost / latency — 3 tables]

## Quality by Tag
[per-tag breakdown for the best-performing combination — only shown if tags are present]

tag           | cases | quality
--------------|-------|--------
easy          | 30    | 99%
medium        | 15    | 91%
edge_case     | 5     | 60%  ⚠️
billing       | 12    | 63%  ⚠️
non_english   | 4     | 50%  ⚠️

## Cost Breakdown
[models cost + judge cost, per prompt version and per model]

## Failed Test Cases
[exact_match mismatches + JSON parse errors + API errors + judge errors]

## Judge Reasoning
[per llm_judge case: criteria · reference (if set) · reasoning · score]

## Model Specs
[from OpenRouter API: context window, pricing]

## Context Window Warnings
[test cases that exceed a model's context window]

## Recommendations
[auto-generated: best quality, best value, biggest improvement, tag-level warnings]
```

---

## 14. Results History & Compare

### Storage

```
/results
  2026-06-10_14-32_classification_a3f2b1.json
  2026-06-10_14-32_classification_a3f2b1.md
```

JSON stores: full config snapshot, all rendered prompts, responses, scores, tokens, costs, latencies, model specs, cache stats.

Every result JSON includes a top-level `schema_version` field written at run time:

```json
{
  "schema_version": "1.0",
  "run_id": "a3f2b1",
  ...
}
```

**Why schema versioning matters:** when new fields are added in future releases, `evalblink compare` can detect mismatches and handle them gracefully rather than silently producing wrong deltas.

**`evalblink compare` version handling:**

| Scenario | Behaviour |
|---|---|
| Both files same version | Proceed normally |
| Different versions | Print warning listing which fields may be absent, then compare on fields common to both versions |
| File predates versioning (no field) | Treat as `"0.9"`, warn the user, proceed with best effort |

The comparison never hard-blocks on version mismatch — that would make historical result files unusable. The warning gives the user full visibility without destroying the workflow.

Markdown is generated once at run time. Re-generate from JSON on demand:

```bash
evalblink report results/run_001.json    # regenerate .md from existing .json
```

### Compare

```bash
evalblink compare results/run_001.json results/run_002.json
```

```
DELTA: run_001 → run_002
Prompt: v1 → v2

Model             | Quality    | Cost       | Latency
------------------|------------|------------|----------
gpt-4o-mini       | +4% ↑      | +$0.01     | +0.1s
claude-sonnet     | +2% ↑      | stable     | stable
mistral-small-3   | +6% ↑      | stable     | -0.1s ↓

Verdict: prompt_v2 improves quality across all models.
         mistral-small-3: +6% quality at no additional cost.
```

---

## 15. OpenRouter API Integration

Called at run start to fetch live model metadata.

**Endpoint:** `GET https://openrouter.ai/api/v1/models`  
**Auth:** `OPENROUTER_API_KEY` environment variable  
**Local cache:** TTL 24h (stored in `.evalblink_cache/models.json`)

Data extracted per model:
- `context_length` — triggers warnings if test cases exceed it
- `pricing.prompt` — cost per input token
- `pricing.completion` — cost per output token

Cost estimation accounts for: model calls + judge calls (llm_judge cases).

---

## 16. CI/CD Integration

evalblink exits `0` (pass) or `1` (fail) based on `quality_threshold`.

```yaml
# .github/workflows/llm-eval.yml
- run: evalblink run benchmarks/classification.yaml
  env:
    OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
```

Best quality score across all combinations ≥ threshold → exit 0.  
Below threshold → exit 1, pipeline blocked.

---

## 17. Technical Stack

| Component | Tool | Rationale |
|---|---|---|
| CLI | Typer | Type-safe, auto-generates help |
| Terminal output | Rich | Tables, progress bars, colors |
| Config | PyYAML (`safe_load`) | Standard, readable |
| Template injection | `str.format_map()` | Native Python, no dependency |
| API calls | `httpx` (sync) | HTTP/2 connection reuse across calls to the same host; near-identical API to `requests`; trivial migration to `httpx.AsyncClient` for V2 async |
| Concurrency | `ThreadPoolExecutor` | Standard library, sufficient for batch API calls |
| Retry logic | `tenacity` | Exponential backoff, configurable |
| Data validation | `TypedDict` (typing) + `validator.py` (runtime) | TypedDicts document shapes for IDE/mypy; `validator.validate()` catches bad YAML at the CLI boundary with explicit error messages. No Pydantic. |
| LLM routing | OpenRouter | Multi-vendor, single API key |
| Caching | Local JSON (`SHA256` key) | Zero infra, reproducible |
| Result storage | Local JSON + Markdown | Human-readable, git-friendly |

**Why `httpx` over `requests`:** evalblink makes all API calls to a single host (`openrouter.ai`). `httpx.Client()` used as a context manager keeps the TCP connection alive across the entire benchmark run — connection reuse that `requests` does not provide (each call opens a new connection). On a 300-call benchmark, this difference is measurable. The API is nearly identical to `requests`, so the learning curve is the same. The async migration path is explicit: swap `httpx.Client` for `httpx.AsyncClient` and add `await` — no request logic rewrite required.

```python
import httpx

with httpx.Client(timeout=30.0) as client:
    response = client.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
    )
    response.raise_for_status()
    return response.json()
```

**Environment variables:**

```bash
OPENROUTER_API_KEY=     # required
```

---

## 18. Repo Structure

```
evalblink/
├── evalblink/
│   ├── main.py             # argparse CLI entrypoint + subcommand handlers
│   ├── runner.py           # Benchmark execution (ThreadPoolExecutor)
│   ├── evaluator.py        # exact_match + weighted_match + llm_judge
│   ├── reporter.py         # Markdown + terminal output (Rich)
│   ├── openrouter.py       # httpx client + model metadata + retry logic
│   ├── cache.py            # SHA256 cache management
│   ├── compare.py          # Delta between two run JSON files
│   ├── estimate.py         # Offline cost estimation for --dry-run
│   ├── validator.py        # Static config validation (no API calls)
│   └── schemas.py          # TypedDicts for IDE/mypy only — no runtime effect
├── benchmarks/
│   └── example_classification.yaml
├── docs/
│   └── skills/
│       ├── 01_design_your_test_set.md
│       ├── 02_write_a_testable_prompt.md
│       ├── 03_read_your_results.md
│       └── 04_refine_your_prompt.md
├── results/                # Git-ignored
├── .evalblink_cache/       # Git-ignored
├── PRODUCT.md              # PRD — public, version-controlled
├── pyproject.toml
├── README.md
└── .env.example
```

---

## 19. Differentiation vs PromptFoo

| | evalblink | PromptFoo |
|---|---|---|
| Language | Python (pip install) | Node.js (npx) |
| Vendor neutrality | Independent | Acquired by OpenAI (March 2026) |
| Live model metadata | ✅ OpenRouter API | ❌ |
| Cost estimation incl. judge | ✅ | ❌ |
| Run comparison (delta) | ✅ `evalblink compare` | ❌ |
| Markdown report per run | ✅ | ❌ |
| JSON-enforced output | ✅ | ❌ |
| Self-preference detection | ✅ warning at run start | ❌ |
| Verbosity + style bias guardrails | ✅ baked into judge prompt | ❌ |
| Reference-guided grading | ✅ optional `reference` field | ❌ |
| Judge calibration vs human labels | ✅ V1.1 `evalblink calibrate` | ❌ |
| Pairwise judge mode | ✅ V1.1 with order-swap tie detection | ❌ |
| Per-tag result breakdown | ✅ subgroup quality + ⚠️ auto-warnings | ❌ |
| Caching | ✅ | ✅ |
| Concurrency | ✅ ThreadPoolExecutor | ✅ |
| CI/CD integration | ✅ | ✅ |
| Red teaming | ❌ Out of scope | ✅ |
| Web UI | V2 Streamlit | ✅ |

---

## 20. Eval Methodology — `/docs/skills/`

evalblink ships with four markdown guides covering the full evaluation workflow. No code required — these are practitioner-facing documents that teach the methodology alongside the tool.

### Why skills matter

Most eval tools ship documentation about their API. evalblink ships documentation about *how to evaluate*. This distinction matters: a team that runs evalblink without understanding what makes a good test set, a testable prompt, or a meaningful rubric will get numbers that are meaningless at best and misleading at worst. The skills close that gap.

### `[V1]` Four core skills

**`01_design_your_test_set.md` — Design your test set**

Covers: what makes a golden set credible, the 50/30/20 easy/medium/edge_case distribution, minimum viable set size (50–200 cases), tag taxonomy design, contamination risk and why BYOD is the only safe evaluation approach, how to source real production inputs.

**`02_write_a_testable_prompt.md` — Write a testable prompt**

Covers: what makes a prompt evaluable (deterministic output format, unambiguous task definition, isolated variables), the `str.format_map()` escaping rules, system prompt design, how to separate what you're testing from what you're controlling for, common prompt anti-patterns that break exact match evaluation.

**`03_read_your_results.md` — Read your results**

Covers: how to interpret the results matrix (quality vs cost vs latency tradeoffs), reading the tag stratification table, what parse errors signal about your prompt, how to distinguish a model failure from a prompt failure, when a 96% aggregate score is not good enough.

**`04_refine_your_prompt.md` — Refine your prompt**

Covers: using `evalblink compare` to validate a hypothesis, one-variable-at-a-time iteration discipline, how to use judge reasoning output to diagnose failures, when to expand your test set vs when to change your prompt, how to know when you're done iterating.

### `[V1.1]` Interactive scaffolding — `evalblink init`

`evalblink init` generates a pre-structured YAML benchmark file interactively. It asks the user for their task type, label set, and number of test cases — then scaffolds the file with the correct tag distribution and placeholder test cases. The skills are referenced inline.

```bash
evalblink init
> Task type: classification
> Labels: order_issue, billing, product_question, other
> Number of test cases: 20
> ✅ Generated benchmarks/my_benchmark.yaml (6 easy · 6 medium · 4 edge_case · 4 non_english)
```

---

## 21. V2 — Streamlit Dashboard

### Overview

Visual interface for running benchmarks and sharing results with non-technical stakeholders.

### Screens

**Configure** — Upload YAML or build from form. Model selector with live OpenRouter metadata. Prompt editor. Test case table with CSV import. Cost preview including judge cost.

**Run** — Select prompt × model combinations. Dry-run toggle. Progress bar per combination. Live cost counter. Cancel button.

**Results** — Interactive heatmap (prompts × models). Toggle: Quality / Cost / Latency. Drill down per cell → individual test case results. Tag breakdown table. Export Markdown or PDF.

**Compare** — Two-run delta view. Highlight cells with improvement > 5%. LLM-generated narrative summary.

**History** — List of all runs: name, date, models, best score, total cost. Search, filter, delete.

**Models** — Full OpenRouter model table. Filter by provider, context size, price.

### `[V2]` Multi-turn Conversation Tests

Single-turn evaluation (one input → one output) covers most classification and generation tasks. But many LLM applications are conversational — the model must retain context, stay consistent, and handle topic switches across multiple turns.

Multi-turn test cases specify the full conversation history and the expected behaviour at each turn:

```yaml
test_cases:
  - id: "multi_001"
    type: "multi_turn"           # V2 addition
    tags: ["context_retention", "edge_case"]
    turns:
      - role: "user"
        content: "I can't find my last order."
      - role: "assistant"
        expected_contains: "order_issue"   # evaluated at this turn
      - role: "user"
        content: "Actually, I think I was also charged twice."
      - role: "assistant"
        evaluation: "llm_judge"
        criteria: >
          Should acknowledge both the missing order and the billing issue.
          Must not contradict the previous turn.
```

**What multi-turn evaluation catches:**
- **Context retention** — does the model reference earlier turns correctly?
- **Consistency** — does the model contradict itself across turns?
- **Topic switching** — does the model handle abrupt subject changes gracefully?
- **Clarification handling** — does the model respond appropriately to follow-up questions?

**Why V2 and not V1:** multi-turn test cases are more complex to author than single-turn cases — this complexity lands on the user's YAML and increases the learning curve. V1 targets zero-to-benchmark in under 5 minutes. Multi-turn authoring is a deliberate V2 feature once the core workflow is validated.

---

## 22. V3 — Assisted Optimisation

> **Vision, not a commitment.** This section documents the long-term direction. Nothing here ships before V2 is validated.

### The workflow as skills — Phase 1 (V1, shipped)

The `/docs/skills/` folder makes the eval workflow explicit: design test set → write prompt → run benchmark → read results → refine. This is a methodology, not just a tool.

### Interactive scaffolding — Phase 2 (V1.1)

`evalblink init` turns the skills into an interactive CLI wizard. The human makes every decision; evalblink structures the output.

### Assisted optimisation — Phase 3 (V3)

`evalblink suggest` reads a benchmark result and proposes *hypotheses* — not a rewritten prompt, but diagnostic observations grounded in the results:

`evalblink generate` — natural language → draft YAML, with the same self-preference warning logic reused, and an explicit "human review required" banner on output

Expand the existing `evalblink suggest` mockup to show three distinct suggestion types (test case / model / prompt hypothesis) with the accept/reject framing made explicit in the CLI output

A short subsection codifying the risk table above as a design principle — same spirit as your "why not a fully autonomous agent" paragraph, just applied to suggestion-type by suggestion-type


```
evalblink suggest results/run_002.json

DIAGNOSIS — claude-sonnet / prompt_v2
Overall quality: 96%

⚠️  Subgroup failures detected:
    edge_case   60% — model struggles with ambiguous phrasing
    non_english 50% — model defaults to English label names

Hypotheses to test:
  1. Add explicit handling for ambiguous cases in the system prompt
  2. Add a non-English example to the few-shot block
  3. Increase max_tokens — truncated outputs may explain the 14% parse error rate on edge_case

→ Run: evalblink run benchmarks/classification_v3.yaml to validate hypothesis 1
```

**The human stays in the decision loop.** evalblink diagnoses and hypothesises. The human decides what to test next and writes the new prompt. This preserves the core value proposition: understanding *why* one prompt is better, not just optimising a metric.

### Why not a fully autonomous agent

A fully autonomous agent that generates test cases, optimises prompts, and evaluates its own outputs without human validation creates the exact problem evalblink is designed to detect: optimisation on the evaluation set itself. If the agent generates test cases and optimises against them in the same loop, it produces overfitting by design — a system that scores well on its own benchmark and fails in production. The human checkpoint between hypothesis and validation is not a limitation; it is the architecture.

---

## 23. Success Metrics

| Metric | V1 Target |
|---|---|
| GitHub stars | 100+ in first month |
| Time to first benchmark | < 5 minutes from `pip install` |
| README clarity | A non-engineer runs the example without help |
| Personal use | Validated on a real production benchmark |

---

*evalblink V6.0 — PRD reflects fifth CTO review incorporating: (1) httpx (sync) replacing requests — HTTP/2 connection reuse across all calls to openrouter.ai, near-identical API, explicit async migration path for V2; (2) /docs/skills/ methodology layer — four practitioner guides covering the full eval workflow (design test set, write testable prompt, read results, refine prompt), shipped as markdown in V1; (3) evalblink init interactive scaffolding in V1.1 — CLI wizard generating pre-structured YAML with correct tag distribution; (4) V3 assisted optimisation roadmap — evalblink suggest as a hypothesis engine, human-in-the-loop by design, with explicit architectural rationale for why a fully autonomous agent creates the overfitting problem evalblink is built to prevent. Sources: G-Eval (Liu et al., 2023), MT-Bench (Zheng et al., 2024), Judging the Judges (Gu et al., 2025), LLM Evaluation Guide (Commey, 2026).*