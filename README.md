# evalblink

**Benchmark LLM prompts × models — quality and cost in one command.**

Vendor-neutral Python CLI to benchmark LLM prompts and models on your own data. No JavaScript, no vendor lock-in.

```bash
pipx install evalblink        # or: uv tool install evalblink
evalblink init                # 3 questions (Enter = defaults) → a starter benchmark that runs as-is
evalblink run benchmarks/my-benchmark.yaml
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
…
⚠️  API error on openai/gpt-4o-mini / v1 / conv_048: OpenRouter request timed out after 4 attempt(s)
…
               Customer Support Classification — 2026-06-10_143205_customer-support-classification
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Model                        ┃ Prompt ┃ Score ┃ Pass/Scored ┃ Errors ┃ Prompt tok ┃ Completion tok ┃      Cost ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ openai/gpt-4o-mini           │ v1     │ 87.8% │       43/49 │      1 │       4018 │            294 │ $0.000784 │
│ openai/gpt-4o-mini           │ v2     │ 90.0% │       45/50 │      0 │       5000 │            300 │ $0.000960 │
│ anthropic/claude-sonnet-4.6  │ v1     │ 94.0% │       47/50 │      0 │       4100 │            300 │ $0.017500 │
│ anthropic/claude-sonnet-4.6  │ v2     │ 96.0% │       48/50 │      0 │       5000 │            300 │ $0.021000 │
│ mistralai/mistral-small-2603 │ v1     │ 85.4% │       41/48 │      2 │       3936 │            288 │ $0.000768 │
│ mistralai/mistral-small-2603 │ v2     │ 88.0% │       44/50 │      0 │       5000 │            300 │ $0.000960 │
└──────────────────────────────┴────────┴───────┴─────────────┴────────┴────────────┴────────────────┴───────────┘
          QUALITY BY TAG — best combo
      (anthropic/claude-sonnet-4.6 / v2)
┏━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━┓
┃ Tag         ┃ Cases ┃ Quality ┃ Errors ┃    ┃
┡━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━┩
│ edge_case   │     5 │     60% │      0 │ ⚠️ │
│ non_english │     4 │     50% │      0 │ ⚠️ │
│ billing     │    12 │    100% │      0 │    │
│ easy        │    30 │    100% │      0 │    │
│ medium      │    15 │    100% │      0 │    │
└─────────────┴───────┴─────────┴────────┴────┘

RECOMMENDATION
Best quality : anthropic/claude-sonnet-4.6 / v2 (96.0%)
Best value   : openai/gpt-4o-mini / v1 (87.8% · $0.0008/run)
3 pipeline error(s) excluded from scoring.
⚠️  Warning  : edge_case quality is below 70%.
⚠️  Warning  : non_english quality is below 70%.
Results saved to results/2026-06-10_143205_customer-support-classification.json
Markdown report saved to results/2026-06-10_143205_customer-support-classification.md

⚠  3 of 300 case(s) hit a pipeline error (API / judge failure) and are excluded from scores.

PASS: best score meets quality_threshold (85).
```

Sample from a 50-case run (3 models × 2 prompts, `quality_threshold: 85`) in which three calls timed out. The lines before the table are trimmed: a `Test case: <id>` progress line per case, a one-line warning per failed call, and per-combo totals. Rows follow the config's model × prompt order; the tag table lists warning rows first, then the rest alphabetically. **Pass/Scored** counts only genuinely scored cases — pipeline errors (a failed API call or an unusable judge reply) are shown in **Errors** and excluded from the score. The `pipeline error(s) excluded from scoring` line and the closing warning give the run total across all combos; the last line is the `quality_threshold` gate (see the CI/CD Integration section below). Names, ids, tags, error messages and model output are printed literally: Rich markup such as `[bold]` and `:emoji:` codes in them are not interpreted.

---

## Quickstart

**1. Install**

```bash
pipx install evalblink        # or: uv tool install evalblink / pip install evalblink
```

Example configs live in [`benchmarks/`](https://github.com/alexandre-darmon/evalblink/tree/main/benchmarks) in this repo; `evalblink init` writes a commented starter you edit.

> **Local development (from the repo root):**
> ```bash
> pip install -e ".[dev]"
> ```

**2. Set your API key**

```bash
export OPENROUTER_API_KEY=your_key_here    # or put it in a .env file
```

The `.env` file is read from the directory you run `evalblink` in (or the nearest parent that has one), never from the installed package's directory. `evalblink run` exits 1 with a clear message, before any API call, if `OPENROUTER_API_KEY` is unset, empty or whitespace-only. A key with a non-ASCII character (a pasted `…` or curly quote) can't be sent in an HTTP header: every call fails with a clear `api_error` (and `models` / `--dry-run` / `init` say so in one line). Save `.env` as UTF-8 (a BOM is fine); a file in another encoding is ignored with a warning. `evalblink validate` and `evalblink run --dry-run` need no API key and make no completion calls (`--dry-run` only fetches the public model catalog).

**3. Write a benchmark**

```yaml
# my_benchmark.yaml
name: "Customer Support Classification"
quality_threshold: 85        # CI/CD gate: exits 1 if best combo score < 85%
max_cost_usd: 10.00          # --dry-run exits 1 if estimated cost exceeds this

models:
  - "openai/gpt-4o-mini"
  - "anthropic/claude-sonnet-4.6"
  - "mistralai/mistral-small-2603"

inference:
  temperature: 0
  max_tokens: 200

evaluation:
  judge_model: "google/gemini-2.5-flash"  # vendor shared by no candidate (avoids self-preference)
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
evalblink validate my_benchmark.yaml          # optional: static check, no API calls
evalblink run my_benchmark.yaml --dry-run     # optional: cost estimate, no completion calls
evalblink run my_benchmark.yaml
```

Want a $0 first run? The configs in [`benchmarks/`](https://github.com/alexandre-darmon/evalblink/tree/main/benchmarks) use only free (`:free`) OpenRouter models — free models are rate-limited, so expect the occasional retry. They live in the repository, not in the installed package:

```bash
git clone https://github.com/alexandre-darmon/evalblink && cd evalblink
evalblink run benchmarks/exact_match_classification.yaml
```

---

## Configuration Reference

| Field | Required | Description |
|---|---|---|
| `name` | ✅ | Benchmark display name |
| `models` | ✅ | List of OpenRouter model IDs |
| `quality_threshold` | — | Top-level CI/CD gate: `evalblink run` exits 1 if the best combo score falls below this percentage (0–100) |
| `max_cost_usd` | — | `--dry-run` exits 1 if the estimated cost exceeds this amount |
| `gate` | — | `{model, prompt}`, either optional: the `quality_threshold` gate checks the combos it names instead of the best combo of the run |
| `concurrency` | — | Max parallel API requests — a whole number from 1 to 100 (default: `5`) |
| `inference.temperature` | — | Sampling temperature (default: `0` for reproducibility) |
| `inference.max_tokens` | — | Max output tokens per call — a positive whole number such as `100`; `100.0` is rejected (default: `100`) |
| `evaluation.judge_model` | ✅ if `llm_judge` used | OpenRouter model ID used as the judge — must be a string whenever present |
| `evaluation.judge_threshold` | — | Per-case pass threshold for `llm_judge` — normalized score 0–1 (default: `0.70`) |
| `evaluation.judge_max_tokens` | — | Output budget of each judge call — a positive whole number (default: `4096`; reasoning judges need thousands) |
| `evaluation.quality_threshold` | ✅ if `weighted_match` used | Per-case pass threshold for `weighted_match` — weighted score 0–1 |
| `evaluation.variables` | ✅ if `weighted_match` used | Weighted dimension definitions (`name`, `weight`, optional `tolerance`) |
| `prompts[].id` | ✅ | Unique identifier used in reports |
| `prompts[].template` | ✅ | Jinja2 template — use `{{ variable }}` (double braces); rendered in a sandbox |
| `prompts[].system` | — | Optional system message (also a sandboxed Jinja2 template) |
| `variables` | — | Global key/value pairs injected into all templates |
| `test_cases` | ✅ | The list of test cases below, or `{file: cases.csv, evaluation?}` to load them from a CSV, TSV or JSONL file (see [Test cases from a file](#test-cases-from-a-csv-or-jsonl-file)) |
| `test_cases[].id` | ✅ | Unique identifier |
| `test_cases[].evaluation` | ✅ | `exact_match` \| `llm_judge` \| `weighted_match` |
| `test_cases[].expected_output` | ✅ if `exact_match` or `weighted_match` | Ground truth (string for `exact_match`, JSON array for `weighted_match`) |
| `test_cases[].criteria` | ✅ if `llm_judge` | Evaluation rubric shown to the judge |
| `test_cases[].reference` | — | Optional gold answer injected into the judge prompt |
| `test_cases[].variables` | — | Per-case variables that override or extend global `variables` |
| `test_cases[].tags` | — | Free-form strings for per-category result breakdown |

> **Threshold naming:** `quality_threshold` at the top level is the CI/CD gate (0–100 %). `evaluation.quality_threshold` is the per-case pass threshold for `weighted_match` (0–1). `evaluation.judge_threshold` is the per-case pass threshold for `llm_judge` (0–1, default 0.70). These are three distinct concepts.

> **Validation:** `evalblink validate` — and `evalblink run`, which validates first — reports config mistakes as errors (exit 1) instead of failing mid-run:
>
> - **Structure:** an empty or non-mapping file; missing `name` / `models` / `prompts` / `test_cases`; a non-string `name`, model ID, `prompts[].id` or `test_cases[].id`; duplicate entries in `models`; duplicate prompt or test-case ids; an `inference` / `evaluation` block that isn't a mapping (a bare `inference:` with no value counts as absent); an unknown `evaluation` type; a self-referencing YAML alias (`&a [*a]`), which could not be written to the results JSON.
> - **Unknown keys:** a top-level key the code doesn't read is an error, with a suggestion when one is close (`unknown key 'qualty_threshold' — did you mean 'quality_threshold'?`): a misspelled `quality_threshold` would otherwise switch the CI gate off without a word. An unknown key inside `inference`, `evaluation`, a prompt or a test case is a warning naming where it is. Keys starting with `_` or `x-` are yours (YAML anchors, notes) and never flagged.
> - **Numbers:** a top-level `quality_threshold` that isn't a number in 0–100; a `concurrency` that isn't a whole number from 1 to 100; a negative `max_cost_usd`; a non-numeric `inference.temperature`; an `inference.max_tokens` that isn't a positive whole number (`50.5`, `100.0` and `true` are rejected); and, anywhere a number is checked (`concurrency` included), a non-finite value (`.inf`, `.nan`) or one too large for a float (e.g. a 400-digit integer). An integer of more than 4300 digits, or YAML nested thousands of levels deep, makes the whole file `Invalid YAML`.
> - **Per mode:** an `exact_match` case without `expected_output`, or with one that isn't a string (YAML reads unquoted `yes`, `true` or `3` as a boolean or number — quote it); `llm_judge` cases without `criteria` or without `evaluation.judge_model`, or a `judge_threshold` outside 0–1, or a `judge_max_tokens` that isn't a positive whole number; an `evaluation.judge_model` that isn't a string (checked whenever it is present, even with no `llm_judge` cases); for `weighted_match`, a missing `evaluation.quality_threshold` (0–1), an `evaluation.variables` list missing any of `use_case` / `percent` / `order` or with non-numeric weights / tolerances, and an `expected_output` that is empty, not a list, or has items without a scalar `use_case` (not a list, map or set), a numeric `percent`, and an `order`.
> - **Templates:** Jinja2 syntax errors in `template` / `system`, and template variables not provided by `variables`. Once everything else is valid, `validate` also trial-renders every prompt (`template` and `system`) against every test case — offline, with the runner's own render function and sandbox. Any exception (an undefined nested attribute, a `ZeroDivisionError`, a sandbox `SecurityError`, a `TypeError` from a filter…) is an error naming the prompt id and test case id, so a template that would fail at run time fails `validate` before any API call. (A template whose output is random — e.g. the `random` filter — can still fail on a later render; that case is then recorded as an `internal_error`, below.)
>
> Single-brace `{variable}` syntax, a `max_tokens` below 200 with `llm_judge`, and a long `expected_output` are warnings (exit 0).
>
> **Sandboxed templates:** `template` and `system` render in a Jinja2 `SandboxedEnvironment` — filters such as `{{ items | join(", ") }}` work, but unsafe attribute access is blocked: a single unsafe attribute such as `{{ x.__class__ }}` renders as empty, while calling or chaining through one (`{{ x.__class__() }}`, `{{ x.__class__.__mro__ }}`) raises `jinja2.exceptions.SecurityError` — which `evalblink validate` reports as an error.

---

## CLI Reference

```bash
# Run a benchmark
evalblink run benchmarks/classification.yaml
evalblink run benchmarks/classification.yaml -v          # verbose: per-case detail
evalblink run benchmarks/classification.yaml --no-cache  # bypass the response cache (candidate and judge calls)
evalblink run benchmarks/classification.yaml --dry-run   # estimate cost, no completion calls, no API key needed
evalblink run my_benchmark.yaml -m openai/gpt-4o-mini -p v2   # one model, one prompt (repeatable)
evalblink run my_benchmark.yaml --tag edge_case --limit 5     # a slice of the test cases
python -m evalblink run …                                 # same CLI when `evalblink` isn't on PATH

# Validate a config file (errors and warnings, no API calls)
evalblink validate benchmarks/classification.yaml

# Compare two runs (quality + cost deltas; offline)
evalblink compare                                        # the two latest runs of a benchmark
evalblink compare results/<run_a>.json results/<run_b>.json
evalblink compare <run_a> <run_b> --detailed             # run IDs or unique prefixes; per-case transitions
evalblink compare baseline.json --fail-on-regression 2   # CI: the latest run vs a kept baseline; exit 1 on a drop > 2 points

# Regenerate a Markdown report from an existing JSON result
evalblink report                                         # the latest run
evalblink report results/<run>.json

# List past runs
evalblink history

# Browse available models from OpenRouter
evalblink models
evalblink models --provider anthropic
evalblink models --free
evalblink models --free --text     # text in, text-only out (drops audio/image generators)
evalblink models --min-context 100k
evalblink models --no-cache          # refetch instead of using the 24h catalog cache

# Write a commented starter benchmark (3 questions; --yes for none)
evalblink init
evalblink init my-eval.yaml --yes

# Cache management
evalblink cache stats
evalblink cache clear --yes                              # without --yes it only prints the entry count
```

`evalblink run` and `evalblink validate` with no path use the only YAML in `benchmarks/` (what `init` writes) and print `Using <path>` to stderr. With several YAMLs there, they use `benchmarks/exact_match_classification.yaml` when it exists (as in this repo) and otherwise exit with the list to pick from; with none, they suggest `evalblink init`.

- `models --min-context` takes values like `100k`, `8K` or `131072`; anything else is an argparse usage error (exit 2). A catalog price that is negative or non-numeric (OpenRouter quotes `-1` for its `openrouter/*` routers) is unknown: `models` shows it as `?` and `--free` leaves it out.
- `init` asks three questions: a name, the evaluation mode (`exact_match` or `llm_judge`) and the models to compare, comma-separated. Press Enter at each for the defaults. It then writes a complete, commented starter: an example prompt and example test cases, `temperature: 0`, `max_tokens: 1024` (headroom for free reasoning models), `concurrency: 2` (free models rate-limit bursts) and, for `llm_judge`, the free judge `nvidia/nemotron-3-ultra-550b-a55b:free`. The file validates and runs as-is; replace the example test cases with your own in your editor. The default model is the free `poolside/laguna-xs-2.1:free` (or the first free catalog model if OpenRouter has retired it). While you choose, it lists up to five free text models from the catalog (text in, text-only out; audio and image generators are left out) and a small paid option without free-tier rate limits, `google/gemma-4-31b-it`, with its price. That option is never preselected. Models absent from the catalog get a warning, and an empty name or model list is asked again. `init` carries on without suggestions when the catalog can't be fetched. The output file is the optional path argument, or `benchmarks/<slug>.yaml`, relative to the directory you run `init` in (never the install directory); `init` prints its absolute path when done (`benchmarks/benchmark.yaml` for a name with no letters or digits); an existing file is only overwritten after you confirm. `init --yes` asks nothing and writes the `exact_match` starter with the default model, refusing to overwrite an existing file. Before writing, `init` runs `validate` on the result and exits 1 without writing if there are errors.

---

## Test cases from a CSV or JSONL file

Past a handful of cases, keep them in a spreadsheet instead of YAML. Point `test_cases` at the file (the path is relative to the YAML):

```yaml
test_cases:
  file: cases/support_tickets.csv   # .csv, .tsv or .jsonl
  evaluation: exact_match           # optional default for rows without an `evaluation`
```

```
id,conversation,expected_output,tags
conv_001,I cannot find my last order.,order_issue,order;easy
conv_004,"I was charged twice, and the parcel never arrived.",billing,bill;order;edge_case
```

- **Columns** `id`, `evaluation`, `expected_output`, `criteria`, `reference`, `tags` and `human_score` are the test-case fields. Every other column is a template variable (`conversation` above fills `{{ conversation }}`); global `variables:` still apply.
- **CSV / TSV:** the first row names the columns. `tags` are separated by `;`. An empty cell leaves a field unset, while an empty variable stays an empty string. A `weighted_match` `expected_output` is a JSON array in its cell. Files saved by Excel as "CSV UTF-8" (with a byte-order mark) work.
- **JSONL:** one JSON object per line, with the same keys as a YAML test case; extra keys are variables too.
- **No `id` column?** Each row is named after its line (`line_2`, `line_3`…).
- Mistakes stop before any API call with the file and line: `Invalid test cases in bench.yaml: cases.csv:4: 3 cells but 2 columns — quote values that contain the delimiter`. A column whose name is close to a field (`expeced_output`) gets a warning, since it would otherwise silently become a variable.

The file is expanded into the usual list before validation, so `validate`, `--dry-run`, the run filters and every report treat its rows exactly like YAML test cases. [`benchmarks/csv_classification.yaml`](https://github.com/alexandre-darmon/evalblink/blob/main/benchmarks/csv_classification.yaml) is a complete example.

---

## Evaluation Modes

### Exact match

For classification tasks where the model must return a specific label. Prompt the model explicitly to return no JSON and no extra text.

```yaml
# Excerpt of benchmarks/exact_match_classification.yaml (first test case shown)
name: "Customer Support Classification (exact match)"

inference:
  temperature: 0
  max_tokens: 1024  # headroom in case the model spends tokens on reasoning before the label

prompts:
  - id: "v1"
    template: >
      Classify this conversation.
      Return only a valid label from the list. No JSON. No explanation. No punctuation.
      Choose from: {{ labels }}.
      Conversation: {{ conversation }}

models:
  - "nvidia/nemotron-3.5-lightning:free"

variables:
  labels: "order_issue, billing, product_question, other"

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
# Excerpt of benchmarks/llm_as_judge.yaml (prompt v1 and the first test case shown)
name: "Customer Support Quality"
inference:
  temperature: 0
  max_tokens: 1024
evaluation:
  judge_model: "nvidia/nemotron-3-ultra-550b-a55b:free"  # different vendor from the candidate
  judge_threshold: 0.70

prompts:
  - id: "v1"
    system: >
      You are Paul, a senior customer support agent at an e-commerce company.
      Your goal is to resolve customer issues efficiently while making them
      feel heard.

      Rules:
      - Always acknowledge the customer's emotion before proposing a solution.
      - Ask one clarifying question if the issue is ambiguous — never assume.
      - If multiple issues are raised, address each one explicitly.
      - Never use generic phrases like "I understand your frustration" alone —
        show you understood the specific situation.
      - Propose a concrete next step in every response.
      - Keep your response under 100 words.

    template: "Customer message: {{ customer_message }}"

models:
  - "poolside/laguna-xs-2.1:free"

test_cases:
  # Edge case — emotional frustration before resolution
  - id: "conv_001"
    variables:
      customer_message: >
        This is the third time my order arrived damaged.
        I am done with this company.
    evaluation: "llm_judge"
    criteria: >
      The assistant must acknowledge the customer's frustration explicitly
      before proposing any resolution. A response that jumps straight to
      a solution without emotional acknowledgment scores below 3.
      The tone must be empathetic throughout — not defensive, not robotic.
      A concrete next step must be proposed (refund, replacement, escalation).
    tags: ["frustration", "edge_case"]
```

The judge reasons step by step before committing to a score (chain-of-thought-first, G-Eval / MT-Bench style). The judge returns a score of 1–5, normalized to 0–1. Cases scoring below `judge_threshold` count as failures. The judge gets its own output budget, `evaluation.judge_max_tokens` (default 4096), independent of the candidate's `inference.max_tokens`: judges that reason at length before answering can use thousands of tokens (the free `nvidia/nemotron-3.5-lightning:free` takes 2,000–6,000 per verdict, which is why the examples use the more concise `nemotron-3-ultra`) and return nothing when the budget runs out — raise it if you see `judge_api_error … finish_reason: length`. A failed judge call prints a one-line `Judge API error on …` warning, and a reply with no usable score (often JSON cut off by the budget) prints `Judge error on …`; either way the reason is kept in the case's `error` field. The judge's reasoning is stored in the JSON record (`results[].test_cases[].reasoning`) — not in the Markdown report — so you can check *why* a response scored poorly. If the judge call fails or its reply has no usable score, the case is a pipeline error (`status` `judge_api_error` / `judge_error`, score `null`): it counts under **Errors** and is excluded from the score, never as a 0.

**Built-in bias mitigations:**
- Judge prompt instructs against rewarding length (verbosity bias: 10–20% magnitude)
- Judge prompt instructs against rewarding confident tone over accuracy (style bias)
- If `evaluation.judge_model` shares a vendor with any candidate model, evalblink warns you at run start (self-preference bias: 10–25% magnitude)

### Weighted match

For structured outputs where the model must return a JSON array of `{"use_case", "percent", "order"}` objects. Three dimensions are scored independently and combined by weight.

```yaml
# Excerpt of benchmarks/weighted_match_config.yaml (first test case shown)
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

prompts:
  - id: "v1"
    system: >
      Analyse cette conversation et identifie les cas d'usage présents.
      Retourne uniquement un JSON valide, sans explication, sans markdown.
      Format: [{"use_case": "<label>", "percent": <float>, "order": <int>}]
      Les pourcentages doivent sommer à 1.0.
      Use cases disponibles: {{ use_cases }}.
    template: >
       Conversation: {{ conversation }}

models:
  - "poolside/laguna-xs-2.1:free"

variables:
  use_cases: "Translate content, Analyze content, Summarize content, Generate new ideas, Write content, Generate simple code, Other"

test_cases:
  - id: "conv_001"
    variables:
      conversation: |
        User: Fais moi une synthèse de ce document.
        User: Reformule le en 10 bullet points.
        User: Traduis le en anglais.
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
- `use_case`: F1 score — `2 × precision × recall / (precision + recall)` — on matched labels
- `percent`: fraction of expected items within `tolerance` of the expected value
- `order`: fraction of expected items with an exact order match

---

## Cost Estimation

Before running a benchmark, estimate the cost without making any completion calls (only the public OpenRouter model catalog is fetched, cached for 24h):

```bash
evalblink run my_benchmark.yaml --dry-run    # the Quickstart config above
```

```
                          DRY RUN — estimated cost (no API calls)
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Model                        ┃ Prompt ┃ Est prompt tok ┃ Est completion tok ┃  Est cost ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ openai/gpt-4o-mini           │ v1     │            191 │                800 │ $0.010830 │
│ openai/gpt-4o-mini           │ v2     │            246 │                800 │ $0.010843 │
│ anthropic/claude-sonnet-4.6  │ v1     │            191 │                800 │ $0.022895 │
│ anthropic/claude-sonnet-4.6  │ v2     │            246 │                800 │ $0.023064 │
│ mistralai/mistral-small-2603 │ v1     │            191 │                800 │ $0.010830 │
│ mistralai/mistral-small-2603 │ v2     │            246 │                800 │ $0.010843 │
└──────────────────────────────┴────────┴────────────────┴────────────────────┴───────────┘
Judge cost (llm_judge cases): $0.061942
Estimated total cost: $0.089304
Estimate only — completion tokens assume max_tokens; prompt tokens ≈ chars/4.
✓ Within budget: estimate $0.089304 ≤ max_cost_usd $10.000000.
```

Per combo, **Est completion tok** is `max_tokens` (or the model's output limit, if lower) × the number of test cases (a ceiling — real outputs are usually shorter), and **Est cost** includes that combo's share of judge calls. The `Judge cost` line appears only when the estimated judge cost is non-zero — so not with a free (`:free`) judge model. A model that isn't in the catalog at all is almost always a typo: `--dry-run` prints `✗ Unknown model 'openai/gpt-4o-mni': not in the OpenRouter catalog — did you mean 'openai/gpt-4o-mini'?` and exits 1. A model listed with a negative or non-numeric price (such as the `openrouter/*` routers) has unknown pricing: it is excluded from the estimate with a warning (`No pricing for '<model>' — excluded from the estimate.`). Each budget is capped at the model's own output limit from the catalog (`top_provider.max_completion_tokens`); when `inference.max_tokens` or `evaluation.judge_max_tokens` is above it, `--dry-run` warns: `inference.max_tokens (5000) is above <model>'s output limit (2048 tokens) — lower it to 2048 or less.`

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

Transient API failures — HTTP 408 / 429 / 502 / 503 / 504, whether on the status line or in the JSON error body, plus local timeouts and connection errors — are retried with exponential backoff (1 s, 2 s, 4 s), up to 4 attempts. A rate limit (429) waits longer — 5 s, 10 s, 20 s plus up to 1 s of random jitter, so parallel workers don't retry in lockstep — or the server's `Retry-After` seconds when it sends them (capped at 60 s); when the retries run out, the error says it was rate-limited and suggests retrying later, lowering `concurrency` or adding a paid model. Other client-side errors (an undecodable body, a redirect loop) fail the call at once, without a retry. Any other failure of a call surfaces as a clear error (with the HTTP status when known): an error status, a non-JSON body such as an HTML 502 page, a reply with no usable `choices` / `message`, null or empty content, or missing usage data.

**A failed call never aborts the run.** A candidate call that still fails after its retries — or hits a non-transient error such as an unknown model or a bad key — is recorded with `status: "api_error"`, `match_score: null`, `match_result: false`, an empty `response`, the message in `error`, and zero tokens/cost; evalblink prints a one-line warning and carries on. A failed judge call is recorded the same way as a judge error (see the LLM-as-judge section above). Either kind counts in the **Errors** column and is excluded from the score; the JSON and Markdown results are still written, and the end of the run prints a prominent warning with the number of pipeline errors excluded from scores. If *nothing* could be scored, the run exits 1 (see the CI/CD Integration section below).

No single test case can crash a run: anything unexpected while rendering or scoring one case (for example a template that uses `random`) records that case as `status: "internal_error"`, the same way as an API error, and the run carries on.

**Ctrl-C** stops a run at once: queued calls are dropped, calls in flight are abandoned (no more retries), evalblink prints `Interrupted — no results written.` and exits 130. Before the first call, `run` checks that it can write to `results/`; if it can't (a read-only directory, or a file named `results`), it exits 1 without spending anything.

---

## Caching

Every successful API response (candidate and judge) is cached under `.evalblink_cache/` by the SHA256 of the request payload: model, rendered system prompt, rendered prompt, temperature, and max_tokens. When you iterate on prompt v2, all v1 responses are served from cache at zero cost. Failed calls are never cached. A judge reply that arrives but can't be parsed (`judge_error`) *is* cached: `--no-cache` retries it for that run only (it neither reads nor writes the cache), so run `evalblink cache clear --yes` to drop the bad entry for good. `--no-cache` bypasses the cache for every call of the run, candidate and judge. A cache entry that can't be read or doesn't have the expected shape counts as a miss: the call is made again and the entry is overwritten. If the cache can't be written (for example the cache directory is read-only), evalblink prints `Could not write the response cache (…); continuing.` once and the run goes on uncached; a model catalog that can't be cached also gives a warning and the command goes on.

```bash
evalblink cache stats            # cached responses, their size, and whether the model catalog is cached
evalblink cache clear            # prints the response count only — deletes nothing
evalblink cache clear --yes      # delete every cached response
evalblink cache clear --yes --all  # also the cached model catalog (refetched on next use)
```

The model catalog (`.evalblink_cache/models.json`, used by `--dry-run`, `models` and `init`) is counted apart from responses, and `clear` keeps it unless you pass `--all`.

> Cache + `temperature: 0` = fully reproducible benchmarks.

---

## Results & History

Every run produces two files:

```
results/
  2026-06-10_143205_customer-support-classification.json   # full data, git-friendly
  2026-06-10_143205_customer-support-classification.md     # human-readable report
```

A second run of the same benchmark within the same second gets a `-2` (`-3`, …) suffix instead of overwriting the first.

Filenames are `YYYY-MM-DD_HHMMSS_<slug>`, where the slug is the benchmark `name` lowercased with each run of non-alphanumeric characters turned into `-`, capped at 60 characters, and `benchmark` when nothing is left (e.g. a name with no ASCII letters or digits) — so any valid name gives a writable path. In the Markdown report's per-case tables, a pipeline-error case shows its status in the Score column (`API error`, `judge API error`, `judge error` or `internal error`) instead of a number. Lone UTF-16 surrogates (for example a `"\ud800"` escape in the YAML or in model output) are replaced with `?` in requests, terminal output and the Markdown report. The JSON includes schema versioning so `evalblink compare` handles old files gracefully — it never hard-blocks on a version mismatch. Regenerate the Markdown report at any time:

```bash
evalblink report results/2026-06-10_143205_customer-support-classification.json
```

```bash
evalblink compare results/2026-06-08_101500_customer-support-classification.json \
                  results/2026-06-10_143205_customer-support-classification.json
```

```
 DELTA: 2026-06-08_101500_customer-support-classification → 2026-06-10_143205_customer-support-classification
 Model                          Prompt  Quality Δ    Cost Δ       Notes
 anthropic/claude-sonnet-4.6    v1      +4.0% ↑      +0.0002 ↑
 openai/gpt-4o-mini             v1      +6.0% ↑      stable
 anthropic/claude-sonnet-4.6    v2      stable       stable

Verdict: Run B improves quality on 2/3 shared combo(s).
```

Changed combos are listed first. Scores leave out pipeline-error cases, so when a shared combo's error count differs between the runs, its quality Δ compares different sets of cases. Such a row is yellow, its Notes column reads `pipeline errors 2 → 0`, and the verdict leaves it out of the improved count and says how many were left out. Use `--detailed` to see which cases changed. `--detailed` adds per-combo test-case transitions (regressed / improved / new error / recovered) and a global summary.

`report` and `compare` exit with a one-line message when a results file is missing, unreadable, not JSON, or not shaped like a run record (it needs string `benchmark` and `timestamp` fields and a well-formed `results` list). `history` lists every readable run and prints `⚠  Skipping <reason>` for each file it can't use. `report` and `history` recompute the insights from `results`, so the `pipeline error(s) excluded from scoring` line and the history **Errors** column show the run total, matching the warning at the end of the run.

---

## CI/CD Integration

`evalblink run` exits `0` (pass) or `1` (fail):

```yaml
# .github/workflows/llm-eval.yml
- run: evalblink run my_benchmark.yaml
  env:
    OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
```

- **Exit 1 before any API call** when the config fails validation or `OPENROUTER_API_KEY` is unset, empty or whitespace-only. `--dry-run` needs no API key.
- **Exit 1 when nothing could be scored** — every case of the run is a pipeline error (`api_error`, `judge_api_error`, `judge_error`, `internal_error`). The results are still written.
- **Otherwise the `quality_threshold` gate decides:** best combo ≥ `quality_threshold` → exit 0; below it → exit 1. With no `quality_threshold`, exit 0. The JSON record's `insights.passed` matches the exit code: it is `false` whenever nothing could be scored, even with `quality_threshold: 0` or none. The gate is fail-safe: it counts every case that could not be scored as a failure (passed / total cases), so a run where most calls failed can't pass on the few that were scored — while one timeout in a strong run still passes. The **Score** column itself still covers scored cases only (pipeline errors are counted under **Errors**), and when any occurred, the run ends with a warning giving their count.
- **`gate:`** narrows the gate to one combo: `gate: {model: openai/gpt-4o-mini, prompt: v2}` makes the PASS/FAIL line read `PASS: openai/gpt-4o-mini / v2 score meets quality_threshold (80).` Either key can be left out. A `--model` / `--prompt` filter that removes the gated combo drops the gate for that run, with a note.
- `--dry-run` exits 1 when the estimate exceeds `max_cost_usd` or a model isn't in the OpenRouter catalog (with a `did you mean` suggestion), 0 otherwise.
- **Regression check between runs:** `evalblink compare <baseline> --fail-on-regression POINTS` compares a kept run record with the latest run of the same benchmark (and the same `run` filters), and exits 1 when a combo's pass rate drops by more than `POINTS` (`0` = any drop beyond ±0.05). The rate counts cases that couldn't be scored as failures, like the gate, so a run that breaks with new API errors fails it. Combos that ran different test cases (a `--tag` / `--limit` slice against a full run) aren't compared, and when no combo can be compared at all — the test set changed, or a model was renamed — it exits 1 and asks you to refresh the baseline. With no other run to compare with, it exits 0 with a note.

```yaml
# Keep a baseline record in the repo (copy one from results/ after a good run).
- run: evalblink run my_benchmark.yaml
  env:
    OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
- run: evalblink compare benchmarks/baseline.json --fail-on-regression 2
```

---

## Evaluation Methodology — `/docs/skills/` (planned, V1.1)

Four practitioner guides covering the full evaluation workflow are planned for V1.1. No code required.

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
- `evalblink init` — a commented starter benchmark from three questions
- `evalblink models` — live OpenRouter model catalog with filtering

**V1.1**
- `/docs/skills/` methodology guides
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
pipx install evalblink        # or: uv tool install evalblink / pip install evalblink
evalblink --version
```

Requires Python 3.10+. One environment variable: `OPENROUTER_API_KEY`.

---

## License

MIT
