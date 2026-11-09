# evalblink reference

Everything the [README](https://github.com/alexandre-darmon/evalblink#readme) leaves out: every config key, what `validate` checks, each evaluation mode in detail, local models, how latency and the statistics are computed, caching, results files and CI. The README is the place to start.

## Contents

- [Configuration reference](#configuration-reference)
- [Test cases from a CSV or JSONL file](#test-cases-from-a-csv-or-jsonl-file)
- [Evaluation modes](#evaluation-modes)
- [Local and self-hosted models](#local-and-self-hosted-models)
- [Latency](#latency)
- [Statistics: intervals, noise and repeats](#statistics-intervals-noise-and-repeats)
- [Cost estimation](#cost-estimation)
- [Test case stratification](#test-case-stratification)
- [Concurrency and retries](#concurrency-and-retries)
- [Caching](#caching)
- [Results and history](#results-and-history)
- [CI/CD integration](#cicd-integration)
- [CLI details](#cli-details)

## Configuration reference

| Field | Required | Description |
|---|---|---|
| `name` | ✅ | Benchmark display name |
| `models` | ✅ | List of model IDs: OpenRouter ones (`openai/gpt-4o-mini`), or `provider:model` for a local or self-hosted server (`ollama:llama3.2`). See [Local and self-hosted models](#local-and-self-hosted-models) |
| `providers` | — | Where local / self-hosted servers listen: `{name: {base_url, api_key_env?, concurrency?, timeout?}}`, to move a built-in one or add your own |
| `quality_threshold` | — | Top-level CI/CD gate: `evalblink run` exits 1 if the best combo score falls below this percentage (0–100) |
| `max_cost_usd` | — | `--dry-run` exits 1 if the estimated cost exceeds this amount |
| `gate` | — | `{model, prompt}`, either optional: the `quality_threshold` gate checks the combos it names instead of the best combo of the run |
| `concurrency` | — | Max parallel API requests — a whole number from 1 to 100 (default: `5`) |
| `repeats` | — | Runs of every model × prompt × test case — a whole number from 1 to 100 (default: `1`). See [Statistics](#statistics-intervals-noise-and-repeats) |
| `inference.temperature` | — | Sampling temperature (default: `0` for reproducibility) |
| `inference.max_tokens` | — | Max output tokens per call — a positive whole number such as `100`; `100.0` is rejected (default: `100`) |
| `evaluation.judge_model` | ✅ if `llm_judge` used | Model ID used as the judge (OpenRouter or `provider:model`) — must be a string whenever present |
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
| `test_cases[].evaluation` | ✅ | `exact_match` \| `contains` \| `not_contains` \| `regex` \| `json_match` \| `json_schema` \| `python` \| `llm_judge` \| `weighted_match` |
| `test_cases[].expected_output` | ✅ except for `llm_judge` and `python` | The label (`exact_match`), the text that must (`contains`) or must not (`not_contains`) appear — a string or a list of strings — the pattern (`regex`), the JSON values to find (`json_match`), a JSON Schema (`json_schema`), or a JSON array (`weighted_match`). Optional for `python`, which receives it |
| `test_cases[].scorer` | ✅ if `python` | `"file.py:function"`, the file relative to the YAML |
| `test_cases[].criteria` | ✅ if `llm_judge` | Evaluation rubric shown to the judge |
| `test_cases[].reference` | — | Optional gold answer injected into the judge prompt |
| `test_cases[].variables` | — | Per-case variables that override or extend global `variables` |
| `test_cases[].tags` | — | Free-form strings for per-category result breakdown |

> **Threshold naming:** `quality_threshold` at the top level is the CI/CD gate (0–100 %). `evaluation.quality_threshold` is the per-case pass threshold for `weighted_match` (0–1). `evaluation.judge_threshold` is the per-case pass threshold for `llm_judge` (0–1, default 0.70). These are three distinct concepts.

> **Validation:** `evalblink validate` — and `evalblink run`, which validates first — reports config mistakes as errors (exit 1) instead of failing mid-run:
>
> - **Structure:** an empty or non-mapping file; missing `name` / `models` / `prompts` / `test_cases`; a non-string `name`, model ID, `prompts[].id` or `test_cases[].id`; duplicate entries in `models`; duplicate prompt or test-case ids; an `inference` / `evaluation` block that isn't a mapping (a bare `inference:` with no value counts as absent); an unknown `evaluation` type; a `repeats` that isn't a whole number from 1 to 100; a self-referencing YAML alias (`&a [*a]`), which could not be written to the results JSON.
> - **Unknown keys:** a top-level key the code doesn't read is an error, with a suggestion when one is close (`unknown key 'qualty_threshold' — did you mean 'quality_threshold'?`): a misspelled `quality_threshold` would otherwise switch the CI gate off without a word. An unknown key inside `inference`, `evaluation`, a prompt or a test case is a warning naming where it is. Keys starting with `_` or `x-` are yours (YAML anchors, notes) and never flagged.
> - **Numbers:** a top-level `quality_threshold` that isn't a number in 0–100; a `concurrency` that isn't a whole number from 1 to 100; a negative `max_cost_usd`; a non-numeric `inference.temperature`; an `inference.max_tokens` that isn't a positive whole number (`50.5`, `100.0` and `true` are rejected); and, anywhere a number is checked (`concurrency` included), a non-finite value (`.inf`, `.nan`) or one too large for a float (e.g. a 400-digit integer). An integer of more than 4300 digits, or YAML nested thousands of levels deep, makes the whole file `Invalid YAML`.
> - **Per mode:** an `exact_match`, `contains`, `not_contains` or `regex` case without `expected_output`, or with one that isn't a string (YAML reads unquoted `yes`, `true` or `3` as a boolean or number — quote it; `contains` / `not_contains` also take a non-empty list of non-empty strings); an empty `contains` / `not_contains` text, which would match every reply; a `regex` that doesn't compile; a `json_match` `expected_output` that isn't a non-empty mapping or list; a `json_schema` `expected_output` that isn't a valid JSON Schema; a `python` case without `scorer`, or whose scorer file is missing, fails to import or has no such function (`validate` imports the file, so its module-level code runs); `llm_judge` cases without `criteria` or without `evaluation.judge_model`, or a `judge_threshold` outside 0–1, or a `judge_max_tokens` that isn't a positive whole number; an `evaluation.judge_model` that isn't a string (checked whenever it is present, even with no `llm_judge` cases); for `weighted_match`, a missing `evaluation.quality_threshold` (0–1), an `evaluation.variables` list missing any of `use_case` / `percent` / `order` or with non-numeric weights / tolerances, and an `expected_output` that is empty, not a list, or has items without a scalar `use_case` (not a list, map or set), a numeric `percent`, and an `order`.
> - **Providers:** a model ID (or `judge_model`) whose `provider:` prefix is neither built in nor under `providers:` (`unknown provider 'olama' — did you mean 'ollama:'?`); a `providers` block that isn't a mapping, a provider name that isn't lowercase letters, digits, `-` or `_`, the reserved name `openrouter`, a new provider without `base_url`, a `base_url` that isn't an `http(s)://` URL, an `api_key_env` that isn't a variable name, a `concurrency` that isn't a whole number from 1 to 100, and a `timeout` that isn't a number of seconds above 0. An unknown key inside a provider is a warning.
> - **Templates:** Jinja2 syntax errors in `template` / `system`, and template variables not provided by `variables`. Once everything else is valid, `validate` also trial-renders every prompt (`template` and `system`) against every test case — offline, with the runner's own render function and sandbox. Any exception (an undefined nested attribute, a `ZeroDivisionError`, a sandbox `SecurityError`, a `TypeError` from a filter…) is an error naming the prompt id and test case id, so a template that would fail at run time fails `validate` before any API call. (A template whose output is random — e.g. the `random` filter — can still fail on a later render; that case is then recorded as an `internal_error`, below.)
>
> Single-brace `{variable}` syntax, a `max_tokens` below 200 with `llm_judge`, and a long `expected_output` are warnings (exit 0).
>
> **Sandboxed templates:** `template` and `system` render in a Jinja2 `SandboxedEnvironment` — filters such as `{{ items | join(", ") }}` work, but unsafe attribute access is blocked: a single unsafe attribute such as `{{ x.__class__ }}` renders as empty, while calling or chaining through one (`{{ x.__class__() }}`, `{{ x.__class__.__mro__ }}`) raises `jinja2.exceptions.SecurityError` — which `evalblink validate` reports as an error.

## Test cases from a CSV or JSONL file

Past a handful of cases, keep them in a spreadsheet instead of YAML. The quickest start is `evalblink init --csv`: it writes the starter benchmark and its test cases as a CSV next to it (`benchmarks/my-benchmark.yaml` + `benchmarks/my-benchmark.csv`), ready to open in any spreadsheet app:

```
id,text,expected_output,tags
tc_001,Arrived a day early and works perfectly.,positive,easy
tc_002,Stopped charging after a week. Support never answered.,negative,easy
tc_003,"It does what the box says. Nothing more, nothing less.",neutral,edge_case
```

Add a row per case and a column per template variable. To use your own file, point `test_cases` at it (the path is relative to the YAML):

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

- **Columns** `id`, `evaluation`, `expected_output`, `criteria`, `reference`, `tags`, `scorer` and `human_score` are the test-case fields. Every other column is a template variable (`conversation` above fills `{{ conversation }}`); global `variables:` still apply.
- **CSV / TSV:** the first row names the columns; field headers match whatever their case (`ID`, `Tags`, `Expected Output`). `tags` are separated by `;`. An empty cell leaves a field unset, while an empty variable stays an empty string. A `json_match`, `json_schema` or `weighted_match` `expected_output` is JSON in its cell (`"{""label"": ""billing""}"` once quoted the CSV way). Every row must have one cell per column, and an unclosed quote is an error rather than a cell that swallows the next rows. Excel's exports work: "CSV UTF-8" (with a byte-order mark), and the `;`-separated CSV it writes in locales that use a decimal comma.
- **JSONL** (or `.ndjson`): one JSON object per line, with the same keys as a YAML test case; extra keys are variables too, and `null` leaves a field unset.
- **No `id` column?** Each row is named after the line it starts on (`line_2`, `line_3`…).
- Mistakes stop before any API call with the file and line: `Invalid test cases in bench.yaml: cases.csv:4: 3 cells but 2 columns — quote values that contain the delimiter`. A column whose name is close to a field (`expeced_output`) gets a warning, since it would otherwise silently become a variable.

The file is expanded into the usual list before validation, so `validate`, `--dry-run`, the run filters and every report treat its rows exactly like YAML test cases. [`benchmarks/csv_classification.yaml`](https://github.com/alexandre-darmon/evalblink/blob/main/benchmarks/csv_classification.yaml) is a complete example.

## Evaluation modes

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

### Contains, not contains and regex

For free-text replies with a checkable fact in them — "mentions the 30-day refund window", "echoes the order ID", "never leaks the staff discount code" — without paying for a judge. Like `exact_match`, they are deterministic, free and instant, and a case scores pass or fail.

```yaml
# Excerpt of benchmarks/text_checks.yaml
test_cases:
  - id: "order_id_echoed"
    evaluation: "contains"          # every text in expected_output appears in the reply
    expected_output: "ORD-48213"
  - id: "no_staff_code"
    evaluation: "not_contains"      # none of them appears
    expected_output: ["STAFF50"]
  - id: "refund_window"
    evaluation: "regex"             # the pattern matches somewhere in the reply
    expected_output: '(?i)\b30\W?days?\b'   # "30 days", "30-day", any space or dash
```

- **`contains`** passes when the reply includes `expected_output`, ignoring case. A list means *all* of them.
- **`not_contains`** passes when the reply includes none of them, ignoring case.
- **`regex`** passes when Python's [`re.search`](https://docs.python.org/3/library/re.html) finds the pattern anywhere in the reply. It is case-sensitive: start the pattern with `(?i)` to ignore case. Wrap it in `^…$` to match the whole reply; as with `exact_match`, whitespace around the reply is ignored. Write it in single quotes in YAML so backslashes stay as they are. `evalblink validate` rejects a pattern that doesn't compile. Avoid nested repeats such as `(a+)+`: on a long reply they can take practically forever, and Ctrl-C can't interrupt them.

Models often write `30 days` with a no-break or narrow space, `30‑day` with a non-breaking hyphen, or hide a soft hyphen inside a word. So `contains` and `not_contains` compare the text as a reader sees it, on both sides: Unicode spaces and line breaks count as one space, every dash as `-`, invisible characters are ignored, and fullwidth letters read as ASCII (Unicode NFKC). `regex` sees the reply as written, so a pattern can look for, or ban, a `—` or count characters: use `\s` for any space (it covers no-break spaces) and `\W` for any space or dash.

They match text, not meaning. "I can't guarantee a date" still contains "guarantee", and `contains: "30 days"` fails a reply that says "30-day" (the regex above accepts both). So check what must, or must never, appear verbatim (a code, an ID, a number, a banned phrase), keep each text short, and use `llm_judge` when the wording can vary.

In a CSV file the `expected_output` cell holds one text (for several, use one `regex` such as `(?is)(?=.*refund)(?=.*30 days)` — `i` ignores case like `contains`, `s` lets `.` cross lines —, or several rows).

### JSON match and JSON Schema

For structured output — extraction, routing, tool arguments — checked field by field, with no judge. Both parse the reply as JSON (a markdown code fence around it is tolerated); a reply that isn't JSON fails the case (a task failure, not a pipeline error).

```yaml
# Excerpt of benchmarks/structured_output.yaml
test_cases:
  - id: "double_charge"
    evaluation: "json_match"          # these values must appear in the reply's JSON
    expected_output: {label: "billing", priority: 1, needs_human: true}
  - id: "shape"
    evaluation: "json_schema"         # the reply's JSON must satisfy this schema
    expected_output:
      type: object
      required: [label, priority, needs_human, summary]
      properties:
        label: {enum: [billing, order_issue, product_question, other]}
        priority: {enum: [1, 2, 3]}
```

- **`json_match`** compares the reply's JSON with `expected_output` (a mapping or a list). Objects match key by key, recursively, and keys the reply adds are ignored, so list only what matters. Arrays must have the same length and match element by element, in order. Strings ignore case and surrounding whitespace, like `exact_match`; numbers compare by value (`2` matches `2.0`), and `true` never matches `1`. The score is the share of expected values found (0.5 when one of two fields is right, shown on the progress line); the case passes only at 1.0.
- **`json_schema`** validates the reply's JSON against a [JSON Schema](https://json-schema.org/) (draft 2020-12 unless the schema's `$schema` says otherwise), with `jsonschema`. It passes or fails; the first violation (`priority: 4 is not one of [1, 2, 3]`) is kept in the case's `reasoning` field in the JSON record. `validate` rejects a schema that isn't valid.

In a CSV file, `expected_output` holds the JSON as text.

### Python scorer

When no built-in check fits — a business rule, a length budget, a call to your own library — write it in Python:

```yaml
  - id: "short_summary"
    evaluation: "python"
    scorer: "scorers.py:short_summary"   # file relative to the YAML : function
```

```python
# benchmarks/scorers.py
def short_summary(response, case):
    """response: the model's reply (text). case: the test case (id, variables, expected_output, tags…)."""
    words = len(response.split())
    return {
        "passed": words <= 25,
        "score": min(1.0, 25 / max(words, 1)),
        "reason": f"{words} words",
    }
```

A scorer returns `True` / `False`, or a dict with `passed` (bool) and optionally `score` (0–1, default 1.0 / 0.0) and `reason` (kept in the case's `reasoning`). Each file is imported once per run, so module-level setup runs once; scorers run in parallel threads, so keep them free of shared mutable state. `validate` imports the file and checks the function exists. A scorer that raises, or returns anything else, records the case as a pipeline error (`scorer_error`, shown as `⚠ scorer error: …`), excluded from the score like an API error — your bug never reads as a model failure. The scorer is your own code, run with your permissions: only use scorer files you trust.

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
  - "cohere/north-mini-code:free"

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

The judge reasons step by step before committing to a score (chain-of-thought-first, G-Eval / MT-Bench style). The judge returns a score of 1–5, normalized to 0–1. Cases scoring below `judge_threshold` count as failures. The judge gets its own output budget, `evaluation.judge_max_tokens` (default 4096), independent of the candidate's `inference.max_tokens`: judges that reason at length before answering can use thousands of tokens (the free `nvidia/nemotron-3.5-lightning:free` takes 2,000–6,000 per verdict, which is why the examples use the more concise `nemotron-3-ultra`) and return nothing when the budget runs out — raise it if you see `judge_api_error … finish_reason: length`. A failed judge call shows `⚠ judge API error: …` on the case's progress line, and a reply with no usable score (often JSON cut off by the budget) shows `⚠ judge error: …`; either way the reason is kept in the case's `error` field. The judge's reasoning is stored in the JSON record (`results[].test_cases[].reasoning`) — not in the Markdown report — so you can check *why* a response scored poorly. If the judge call fails or its reply has no usable score, the case is a pipeline error (`status` `judge_api_error` / `judge_error`, score `null`): it counts under **Errors** and is excluded from the score, never as a 0.

**Built-in bias mitigations:**
- Judge prompt instructs against rewarding length (verbosity bias: 10–20% magnitude)
- Judge prompt instructs against rewarding confident tone over accuracy (style bias)
- If `evaluation.judge_model` shares a vendor with any candidate model, evalblink warns you at run start (self-preference bias: 10–25% magnitude)

### Weighted match

A specialised scorer, kept for configs that use it: for any other structured output, `json_match` is the general tool. For structured outputs where the model must return a JSON array of `{"use_case", "percent", "order"}` objects. Three dimensions are scored independently and combined by weight.

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
  - "cohere/north-mini-code:free"

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

## Local and self-hosted models

A model can run on your own machine or network instead of through OpenRouter: the test data never leaves it, nothing is billed, and a run of local models needs no OpenRouter key. Name it `provider:model`; the part after the first colon is the name the server knows, colons included (`ollama:llama3.2:3b`).

```yaml
models:
  - "ollama:llama3.2"           # local
  - "openai/gpt-4o-mini"        # OpenRouter — both in one run, one table
```

Four servers are built in, at their default addresses:

| Prefix | Server | Default `base_url` |
|---|---|---|
| `ollama:` | [Ollama](https://ollama.com) | `http://localhost:11434/v1` |
| `lmstudio:` | [LM Studio](https://lmstudio.ai) | `http://localhost:1234/v1` |
| `llamacpp:` | llama.cpp's `llama-server` | `http://localhost:8080/v1` |
| `vllm:` | [vLLM](https://docs.vllm.ai) | `http://localhost:8000/v1` |

Any other server that speaks the OpenAI chat completions API (`POST {base_url}/chat/completions`) goes under `providers:`, which also moves a built-in one:

```yaml
providers:
  ollama:
    base_url: "http://gpu-box.local:11434/v1"   # Ollama on another machine
  lab:                                          # models: ["lab:qwen2.5-7b"]
    base_url: "http://10.0.0.5:8000/v1"
    api_key_env: "LAB_API_KEY"   # sent as `Authorization: Bearer …`; no key by default
    concurrency: 2               # requests at a time to this server (default 1)
    timeout: 900                 # seconds per request (default 600)
```

An OpenRouter ID always has a `vendor/` before any colon (`meta-llama/llama-3.3-70b-instruct:free`), so it is never read as a provider. The judge (`evaluation.judge_model`) can be local too.

**Before the first call**, `run` asks each local server for its models (`GET {base_url}/models`). A server that doesn't answer stops the run with how to start it (`Can't reach ollama at http://localhost:11434/v1 (ConnectError). Start it: ollama serve`); a model it doesn't list stops it with how to get it and what it has (`… has no model 'llama3.2'. Get it: ollama pull llama3.2. Available: qwen2.5:7b`). Ollama's `llama3.2` matches its `llama3.2:latest`. llama.cpp is not checked (it serves the model it loaded, whatever the name), nor is a server that doesn't list models. A provider whose `api_key_env` variable is unset stops the run too. When every model and the judge are local, `OPENROUTER_API_KEY` is not needed.

**Cost** reads $0 (unless the server reports a cost in `usage`): it runs on your hardware. `--dry-run` prices local models at $0, says so, and makes no network call at all when every model is local. Tokens come from the server's `usage`; a server that leaves it out counts 0.

**Thinking models** (qwen3, gemma4, deepseek-r1…), common locally, reason before they answer and spend `max_tokens` on it: give them headroom (`benchmarks/local_models.yaml` uses 2048), or every reply comes back empty (`… returned empty/non-text ('') content. Increase max_tokens. (finish_reason: length)`).

**Latency** is the server's: load time on a cold model included, so the first case can be slow. Each server gets one request at a time by default (`concurrency` per provider), so a local server's queue never counts as latency; the run's own `concurrency` still applies to OpenRouter models, and the progress header says both (`5 at a time (ollama: 1 at a time)`). Raise a provider's `concurrency` only if the server really serves requests in parallel (vLLM, or Ollama with `OLLAMA_NUM_PARALLEL`).

**Cache and errors** work as with OpenRouter: the cache key uses the full ID (`ollama:qwen2.5`), so two servers with a model of the same name never share replies; transient failures are retried; a failed call is an `api_error` named after the server (`ollama (http://localhost:11434/v1) error …`). Example: `benchmarks/local_models.yaml`.

## Latency

Every candidate call's latency is recorded: the wall time of the attempt that returned the reply, from sending the request to parsing the answer, in seconds. Retries, backoff waits and judge calls are left out, so a rate-limited first attempt doesn't inflate it.

- **Per case:** `latency_s` in the JSON record, the last figure on the progress line (`pass  0.62s`), and a column in the Markdown report. An API error has none (`null`).
- **Per combo:** `latency_p50_s` (median), `latency_p95_s` and `latency_mean_s` over the calls that returned a reply. The terminal table shows the p50, the Markdown report p50 and p95.
- **Fastest:** with several combos, the recommendation names the one with the lowest p50, with its score, so you can weigh speed against quality.
- **`compare`** shows the p50 change per combo (`-0.12 s ↓` is faster). A record from before latency was measured shows `—`.

Latency is what you get through OpenRouter (or from your local server — see [Local and self-hosted models](#local-and-self-hosted-models)), from where the run happens: it includes the network and OpenRouter's routing, and it rises with `concurrency` when a provider queues requests. Compare combos within one run, or runs made the same way. A cached reply keeps the latency measured when it was fetched, so a rerun from the cache reports the same numbers; use `--no-cache` to measure afresh.

## Statistics: intervals, noise and repeats

A score from 20 or 50 test cases is an estimate: run it on other cases of the same kind and it would move by several points. evalblink says by how much, so a 2-point gap isn't read as a winner.

- **95 % interval.** Every combo's score comes with a 95 % confidence interval (Wilson score interval): `96.0% (87–99)` in the terminal table, a `95% CI` column in the Markdown report, `score_ci` in the JSON. It is computed over the number of scored *test cases*: more cases, narrower interval.
- **Within noise.** The recommendation lists the combos whose score can't be told apart from the best one's at 95 % (Newcombe's interval for the difference of two proportions contains 0), closest first: `Within noise : anthropic/claude-sonnet-4.6 / v1 (94.0%), … — add test cases to separate them.` Treat those as ties and decide on cost or latency. `compare` marks a quality change of the same kind `within noise` in its Notes column, and the verdict says how many gains are beyond noise.
- **`repeats: N`** runs every model × prompt × test case N times. Models are not fully deterministic, even at `temperature: 0`; repeats show it. Each repeat is a separate call with its own cache entry (`conv_001 #2` on the progress line and in `compare --detailed`), the score is the pass rate over every repeat, and a case that passes on some repeats and fails on others is **flaky**: the recommendation counts them and the JSON lists them per combo (`flaky`). Repeats steady each case's verdict but are not new test cases, so the interval stays as wide as the number of test cases allows. `--dry-run` multiplies the estimate by N. With repeats, the per-tag table's **Cases** column counts runs (3 repeats of one case count 3).

These are statistics over your test set, not guarantees: they can't fix a test set that doesn't look like your production traffic.

## Cost estimation

Before running a benchmark, estimate the cost without making any completion calls (only the public OpenRouter model catalog is fetched, cached for 24h):

```bash
evalblink run my_benchmark.yaml --dry-run    # the README Quickstart config
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
Estimate only — completion tokens assume max_tokens (capped at each model's output limit); prompt tokens ≈ chars/4.
✓ Within budget: estimate $0.089304 ≤ max_cost_usd $10.000000.
```

Per combo, **Est completion tok** is `max_tokens` (or the model's output limit, if lower) × the number of test cases × `repeats` (a ceiling — real outputs are usually shorter), and **Est cost** includes that combo's share of judge calls. The `Judge cost` line appears only when the estimated judge cost is non-zero — so not with a free (`:free`) judge model. A model that isn't in the catalog at all is almost always a typo: `--dry-run` prints `✗ Unknown model 'openai/gpt-4o-mni': not in the OpenRouter catalog — did you mean 'openai/gpt-4o-mini'?` and exits 1. A model listed with a negative or non-numeric price (such as the `openrouter/*` routers) has unknown pricing: it is excluded from the estimate with a warning (`No pricing for '<model>' — excluded from the estimate.`). Each budget is capped at the model's own output limit from the catalog (`top_provider.max_completion_tokens`); when `inference.max_tokens` or `evaluation.judge_max_tokens` is above it, `--dry-run` warns: `inference.max_tokens (5000) is above <model>'s output limit (2048 tokens) — lower it to 2048 or less.`

Set `max_cost_usd` in the YAML to enforce a budget: `--dry-run` exits 1 if the estimate exceeds it, which makes it safe to gate in CI before committing to a full run.

## Test case stratification

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

## Concurrency and retries

API calls run concurrently using `ThreadPoolExecutor` from Python's standard library — no async complexity required.

```yaml
concurrency: 5    # max parallel requests (default: 5)
```

| Mode | Estimated duration (50 cases × 3 models × 2 prompts) |
|---|---|
| Sequential | ~8–12 minutes |
| Concurrent (default: 5) | ~45–90 seconds |

Transient API failures — HTTP 408 / 429 / 502 / 503 / 504, whether on the status line or in the JSON error body, plus local timeouts and connection errors — are retried with exponential backoff (1 s, 2 s, 4 s), up to 4 attempts. A rate limit (429) waits longer — 5 s, 10 s, 20 s plus up to 1 s of random jitter, so parallel workers don't retry in lockstep — or the server's `Retry-After` seconds when it sends them (capped at 60 s); when the retries run out, the error says it was rate-limited and suggests retrying later, lowering `concurrency` or adding a paid model. Other client-side errors (an undecodable body, a redirect loop) fail the call at once, without a retry. Any other failure of a call surfaces as a clear error (with the HTTP status when known): an error status, a non-JSON body such as an HTML 502 page, a reply with no usable `choices` / `message`, null or empty content, or missing usage data.

**A failed call never aborts the run.** A candidate call that still fails after its retries — or hits a non-transient error such as an unknown model or a bad key — is recorded with `status: "api_error"`, `match_score: null`, `match_result: false`, an empty `response`, the message in `error`, and zero tokens/cost; its progress line shows `⚠ API error:` and the reason, and evalblink carries on. A failed judge call is recorded the same way as a judge error (see [LLM-as-judge](#llm-as-judge)). Either kind counts in the **Errors** column and is excluded from the score; the JSON and Markdown results are still written, and the end of the run prints a prominent warning with the number of pipeline errors excluded from scores. If *nothing* could be scored, the run exits 1 (see [CI/CD integration](#cicd-integration)).

No single test case can crash a run: anything unexpected while rendering or scoring one case (for example a template that uses `random`) records that case as `status: "internal_error"`, the same way as an API error, and the run carries on.

**Ctrl-C** stops a run at once: queued calls are dropped, calls in flight are abandoned (no more retries), evalblink prints `Interrupted — no results written.` and exits 130. Before the first call, `run` checks that it can write to `results/`; if it can't (a read-only directory, or a file named `results`), it exits 1 without spending anything.

## Caching

Every successful API response (candidate and judge) is cached under `.evalblink_cache/` by the SHA256 of the request payload: model (the full ID, `ollama:` prefix included), rendered system prompt, rendered prompt, temperature, and max_tokens. When you iterate on prompt v2, all v1 responses are served from cache at zero cost. Failed calls are never cached. A judge reply that arrives but can't be parsed (`judge_error`) *is* cached: `--no-cache` retries it for that run only (it neither reads nor writes the cache), so run `evalblink cache clear --yes` to drop the bad entry for good. `--no-cache` bypasses the cache for every call of the run, candidate and judge. A cache entry that can't be read or doesn't have the expected shape counts as a miss: the call is made again and the entry is overwritten. If the cache can't be written (for example the cache directory is read-only), evalblink prints `Could not write the response cache (…); continuing.` once and the run goes on uncached; a model catalog that can't be cached also gives a warning and the command goes on.

```bash
evalblink cache stats            # cached responses, their size, and whether the model catalog is cached
evalblink cache clear            # prints the response count only — deletes nothing
evalblink cache clear --yes      # delete every cached response
evalblink cache clear --yes --all  # also the cached model catalog (refetched on next use)
```

The model catalog (`.evalblink_cache/models.json`, used by `--dry-run`, `models` and `init`) is counted apart from responses, and `clear` keeps it unless you pass `--all`.

> Cache + `temperature: 0` = fully reproducible benchmarks.

## Results and history

Every run produces two files:

```
results/
  2026-06-10_143205_customer-support-classification.json   # full data, git-friendly
  2026-06-10_143205_customer-support-classification.md     # human-readable report
```

A second run of the same benchmark within the same second gets a `-2` (`-3`, …) suffix instead of overwriting the first.

Filenames are `YYYY-MM-DD_HHMMSS_<slug>`, where the slug is the benchmark `name` lowercased with each run of non-alphanumeric characters turned into `-`, capped at 60 characters, and `benchmark` when nothing is left (e.g. a name with no ASCII letters or digits) — so any valid name gives a writable path. In the Markdown report's per-case tables, a pipeline-error case shows its status in the Score column (`API error`, `judge API error`, `judge error`, `scorer error` or `internal error`) instead of a number. Lone UTF-16 surrogates (for example a `"\ud800"` escape in the YAML or in model output) are replaced with `?` in requests, terminal output and the Markdown report. The JSON includes schema versioning so `evalblink compare` handles old files gracefully — it never hard-blocks on a version mismatch. Version 2 adds, per case, `latency_s` (and `sample` with `repeats`) and, per combo, `score_ci`, `latency_p50_s` / `latency_p95_s` / `latency_mean_s`, `repeats` and `flaky`; a version 1 record reads with no latency, and its intervals are recomputed from its cases. Regenerate the Markdown report at any time:

```bash
evalblink report results/2026-06-10_143205_customer-support-classification.json
```

```bash
evalblink compare results/2026-06-08_101500_customer-support-classification.json \
                  results/2026-06-10_143205_customer-support-classification.json
```

```
 DELTA: 2026-06-08_101500_customer-support-classification → 2026-06-10_143205_customer-support-classification
 Model                          Prompt  Quality Δ    Cost Δ       p50 latency Δ   Notes
 anthropic/claude-sonnet-4.6    v1      +4.0% ↑      +0.0002 ↑    -0.08 s ↓       within noise
 openai/gpt-4o-mini             v1      +6.0% ↑      stable       stable          within noise
 anthropic/claude-sonnet-4.6    v2      stable       stable       -0.12 s ↓

Verdict: Run B improves quality on 2/3 shared combo(s). None of these gains is beyond noise (95%).
```

Changed combos are listed first. Scores leave out pipeline-error cases, so when a shared combo's error count differs between the runs, its quality Δ compares different sets of cases. Such a row is yellow, its Notes column reads `pipeline errors 2 → 0`, and the verdict leaves it out of the improved count and says how many were left out. Use `--detailed` to see which cases changed. `--detailed` adds per-combo test-case transitions (regressed / improved / new error / recovered) and a global summary.

`report` and `compare` exit with a one-line message when a results file is missing, unreadable, not JSON, or not shaped like a run record (it needs string `benchmark` and `timestamp` fields and a well-formed `results` list). `history` lists every readable run and prints `⚠  Skipping <reason>` for each file it can't use. `report` and `history` recompute the insights from `results`, so the `pipeline error(s) excluded from scoring` line and the history **Errors** column show the run total, matching the warning at the end of the run.

## CI/CD integration

### GitHub Action

The repository is a GitHub Action: it installs evalblink, runs the benchmark, puts the Markdown report in the job summary, and fails the job on the gate or on a regression.

```yaml
# .github/workflows/llm-eval.yml
on: [pull_request]
jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - id: eval
        uses: alexandre-darmon/evalblink@v0.2.0
        with:
          config: benchmarks/my-benchmark.yaml
          openrouter-api-key: ${{ secrets.OPENROUTER_API_KEY }}
          baseline: benchmarks/baseline.json   # optional
          max-drop: 2                          # points; default 2
      - uses: actions/upload-artifact@v4
        if: always()
        with: {name: evalblink-results, path: results/}
```

| Input | Default | |
|---|---|---|
| `config` | the only YAML in `benchmarks/` | Benchmark YAML |
| `args` | — | Extra `evalblink run` flags: `--tag smoke --limit 20`, `--no-cache`, `--dry-run` |
| `openrouter-api-key` | the job's `OPENROUTER_API_KEY` | Pass a secret; not needed with `--dry-run` |
| `baseline` | — | A kept run record; adds `compare <baseline> --fail-on-regression <max-drop>` |
| `max-drop` | `2` | Pass-rate drop, in points, the baseline tolerates |
| `package` | `evalblink` | What to `pip install`: `evalblink==0.2.0` to pin, `.` for a checkout |
| `python-version` | `3.12` | |

Outputs: `json` and `markdown`, the paths of the run record and report (empty on a dry run).

### Exit codes

`evalblink run` exits `0` (pass) or `1` (fail), so it also works as a plain step:

```yaml
- run: evalblink run my_benchmark.yaml
  env:
    OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
```

- **Exit 1 before any API call** when the config fails validation, `OPENROUTER_API_KEY` is unset, empty or whitespace-only while an OpenRouter model is used, or a local server is down or lacks a model. `--dry-run` needs no API key.
- **Exit 1 when nothing could be scored** — every case of the run is a pipeline error (`api_error`, `judge_api_error`, `judge_error`, `scorer_error`, `internal_error`). The results are still written.
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

The regression check compares pass rates point for point; `compare` also marks the changes that are within noise (see [Statistics](#statistics-intervals-noise-and-repeats)), which tells you whether a failed check is worth a second look or a re-run with more cases.

## CLI details

**API key.** `OPENROUTER_API_KEY` comes from the environment or a `.env` file read from the directory you run `evalblink` in (or the nearest parent that has one), never from the installed package's directory. `evalblink run` exits 1 with a clear message, before any API call, if it is unset, empty or whitespace-only — unless every model it calls is local (`ollama:…`), which needs none. A key with a non-ASCII character (a pasted `…` or curly quote) can't be sent in an HTTP header: every call fails with a clear `api_error` (and `models` / `--dry-run` / `init` say so in one line). Save `.env` as UTF-8 (a BOM is fine); a file in another encoding is ignored with a warning. `evalblink validate` and `evalblink run --dry-run` need no API key and make no completion calls (`--dry-run` only fetches the public model catalog).

`evalblink run` and `evalblink validate` with no path use the only YAML in `benchmarks/` (what `init` writes) and print `Using <path>` to stderr. With several YAMLs there, they use `benchmarks/exact_match_classification.yaml` when it exists (as in this repo) and otherwise exit with the list to pick from; with none, they suggest `evalblink init`.

- `models --min-context` takes values like `100k`, `8K` or `131072`; anything else is an argparse usage error (exit 2). A catalog price that is negative or non-numeric (OpenRouter quotes `-1` for its `openrouter/*` routers) is unknown: `models` shows it as `?` and `--free` leaves it out.
- `init` asks three questions: a name, the evaluation mode (`exact_match` or `llm_judge`) and the models to compare, comma-separated. Press Enter at each for the defaults. It then writes a complete, commented starter: an example prompt and example test cases, `temperature: 0`, `max_tokens: 1024` (headroom for free reasoning models), `concurrency: 2` (free models rate-limit bursts) and, for `llm_judge`, the free judge `nvidia/nemotron-3-ultra-550b-a55b:free`. The file validates and runs as-is; replace the example test cases with your own in your editor. The default model is the free `cohere/north-mini-code:free` (or the first free catalog model if OpenRouter has retired it). While you choose, it lists up to five free text models from the catalog (text in, text-only out; audio and image generators are left out) and a small paid option without free-tier rate limits, `openai/gpt-oss-20b`, with its price. That option is never preselected. Models absent from the catalog get a warning, and an empty name or model list is asked again. `init` carries on without suggestions when the catalog can't be fetched. The output file is the optional path argument, or `benchmarks/<slug>.yaml`, relative to the directory you run `init` in (never the install directory); `init` prints its absolute path when done (`benchmarks/benchmark.yaml` for a name with no letters or digits); an existing file is only overwritten after you confirm. `init --yes` asks nothing and writes the `exact_match` starter with the default model, refusing to overwrite an existing file. Before writing, `init` runs `validate` on the result and exits 1 without writing if there are errors.
