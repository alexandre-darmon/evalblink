# evalblink ⚡

**Benchmark any LLM on your data in one command.**  
Compare quality, cost, and latency across models and prompts. Get a structured, versioned report.

---

## The problem

Most teams pick their LLM the wrong way.

They copy-paste a few prompts into ChatGPT, Claude, and Mistral. They compare outputs by eye. They ship based on intuition.

Academic benchmarks (MMLU, HumanEval) don't help — they measure generic capabilities, not performance on *your* prompts, *your* domain, *your* labels.

**evalblink fills that gap.**

Define your benchmark once in YAML. Run it across N models × N prompt versions in one command. Get a structured report with quality scores, cost, and latency — so you make the right model decision with evidence, not gut feeling.

---

## Install

```bash
pip install evalblink
```

**Requirements:** Python 3.10+ · OpenRouter API key

```bash
export OPENROUTER_API_KEY=your_key_here
```

---

## Quickstart

```bash
# Scaffold a new benchmark
evalblink init

# Run it
evalblink run benchmarks/my_benchmark.yaml

# Estimate cost before running
evalblink run benchmarks/my_benchmark.yaml --dry-run

# Compare two runs
evalblink compare results/run_001.json results/run_002.json

# List past runs
evalblink history

# Browse available models with pricing
evalblink models
```

**First benchmark running in under 5 minutes.**

---

## Define your benchmark in YAML

```yaml
name: "Support ticket classification"
judge_model: "anthropic/claude-sonnet-4-6"
max_cost_usd: 5.00

prompts:
  - id: "v1"
    template: >
      Classify this ticket. Return only one label among: {{labels}}.
      Ticket: {{ticket}}

  - id: "v2"
    system: "You are an expert support classifier."
    template: >
      Analyze the following ticket and return only the matching label
      among: {{labels}}.
      Ticket: {{ticket}}

models:
  - "openai/gpt-4o-mini"
  - "anthropic/claude-sonnet-4-6"
  - "mistralai/mistral-small-3"

variables:
  labels: "billing, auth, payment, other"

test_cases:
  - id: "ticket_001"
    variables:
      ticket: "I can't see my invoice from last month."
    expected_output: "billing"
    evaluation: "exact_match"

  - id: "ticket_002"
    variables:
      ticket: "Explain why I was charged twice."
    evaluation: "llm_judge"
    criteria: >
      The response must identify a billing dispute.
      It must return a single label without explanation.
```

---

## Results matrix

```
Support ticket classification
Run: 2026-06-10 14:32 | ID: a3f2b1

RESULTS MATRIX

              | gpt-4o-mini        | claude-sonnet      | mistral-small-3
--------------|--------------------|--------------------|-----------------
prompt_v1     |                    |                    |
  Quality     | 87%                | 94%                | 82%
  Cost        | $0.06              | $0.45              | $0.04
  Latency     | 1.1s               | 2.3s               | 0.9s
--------------|--------------------|--------------------|-----------------
prompt_v2     |                    |                    |
  Quality     | 91% (+4%) ↑        | 96% (+2%) ↑        | 88% (+6%) ↑
  Cost        | $0.07 (+$0.01)     | $0.46 (stable)     | $0.04 (stable)
  Latency     | 1.2s (+0.1s)       | 2.3s (stable)      | 0.8s (-0.1s) ↓

VERDICT
Best quality : claude-sonnet / prompt_v2 (96%)
Best value   : mistral-small-3 / prompt_v2 (88% quality, $0.04/run)
```

---

## Evaluation modes

### Exact match

For classification tasks with a known ground truth label.

Normalization before comparison: whitespace stripped, lowercased, last meaningful word extracted if the model returns a sentence.

### LLM-as-judge

For open-ended tasks (summarization, generation, explanation) where no single correct output exists.

The judge LLM scores each response 1–5 against your criteria. Score is normalized to 0–100% in the report.

---

## Output

Each run generates two files in `/results`:

```
results/
  2026-06-10_14-32_support-classification_a3f2b1.json   ← full results
  2026-06-10_14-32_support-classification_a3f2b1.md     ← human-readable report
```

The markdown report includes:

- Full results matrix (quality / cost / latency)
- Detailed breakdown per prompt × model combination
- Failed test cases with model response vs expected
- Cost analysis per model and prompt version
- Live model specs fetched from OpenRouter (context window, pricing)
- Auto-generated verdict

> `results/` is git-ignored by default. Your benchmark history stays local.

---

## Model metadata

evalblink fetches live model specs from OpenRouter at the start of each run.

```
MODEL SPECS

| Model              | Context Window | Input (per 1M) | Output (per 1M) |
|--------------------|----------------|----------------|-----------------|
| gpt-4o-mini        | 128k tokens    | $0.15          | $0.60           |
| claude-sonnet-4-6  | 200k tokens    | $3.00          | $15.00          |
| mistral-small-3    | 32k tokens     | $0.10          | $0.30           |

⚠️ ticket_031 exceeds mistral-small-3's context window (32k).
   Results for this case on this model may be unreliable.
```

Specs are cached locally for 24h to avoid repeated API calls.

---

## Stack

| Component | Tool |
|---|---|
| CLI | Typer |
| Terminal output | Rich |
| YAML parsing | PyYAML |
| Template injection | Jinja2 |
| API calls | httpx (async) |
| LLM routing | OpenRouter |
| Result storage | Local JSON |
| Report generation | Python + Markdown |

---

## Repo structure

```
evalblink/
├── evalblink/
│   ├── cli.py              # Typer CLI entrypoint
│   ├── runner.py           # Benchmark execution logic
│   ├── evaluator.py        # exact_match + llm_judge
│   ├── reporter.py         # Markdown + terminal output
│   ├── openrouter.py       # API client + model metadata
│   ├── comparator.py       # Compare two run JSON files
│   └── models.py           # Pydantic data models
├── benchmarks/
│   └── example_classification.yaml
├── results/
│   └── .gitkeep
├── pyproject.toml
├── .env.example
└── README.md
```

---

## Roadmap

- [x] V1 — CLI tool (this)
- [ ] V2 — Streamlit dashboard for non-technical stakeholders
- [ ] CSV import for test cases
- [ ] HTML report export
- [ ] Tag-based test case filtering

---

## License

MIT — free to use, fork, and build on.

---

## Author

Built by [Alexandre Darmon](https://alexdarmon.net) — AI Product Manager.  
After running LLM evaluations manually across BNP Paribas AI products, I built the tool I wished existed.

→ [LinkedIn](https://linkedin.com/in/alexdarmon) · [alexdarmon.net](https://alexdarmon.net)
