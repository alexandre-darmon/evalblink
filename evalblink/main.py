"""Benchmark LLM prompts × models on your own data via OpenRouter."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import httpx
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table

from evalblink import cache, compare, estimate, openrouter, reporter, runner
from evalblink.validator import validate as _validate_config

DEFAULT_CONFIG = "benchmarks/exact_match_classification.yaml"
RESULTS_DIR = "results"


def load_config(filepath):
    """Load a benchmark YAML, exiting with a clear message on failure."""
    try:
        with open(filepath) as stream:
            return yaml.safe_load(stream)
    except FileNotFoundError:
        sys.exit(f"Config not found: {filepath}")
    except yaml.YAMLError as exc:
        sys.exit(f"Invalid YAML in {filepath}: {exc}")


def cmd_run(args):
    """Run a benchmark, write the report, and exit on the CI quality gate."""
    config = load_config(args.config)

    errors, warnings = _validate_config(config)
    console = Console()
    for w in warnings:
        console.print(f"[yellow]⚠[/yellow]  {w}")
    if errors:
        for e in errors:
            console.print(f"[red]✗[/red]  {e}")
        console.print(f"\n[red]Config has {len(errors)} error(s) — fix before running.[/red]")
        sys.exit(1)

    if args.dry_run:
        # Estimate cost from the pricing catalog only — no completion calls.
        with httpx.Client() as client:
            models_meta = openrouter.fetch_models(client)
        est = estimate.estimate(config, models_meta)
        reporter.render_estimate(est)
        sys.exit(1 if est["over_budget"] else 0)

    use_cache = not args.no_cache
    results, timestamp = runner.run(config, verbose=args.verbose, use_cache=use_cache)
    result = reporter.write(config, results, timestamp)

    # Top-level CI gate (0-100 %); separate from the per-case scorer thresholds
    # under `evaluation:`.
    threshold = config.get("quality_threshold")
    if threshold is None:
        print("\nGate: no quality_threshold set — not enforcing pass/fail.")
        sys.exit(0)
    if result["passed"]:
        print(f"\nPASS: best score meets quality_threshold ({threshold}).")
        sys.exit(0)
    print(f"\nFAIL: best score is below quality_threshold ({threshold}).")
    sys.exit(1)


def cmd_compare(args):
    """Print the quality/cost delta between two persisted run records."""
    try:
        record_a = compare.load_record(args.file_a)
        record_b = compare.load_record(args.file_b)
    except FileNotFoundError as exc:
        sys.exit(f"Run record not found: {exc.filename}")
    if args.detailed:
        reporter.render_detailed(compare.detailed_diff(record_a, record_b))
    else:
        reporter.render_comparison(compare.diff(record_a, record_b))
    sys.exit(0)


def cmd_report(args):
    """Regenerate the Markdown report from an existing JSON result file."""
    try:
        record = compare.load_record(args.file)
    except FileNotFoundError as exc:
        sys.exit(f"Run record not found: {exc.filename}")
    reporter.write_from_record(record)
    sys.exit(0)


def cmd_history(args):
    """List all past runs in the results/ directory."""
    if not os.path.isdir(RESULTS_DIR):
        print("No results directory found. Run a benchmark first.")
        sys.exit(0)

    records = []
    for fname in os.listdir(RESULTS_DIR):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(RESULTS_DIR, fname)
        try:
            with open(path) as f:
                data = json.load(f)
            total_cost = sum(r.get("total_cost", 0) for r in data.get("results", []))
            insights = data.get("insights") or {}
            best = insights.get("best_quality", {})
            best_score = best.get("score")
            errors = insights.get("errors", 0)
            records.append(
                {
                    "run_id": data.get("run_id", os.path.basename(fname)),
                    "benchmark": data.get("benchmark", "—"),
                    "timestamp": data.get("timestamp", "—"),
                    "best_score": best_score,
                    "total_cost": total_cost,
                    "errors": errors,
                }
            )
        except (json.JSONDecodeError, OSError):
            continue

    if not records:
        print("No run records found in results/.")
        sys.exit(0)

    records.sort(key=lambda r: r["timestamp"], reverse=True)

    console = Console()
    table = Table(title="Run History")
    table.add_column("Run ID")
    table.add_column("Benchmark")
    table.add_column("Timestamp")
    table.add_column("Best Score", justify="right")
    table.add_column("Total Cost", justify="right")
    table.add_column("Errors", justify="right")
    for r in records:
        score = "—" if r["best_score"] is None else f"{r['best_score']:.1f}%"
        table.add_row(
            r["run_id"],
            r["benchmark"],
            r["timestamp"],
            score,
            f"${r['total_cost']:.6f}",
            str(r["errors"]) if r["errors"] else "—",
        )
    console.print(table)
    sys.exit(0)


def _parse_context(value: str) -> int:
    """Parse a context-length string like '100k', '8K', or '131072' to an int."""
    v = str(value).strip().lower()
    if v.endswith("k"):
        return int(float(v[:-1]) * 1_000)
    if v.endswith("m"):
        return int(float(v[:-1]) * 1_000_000)
    try:
        return int(v)
    except ValueError:
        sys.exit(
            f"Invalid --min-context value: {value!r}. Use e.g. '100k' or '131072'."
        )


def cmd_models(args):
    """List available OpenRouter models and their pricing."""
    with httpx.Client() as client:
        models_meta = openrouter.fetch_models(client, use_cache=not args.no_cache)

    rows = sorted(models_meta.items())

    if args.provider:
        prefix = args.provider.rstrip("/") + "/"
        rows = [(k, v) for k, v in rows if k.startswith(prefix)]

    if args.free:
        rows = [(k, v) for k, v in rows if v["prompt"] == 0 and v["completion"] == 0]

    if args.min_context:
        min_tok = _parse_context(args.min_context)
        rows = [(k, v) for k, v in rows if (v.get("context_length") or 0) >= min_tok]

    if not rows:
        print("No models match the given filters.")
        sys.exit(0)

    console = Console()
    table = Table(title=f"OpenRouter models ({len(rows)} shown)")
    table.add_column("Model ID")
    table.add_column("Context", justify="right")
    table.add_column("Prompt $/1M", justify="right")
    table.add_column("Completion $/1M", justify="right")
    for model_id, meta in rows:
        ctx = meta.get("context_length")
        ctx_str = f"{ctx // 1000}k" if ctx else "—"
        table.add_row(
            model_id,
            ctx_str,
            f"{meta['prompt'] * 1_000_000:.3f}" if meta["prompt"] else "free",
            f"{meta['completion'] * 1_000_000:.3f}" if meta["completion"] else "free",
        )
    console.print(table)
    sys.exit(0)


def cmd_cache_stats(args):
    """Print cache entry count and total size."""
    s = cache.stats()
    size_kb = s["size_bytes"] / 1024
    print(f"Cache entries : {s['entries']}")
    print(f"Cache size    : {size_kb:.1f} KB")
    sys.exit(0)


def cmd_cache_clear(args):
    """Delete all cached responses."""
    if not args.yes:
        s = cache.stats()
        print(f"{s['entries']} cache entries. Pass --yes to confirm deletion.")
        sys.exit(0)
    removed = cache.clear()
    print(f"Cleared {removed} cache entries.")
    sys.exit(0)


def _jinja_vars(text: str) -> list[str]:
    """Return unique variable names found in a Jinja2 template string."""
    return list(dict.fromkeys(re.findall(r"\{\{\s*(\w+)\s*\}\}", text)))


def cmd_init(args):
    """Interactively scaffold a new benchmark YAML file."""
    console = Console()
    console.print("\n[bold]evalblink init[/bold] — scaffold a new benchmark\n")

    # 1. Name
    name = Prompt.ask("Benchmark name")

    # 2. Evaluation mode
    mode = Prompt.ask(
        "Evaluation mode",
        choices=["exact_match", "llm_judge"],
        default="exact_match",
    )

    # 3. Models — fetch catalog for free-model suggestions
    with httpx.Client() as client:
        models_meta = openrouter.fetch_models(client, use_cache=True)

    free_suggestions = [
        (k, v)
        for k, v in sorted(models_meta.items())
        if v.get("prompt") == 0 and v.get("completion") == 0
    ][:5]

    console.print("\n[bold]Models[/bold]")
    if free_suggestions:
        console.print("Here are some free OpenRouter models:")
        free_table = Table(show_header=True, header_style="bold", box=None)
        free_table.add_column("Model ID", style="cyan")
        free_table.add_column("Context", justify="right")
        for mid, meta in free_suggestions:
            ctx = meta.get("context_length")
            free_table.add_row(mid, f"{ctx // 1000}k" if ctx else "—")
        console.print(free_table)
        console.print(
            "Use an ID from above, or paste any OpenRouter model ID. "
            "Tip: [bold]evalblink models --free[/bold] shows all free options.\n"
        )

    models: list = []
    while True:
        label = "Model ID" if not models else "Another model (blank to finish)"
        m = Prompt.ask(f"  {label}", default="")
        if not m:
            if not models:
                console.print("[yellow]At least one model is required.[/yellow]")
                continue
            break
        models.append(m)

    # 4. Prompt template
    console.print(
        "\n[bold]Prompt template[/bold] — use [cyan]{{ variable }}[/cyan] for dynamic values"
    )
    if mode == "exact_match":
        console.print(
            "  Example: [dim]Classify this text: {{ text }}. Choose from: {{ labels }}[/dim]"
        )
    else:
        console.print("  Example: [dim]Write a concise summary of: {{ text }}[/dim]")
    template = Prompt.ask("  Template")
    system = Prompt.ask("  System message (optional)", default="")

    # 5. Variable scoping
    all_vars = _jinja_vars(template + " " + system)
    global_vars: dict = {}
    per_case_vars: list = []
    if all_vars:
        console.print(
            f"\n[bold]Template variables:[/bold] {', '.join(all_vars)}\n"
            "Mark each as [g]lobal (same for all test cases) or [p]er-case (varies)."
        )
        for var in all_vars:
            choice = Prompt.ask(f"  '{var}'", choices=["g", "p"], default="p")
            if choice == "g":
                val = Prompt.ask(f"    Value for [cyan]{var}[/cyan]")
                global_vars[var] = val
            else:
                per_case_vars.append(var)

    # 6. Judge config (llm_judge only)
    eval_config: dict = {}
    if mode == "llm_judge":
        console.print("\n[bold]Judge configuration[/bold]")
        judge_model = Prompt.ask("  Judge model", default="openai/gpt-4o")
        while True:
            raw = Prompt.ask("  Judge threshold (0.0–1.0)", default="0.70")
            try:
                judge_threshold = float(raw)
                break
            except ValueError:
                console.print("[yellow]Enter a decimal, e.g. 0.70[/yellow]")
        eval_config = {"judge_model": judge_model, "judge_threshold": judge_threshold}

    # 7. Inference
    console.print("\n[bold]Inference settings[/bold]")
    while True:
        raw = Prompt.ask("  Temperature", default="0")
        try:
            temperature = float(raw)
            break
        except ValueError:
            console.print("[yellow]Enter a number, e.g. 0 or 0.7[/yellow]")
    max_tokens_default = "1024" if mode == "llm_judge" else "100"
    while True:
        raw = Prompt.ask("  Max tokens", default=max_tokens_default)
        try:
            max_tokens = int(raw)
            break
        except ValueError:
            console.print("[yellow]Enter a whole number, e.g. 100 or 1024[/yellow]")

    # 8. Test cases
    test_cases: list = []
    tc_num = 1
    console.print("\n[bold]Test cases[/bold]")
    while True:
        console.print(f"\n  [bold]Test case {tc_num}[/bold]")
        tc_id = Prompt.ask("    ID", default=f"tc_{tc_num:03d}")
        tc: dict = {"id": tc_id, "evaluation": mode}

        if per_case_vars:
            tc_vars: dict = {}
            for var in per_case_vars:
                tc_vars[var] = Prompt.ask(f"    {{{{ {var} }}}}")
            tc["variables"] = tc_vars

        if mode == "exact_match":
            tc["expected_output"] = Prompt.ask("    Expected output")
        else:
            tc["criteria"] = Prompt.ask("    Evaluation criteria")

        tags_str = Prompt.ask("    Tags (comma-separated, optional)", default="")
        if tags_str.strip():
            tc["tags"] = [t.strip() for t in tags_str.split(",") if t.strip()]

        test_cases.append(tc)
        tc_num += 1

        if not Confirm.ask("\n  Add another test case?", default=False):
            break

    # 9. CI/CD gate
    console.print("\n[bold]CI/CD quality gate[/bold]")
    while True:
        qt_str = Prompt.ask("  Quality threshold 0–100 (Enter to skip)", default="")
        if not qt_str.strip():
            quality_threshold = None
            break
        try:
            quality_threshold = int(qt_str)
            break
        except ValueError:
            console.print(
                "[yellow]Enter a whole number 0–100, or press Enter to skip[/yellow]"
            )

    # 10. Output path
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    default_path = f"benchmarks/{slug}.yaml"
    out_path = Prompt.ask("\nOutput file", default=default_path)

    if os.path.exists(out_path):
        if not Confirm.ask(f"  {out_path} already exists — overwrite?", default=False):
            sys.exit(0)

    # 11-12. Build config and write
    config: dict = {"name": name}
    if quality_threshold is not None:
        config["quality_threshold"] = quality_threshold
    config["models"] = models
    config["inference"] = {"temperature": temperature, "max_tokens": max_tokens}
    if eval_config:
        config["evaluation"] = eval_config
    if global_vars:
        config["variables"] = global_vars
    prompt_entry: dict = {"id": "v1", "template": template}
    if system:
        prompt_entry["system"] = system
    config["prompts"] = [prompt_entry]
    config["test_cases"] = test_cases

    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(
            config, f, allow_unicode=True, default_flow_style=False, sort_keys=False
        )

    # 13. Success
    console.print(f"\n[green]✓ Created[/green] [bold]{out_path}[/bold]\n")
    console.print("Next steps:")
    console.print(f"  [bold]evalblink run {out_path}[/bold]      run the benchmark")
    console.print(f"  [bold]evalblink run {out_path} -v[/bold]   verbose output")
    console.print("  [bold]evalblink -h[/bold]                   full CLI help")
    sys.exit(0)


def cmd_validate(args):
    """Lint a benchmark YAML — errors and warnings, no API calls."""
    config = load_config(args.benchmark)
    errors, warnings = _validate_config(config)
    console = Console()
    for w in warnings:
        console.print(f"[yellow]⚠[/yellow]  {w}")
    for e in errors:
        console.print(f"[red]✗[/red]  {e}")
    if errors:
        console.print(
            f"\n[red]Config has {len(errors)} error(s) — fix before running.[/red]"
        )
        sys.exit(1)
    console.print("[green]✓[/green]  Config is valid")
    sys.exit(0)


_RUN_DESCRIPTION = "Run a benchmark YAML against all model × prompt combinations."

_RUN_EPILOG = """\
YAML CONFIG KEYS
  name                          required
  models                        list of OpenRouter model IDs  (required)
  prompts[].id                  prompt identifier             (required)
  prompts[].template            Jinja2 — use {{ var }}        (required)
  prompts[].system              optional system message
  inference.temperature         default: 0
  inference.max_tokens          default: 100
  variables                     global key/value pairs injected into templates
  quality_threshold             CI gate 0–100; exit 1 if best score is below
  evaluation.judge_model        required for llm_judge
  evaluation.judge_threshold    default: 0.70
  test_cases[].id               required
  test_cases[].evaluation       exact_match | llm_judge | weighted_match  (required)
  test_cases[].expected_output  required for exact_match
  test_cases[].criteria         required for llm_judge
  test_cases[].reference        optional gold answer injected into the judge prompt
  test_cases[].variables        per-case variables
  test_cases[].tags             list of strings for per-category breakdown
  max_cost_usd                  --dry-run exits 1 if the estimate exceeds this

EXAMPLES
  evalblink run benchmarks/classification.yaml
  evalblink run benchmarks/classification.yaml -v
  evalblink run benchmarks/classification.yaml --no-cache
  evalblink run benchmarks/classification.yaml --dry-run
"""

_COMPARE_DESCRIPTION = "Diff two run records — quality and cost deltas, no API calls."

_COMPARE_EPILOG = """\
OUTPUT COLUMNS
  Quality A / B   pass-rate % per run
  Quality Δ       ↑ / ↓ / stable (threshold: 5 pp)
  Cost A / B      total cost per combo (models + judge)
  Cost Δ          cost change
  Notes           a_only / b_only when a combo is missing from one run

--detailed also shows per-combo case transitions and a global summary:
  regressed   pass → fail
  improved    fail → pass
  new_error   scored → None  (pipeline error, not counted as a regression)
  recovered   None → scored

EXAMPLES
  evalblink compare results/run_a.json results/run_b.json
  evalblink compare results/run_a.json results/run_b.json --detailed
"""


def build_parser():
    parser = argparse.ArgumentParser(
        prog="evalblink",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    run_p = subparsers.add_parser(
        "run",
        help="run a benchmark config against all model × prompt combinations",
        description=_RUN_DESCRIPTION,
        epilog=_RUN_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    run_p.add_argument(
        "config",
        nargs="?",
        default=DEFAULT_CONFIG,
        help=(f"path to a benchmark YAML (default: {DEFAULT_CONFIG})"),
    )
    run_p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print per-test-case detail (response, score, status) during the run",
    )
    run_p.add_argument(
        "--no-cache",
        action="store_true",
        help="bypass the local cache and force fresh API calls for all candidate requests",
    )
    run_p.add_argument(
        "--dry-run",
        action="store_true",
        help="estimate cost from the OpenRouter pricing catalog without running the benchmark",
    )
    run_p.set_defaults(func=cmd_run)

    # --- compare ---
    compare_p = subparsers.add_parser(
        "compare",
        help="diff two run records — quality and cost deltas, no API calls",
        description=_COMPARE_DESCRIPTION,
        epilog=_COMPARE_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    compare_p.add_argument(
        "file_a",
        metavar="FILE_A",
        help="baseline run JSON (the earlier / reference run)",
    )
    compare_p.add_argument(
        "file_b",
        metavar="FILE_B",
        help="candidate run JSON (the run being evaluated)",
    )
    compare_p.add_argument(
        "-d",
        "--detailed",
        action="store_true",
        help=(
            "drill into per-test-case transitions and show a global summary "
            "(worst-regressed cases, per-tag net change)"
        ),
    )
    compare_p.set_defaults(func=cmd_compare)

    # --- report ---
    report_p = subparsers.add_parser(
        "report",
        help="regenerate the Markdown report from an existing JSON result file",
    )
    report_p.add_argument(
        "file",
        metavar="FILE",
        help="path to a result JSON file (from results/)",
    )
    report_p.set_defaults(func=cmd_report)

    # --- history ---
    history_p = subparsers.add_parser(
        "history",
        help="list all past runs in the results/ directory",
    )
    history_p.set_defaults(func=cmd_history)

    # --- init ---
    init_p = subparsers.add_parser(
        "init",
        help="interactively scaffold a new benchmark YAML",
    )
    init_p.set_defaults(func=cmd_init)

    # --- models ---
    models_p = subparsers.add_parser(
        "models",
        help="list available OpenRouter models and their pricing",
    )
    models_p.add_argument(
        "--provider",
        metavar="NAME",
        help="filter by provider prefix, e.g. 'anthropic'",
    )
    models_p.add_argument(
        "--free",
        action="store_true",
        help="show only free (zero-cost) models",
    )
    models_p.add_argument(
        "--min-context",
        metavar="N",
        help="minimum context window, e.g. '100k' or '131072'",
    )
    models_p.add_argument(
        "--no-cache",
        action="store_true",
        help="bypass the 24h model catalog cache and fetch fresh data",
    )
    models_p.set_defaults(func=cmd_models)

    # --- validate ---
    validate_p = subparsers.add_parser(
        "validate",
        help="lint a benchmark YAML — errors and warnings, no API calls",
    )
    validate_p.add_argument(
        "-b",
        "--benchmark",
        default=DEFAULT_CONFIG,
        metavar="FILE",
        help=f"path to a benchmark YAML (default: {DEFAULT_CONFIG})",
    )
    validate_p.set_defaults(func=cmd_validate)

    # --- cache ---
    cache_p = subparsers.add_parser(
        "cache",
        help="manage the local response cache",
    )
    cache_sub = cache_p.add_subparsers(dest="cache_command", required=True)

    stats_p = cache_sub.add_parser("stats", help="show entry count and total size")
    stats_p.set_defaults(func=cmd_cache_stats)

    clear_p = cache_sub.add_parser("clear", help="delete all cached responses")
    clear_p.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="confirm deletion without prompting",
    )
    clear_p.set_defaults(func=cmd_cache_clear)

    return parser


def main():
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
