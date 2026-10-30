"""Benchmark LLM prompts × models on your own data via OpenRouter."""

from __future__ import annotations

import argparse
import os
import sys

import httpx
import yaml
from dotenv import find_dotenv, load_dotenv
from rich.markup import escape
from rich.prompt import Confirm, Prompt
from rich.table import Table

from evalblink import (
    __version__,
    cache,
    compare,
    estimate,
    openrouter,
    reporter,
    runner,
    starter,
)
from evalblink.analysis import summarize
from evalblink.validator import validate as _validate_config

DEFAULT_CONFIG = "benchmarks/exact_match_classification.yaml"
RESULTS_DIR = "results"


def load_config(filepath):
    """Load a benchmark YAML, exiting with a clear message on failure."""
    try:
        # Binary: YAML detects UTF-8/16 itself and reports bad bytes as a YAMLError.
        with open(filepath, "rb") as stream:
            return yaml.safe_load(stream)
    except FileNotFoundError:
        sys.exit(
            f"Config not found: {filepath}\n"
            "Create one with `evalblink init` or pass a path: evalblink run <config.yaml>"
        )
    except (yaml.YAMLError, ValueError) as exc:  # ValueError: a 5000-digit int
        sys.exit(f"Invalid YAML in {filepath}: {exc}")
    except RecursionError:
        sys.exit(f"Invalid YAML in {filepath}: nested too deeply")
    except OSError as exc:  # a directory, no permission, ...
        sys.exit(f"Cannot read config {filepath}: {exc.strerror or exc}")


def _fetch_models(**kwargs):
    """The OpenRouter pricing catalog, exiting cleanly if it can't be fetched."""
    try:
        with httpx.Client() as client:
            return openrouter.fetch_models(client, **kwargs)
    except (RuntimeError, httpx.HTTPError) as exc:
        sys.exit(f"Could not fetch the OpenRouter model catalog: {exc}")


def cmd_run(args):
    """Run a benchmark, write the report, and exit on the CI quality gate."""
    config = load_config(args.config)

    errors, warnings = _validate_config(config)
    console = reporter.make_console()
    _print_findings(console, errors, warnings)
    if errors:
        sys.exit(1)

    if args.dry_run:
        # Estimate cost from the pricing catalog only — no completion calls.
        models_meta = _fetch_models()
        est = estimate.estimate(config, models_meta)
        reporter.render_estimate(est)
        sys.exit(1 if est["over_budget"] else 0)

    # Fail fast: without a key every candidate call would come back as an error.
    if not (os.getenv("OPENROUTER_API_KEY") or "").strip():
        console.print(
            "✗  OPENROUTER_API_KEY is not set — export it or add it to"
            " a .env file in this directory. `--dry-run` works without it.",
            style="red",
        )
        sys.exit(1)

    # Check before paying for any call: results are written at the very end.
    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)
    except OSError as exc:
        sys.exit(f"Cannot create {RESULTS_DIR}/: {exc.strerror or exc}")
    if not os.access(RESULTS_DIR, os.W_OK | os.X_OK):
        sys.exit(f"Cannot write to {RESULTS_DIR}/ — results would be lost.")

    use_cache = not args.no_cache
    results, timestamp = runner.run(config, verbose=args.verbose, use_cache=use_cache)
    try:
        result = reporter.write(config, results, timestamp)
    except (OSError, ValueError) as exc:
        sys.exit(f"Could not write results: {exc}")

    # Pipeline errors (api_error / judge errors) are excluded from scores; say so
    # loudly, and fail outright when nothing at all could be scored.
    errors = sum(r["errors"] for r in results)
    total = sum(r["total"] for r in results)
    if errors:
        console.print(
            f"\n⚠  {errors} of {total} case(s) hit a pipeline error"
            " (API / judge failure) and are excluded from scores.",
            style="bold red",
        )
    if errors and errors == total:
        print("\nFAIL: no case could be scored — every call hit a pipeline error.")
        sys.exit(1)

    # Top-level CI gate (0-100 %); separate from the per-case scorer thresholds
    # under `evaluation:`.
    threshold = config.get("quality_threshold")
    if threshold is None:
        print("\nGate: no quality_threshold set — not enforcing pass/fail.")
        sys.exit(0)
    if result["passed"]:
        print(f"\nPASS: best score meets quality_threshold ({threshold}).")
        sys.exit(0)
    if errors:
        print(
            f"\nFAIL: best score is below quality_threshold ({threshold}) once the"
            f" {errors} case(s) that could not be scored count as failures."
        )
        sys.exit(1)
    print(f"\nFAIL: best score is below quality_threshold ({threshold}).")
    sys.exit(1)


def cmd_compare(args):
    """Print the quality/cost delta between two persisted run records."""
    try:
        record_a = compare.load_record(args.file_a)
        record_b = compare.load_record(args.file_b)
    except compare.RecordError as exc:
        sys.exit(str(exc))
    if args.detailed:
        reporter.render_detailed(compare.detailed_diff(record_a, record_b))
    else:
        reporter.render_comparison(compare.diff(record_a, record_b))
    sys.exit(0)


def cmd_report(args):
    """Regenerate the Markdown report from an existing JSON result file."""
    try:
        record = compare.load_record(args.file)
    except compare.RecordError as exc:
        sys.exit(str(exc))
    try:
        reporter.write_from_record(record)
    except (OSError, ValueError) as exc:  # read-only .md, NUL / too-long run_id
        sys.exit(f"Could not write the report: {exc}")
    sys.exit(0)


def cmd_history(args):
    """List all past runs in the results/ directory."""
    if not os.path.isdir(RESULTS_DIR):
        print("No results directory found. Run a benchmark first.")
        sys.exit(0)

    try:
        fnames = sorted(os.listdir(RESULTS_DIR))
    except OSError as exc:
        sys.exit(f"Cannot read {RESULTS_DIR}/: {exc.strerror or exc}")
    records = []
    for fname in fnames:
        if not fname.endswith(".json"):
            continue
        try:
            data = compare.load_record(os.path.join(RESULTS_DIR, fname))
        except compare.RecordError as exc:
            print(f"⚠  Skipping {exc}")
            continue
        # Recomputed from results, so every record reads the same way: best
        # combo score, and the run-total pipeline error count.
        insights = summarize(data["results"], None) or {}
        records.append(
            {
                "run_id": data.get("run_id") or fname,
                "benchmark": data["benchmark"],
                "timestamp": data["timestamp"],
                "best_score": (insights.get("best_quality") or {}).get("score"),
                "total_cost": sum(r["total_cost"] for r in data["results"]),
                "errors": insights.get("errors", 0),
            }
        )

    if not records:
        print("No run records found in results/.")
        sys.exit(0)

    records.sort(key=lambda r: r["timestamp"], reverse=True)

    console = reporter.make_console()
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
    """Parse a context-length string like '100k', '8K', or '131072' to an int.

    An ``argparse`` type: a bad value is a clean usage error (exit 2).
    """
    v = str(value).strip().lower()
    scale = {"k": 1_000, "m": 1_000_000}.get(v[-1:], 1)
    try:
        return int(float(v[:-1] if scale > 1 else v) * scale)
    except (ValueError, OverflowError):
        raise argparse.ArgumentTypeError(
            f"invalid value {value!r} — use e.g. '100k' or '131072'"
        ) from None


def _fmt_ctx(ctx) -> str:
    return f"{ctx // 1000}k" if ctx else "—"


def _fmt_price(price) -> str:
    """Per-1M-token price; ``None`` is unknown pricing in the catalog."""
    if price is None:
        return "?"
    return f"{price * 1_000_000:.3f}" if price else "free"


def cmd_models(args):
    """List available OpenRouter models and their pricing."""
    models_meta = _fetch_models(use_cache=not args.no_cache)

    rows = sorted(models_meta.items())

    if args.provider:
        prefix = args.provider.rstrip("/") + "/"
        rows = [(k, v) for k, v in rows if k.startswith(prefix)]

    if args.free:
        rows = [(k, v) for k, v in rows if v["prompt"] == 0 and v["completion"] == 0]

    if args.min_context:
        rows = [
            (k, v)
            for k, v in rows
            if (v.get("context_length") or 0) >= args.min_context
        ]

    if not rows:
        print("No models match the given filters.")
        sys.exit(0)

    console = reporter.make_console()
    table = Table(title=f"OpenRouter models ({len(rows)} shown)")
    table.add_column("Model ID")
    table.add_column("Context", justify="right")
    table.add_column("Prompt $/1M", justify="right")
    table.add_column("Completion $/1M", justify="right")
    for model_id, meta in rows:
        table.add_row(
            model_id,
            _fmt_ctx(meta.get("context_length")),
            _fmt_price(meta["prompt"]),
            _fmt_price(meta["completion"]),
        )
    console.print(table)
    sys.exit(0)


def cmd_cache_stats(args):
    """Print cache entry count and total size."""
    try:
        s = cache.stats()
    except OSError as exc:
        sys.exit(f"Cannot read the cache: {exc.strerror or exc}")
    size_kb = s["size_bytes"] / 1024
    print(f"Cache entries : {s['entries']}")
    print(f"Cache size    : {size_kb:.1f} KB")
    sys.exit(0)


def cmd_cache_clear(args):
    """Delete all cached responses."""
    try:
        if not args.yes:
            s = cache.stats()
            print(f"{s['entries']} cache entries. Pass --yes to confirm deletion.")
            sys.exit(0)
        removed = cache.clear()
    except OSError as exc:
        sys.exit(f"Cannot clear the cache: {exc.strerror or exc}")
    print(f"Cleared {removed} cache entries.")
    sys.exit(0)


def _ask(console, question, parse, hint, **kwargs):
    """``Prompt.ask`` until ``parse(answer)`` accepts it (raises ``ValueError``
    to reject), so every value ``init`` writes is one ``validate`` accepts."""
    while True:
        try:
            return parse(Prompt.ask(question, **kwargs))
        except ValueError:
            console.print(hint, style="yellow")


def _name(raw):
    if not raw.strip():
        raise ValueError(raw)
    return raw.strip()


def _model_ids(raw):
    """Comma-separated model IDs, deduplicated in order; no blanks or spaces."""
    ids = list(dict.fromkeys(m.strip() for m in raw.split(",") if m.strip()))
    if not ids or any(len(m.split()) != 1 for m in ids):
        raise ValueError(raw)
    return ids


def _free_suggestions(models_meta):
    """Up to five free text models (text in, text out), by ID. A model whose
    modalities the catalog doesn't give is kept."""
    return sorted(
        (k, v)
        for k, v in models_meta.items()
        if v.get("prompt") == 0
        and v.get("completion") == 0
        and v.get("text") is not False
    )[:5]


def cmd_init(args):
    """Write a commented starter benchmark: three questions, then edit the file."""
    console = reporter.make_console()

    try:
        with httpx.Client() as client:
            models_meta = openrouter.fetch_models(client, use_cache=True)
    except (RuntimeError, httpx.HTTPError) as exc:
        console.print(
            f"Could not fetch the OpenRouter model catalog ({exc});"
            " continuing without suggestions.",
            style="yellow",
        )
        models_meta = {}
    suggestions = _free_suggestions(models_meta)
    default_model = starter.DEFAULT_MODEL
    if models_meta and default_model not in models_meta and suggestions:
        default_model = suggestions[0][0]  # the shipped default was retired

    if args.yes:
        name, mode, models = "My benchmark", "exact_match", [default_model]
    else:
        console.print("\nevalblink init — write a starter benchmark\n", style="bold")
        name = _ask(
            console,
            "Benchmark name",
            _name,
            "A name is required.",
            default="My benchmark",
        )
        console.print(
            "exact_match: the reply must equal a label. "
            "llm_judge: a judge model scores free text against your criteria."
        )
        mode = Prompt.ask(
            "Evaluation mode", choices=list(starter.MODES), default="exact_match"
        )
        if suggestions:
            console.print("\nSome free OpenRouter text models:")
            table = Table(show_header=True, header_style="bold", box=None)
            table.add_column("Model ID", style="cyan")
            table.add_column("Context", justify="right")
            for mid, meta in suggestions:
                table.add_row(mid, _fmt_ctx(meta.get("context_length")))
            console.print(table)
            console.print("`evalblink models --free` lists them all.")
        paid = models_meta.get(starter.PAID_MODEL)
        if paid:
            console.print(
                f"Small paid option, no free-tier rate limits: {starter.PAID_MODEL}"
                f" (${_fmt_price(paid['prompt'])} in / ${_fmt_price(paid['completion'])}"
                " out per 1M tokens)"
            )
        models = _ask(
            console,
            "Models to compare (comma-separated)",
            _model_ids,
            "Enter one or more model IDs separated by commas, e.g. a/b, c/d",
            default=default_model,
        )

    written = models + ([starter.JUDGE_MODEL] if mode == "llm_judge" else [])
    for m in written if models_meta else []:
        if m not in models_meta:
            console.print(
                f"⚠  {m} is not in the OpenRouter catalog — check the ID.",
                style="yellow",
            )

    out_path = args.path or f"benchmarks/{reporter._slugify(name)}.yaml"
    if os.path.isdir(out_path):
        sys.exit(f"{out_path} is a directory — pass a file path.")
    if os.path.exists(out_path):
        if args.yes:
            sys.exit(f"{out_path} already exists — pass another path.")
        if not Confirm.ask(
            f"{escape(out_path)} already exists — overwrite?", default=False
        ):
            sys.exit(0)

    text = starter.render(name, models, mode)
    # Last line of defence: never write a file `validate` would reject.
    errors, warnings = _validate_config(yaml.safe_load(text))
    _print_findings(console, errors, warnings)
    if errors:
        sys.exit(1)

    try:
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
    except (OSError, ValueError) as exc:  # ValueError: an unencodable path or name
        sys.exit(f"Cannot write {out_path}: {getattr(exc, 'strerror', None) or exc}")

    console.print(f"\n✓ Created {out_path}\n", style="green")
    console.print("Next: replace the example test cases with your own, then")
    console.print(f"  evalblink validate {out_path}")
    console.print(f"  evalblink run {out_path} --dry-run")
    console.print(f"  evalblink run {out_path}")
    sys.exit(0)


def _print_findings(console, errors, warnings) -> None:
    """Print validation warnings and errors (plain text: they quote user input)."""
    for w in warnings:
        console.print(f"⚠  {w}", style="yellow")
    for e in errors:
        console.print(f"✗  {e}", style="red")
    if errors:
        console.print(
            f"\nConfig has {len(errors)} error(s) — fix before running.", style="red"
        )


def cmd_validate(args):
    """Lint a benchmark YAML — errors and warnings, no API calls."""
    config = load_config(args.benchmark)
    errors, warnings = _validate_config(config)
    console = reporter.make_console()
    _print_findings(console, errors, warnings)
    if errors:
        sys.exit(1)
    console.print("✓  Config is valid", style="green")
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
  quality_threshold             CI gate 0–100; exit 1 if the best score (unscored
                                cases counted as failures) is below
  concurrency                   parallel requests, 1–100 (default: 5)
  evaluation.judge_model        required for llm_judge
  evaluation.judge_threshold    default: 0.70
  evaluation.judge_max_tokens   judge output budget (default: 4096)
  evaluation.quality_threshold  per-case pass threshold 0–1, required for weighted_match
  evaluation.variables          use_case/percent/order weights, required for weighted_match
  test_cases[].id               required
  test_cases[].evaluation       exact_match | llm_judge | weighted_match  (required)
  test_cases[].expected_output  required for exact_match and weighted_match
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
  Model / Prompt  the combo being compared
  Quality Δ       pass-rate change B − A: ↑ / ↓, or stable within ±0.05 pp
  Cost Δ          total cost change per combo (models + judge)
  Notes           "only in A" / "only in B" when a combo is missing from one run

--detailed also shows per-combo case transitions and a global summary:
  regressed   pass → fail
  improved    fail → pass
  new error   scored → None  (pipeline error, not counted as a regression)
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
    parser.add_argument("--version", action="version", version=__version__)
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
        help="bypass the local cache and force fresh API calls (candidate and judge)",
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
        help="write a commented starter benchmark YAML (3 questions)",
        description="Ask a name, an evaluation mode and the models, then write a "
        "complete, commented benchmark that validates and runs as-is. Edit its "
        "example test cases in your editor.",
    )
    init_p.add_argument(
        "path",
        nargs="?",
        help="output file (default: benchmarks/<name>.yaml)",
    )
    init_p.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="no questions: write the exact_match starter with the default model",
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
        type=_parse_context,
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
        "benchmark",
        nargs="?",
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
        help="actually delete (without it, only the entry count is printed)",
    )
    clear_p.set_defaults(func=cmd_cache_clear)

    return parser


def main(argv=None):
    # Names, ids and model output may carry lone surrogates (a "\\ud800" YAML or
    # JSON escape), and a cp1252 pipe can't encode emoji; never let printing one
    # abort a command.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):  # not a TextIOWrapper (redirected)
            pass
    # The .env next to the user's benchmarks — not the installed package's dir.
    # utf-8-sig: a Windows editor's BOM would otherwise become part of the key name.
    try:
        load_dotenv(find_dotenv(usecwd=True), encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        print(f"⚠️  Ignoring .env: {exc} (save it as UTF-8).", file=sys.stderr)
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except KeyboardInterrupt:
        sys.stdout.flush()  # so the message below comes last
        print("\nInterrupted — no results written.", file=sys.stderr)
        sys.stderr.flush()
        # Skip joining worker threads still blocked in an HTTP read (up to the
        # 120 s timeout); there is nothing left to write.
        os._exit(130)
    except EOFError:  # `init` with stdin closed
        sys.exit("\nNo input — aborted.")


if __name__ == "__main__":
    main()
