"""Benchmark LLM prompts × models on your own data via OpenRouter."""

from __future__ import annotations

import argparse
import math
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


def _default_config():
    """The config ``run``/``validate`` use without a path: the only YAML in
    ``benchmarks/`` (what ``init`` writes), else the repo's example config."""
    try:
        found = sorted(
            f
            for f in os.listdir("benchmarks")
            if f.endswith((".yaml", ".yml"))
            and not f.startswith(".")  # editor/OS junk such as macOS ._x.yaml
            and os.path.isfile(os.path.join("benchmarks", f))
        )
    except OSError:
        found = []
    if len(found) == 1:
        return os.path.join("benchmarks", found[0])
    if found and not os.path.exists(DEFAULT_CONFIG):
        sys.exit(
            "Several configs in benchmarks/ — pass the one to use:\n"
            + "\n".join(f"  benchmarks/{f}" for f in found)
        )
    return DEFAULT_CONFIG  # if missing, load_config suggests `evalblink init`


def _config_path(path):
    """``path``, or the default config announced on stderr."""
    if path:
        return path
    path = _default_config()
    print(f"Using {path}", file=sys.stderr)
    return path


def _is_text_model(meta) -> bool:
    """Text in, text-only out; kept when the catalog doesn't say."""
    return meta.get("text") is not False


def _fetch_models(**kwargs):
    """The OpenRouter pricing catalog, exiting cleanly if it can't be fetched."""
    try:
        with httpx.Client() as client:
            return openrouter.fetch_models(client, **kwargs)
    except (RuntimeError, httpx.HTTPError) as exc:
        sys.exit(f"Could not fetch the OpenRouter model catalog: {exc}")


def _filtered(config, args):
    """A valid ``config`` narrowed by ``run --model/--prompt/--tag/--limit``.

    Unknown names exit with the available ones. The filters are recorded as
    ``_filters`` (a key the validator leaves alone) for the run record.
    """
    config = dict(config)

    def keep(field, wanted, key, what):
        items = config.get(field)
        if not wanted or not isinstance(items, list):
            return
        have = [key(x) for x in items]
        missing = [w for w in wanted if w not in have]
        if missing:
            sys.exit(
                f"No {what} {', '.join(map(repr, missing))} in the config."
                f" Available: {', '.join(map(str, have))}"
            )
        config[field] = [x for x, k in zip(items, have) if k in wanted]

    def item_id(x):
        return x.get("id") if isinstance(x, dict) else x

    keep("models", args.model, lambda m: m, "model")
    keep("prompts", args.prompt, item_id, "prompt")
    cases = config.get("test_cases")
    if isinstance(cases, list) and (args.tag or args.limit):
        if args.tag:
            cases = [
                tc
                for tc in cases
                if isinstance(tc, dict) and set(tc.get("tags") or []) & set(args.tag)
            ]
        cases = cases[: args.limit] if args.limit else cases
        if args.tag and not cases:
            sys.exit(f"No test case has tag {' or '.join(map(repr, args.tag))}.")
        config["test_cases"] = cases

    gate = config.get("gate")
    if isinstance(gate, dict) and (
        (args.model and gate.get("model") not in (None, *args.model))
        or (args.prompt and gate.get("prompt") not in (None, *args.prompt))
    ):
        print("Note: gate ignored — its combo is filtered out.", file=sys.stderr)
        config.pop("gate")
    filters = {
        k: v
        for k, v in (
            # Sorted: `--tag a --tag b` and `--tag b --tag a` are the same slice.
            ("model", sorted(args.model or [])),
            ("prompt", sorted(args.prompt or [])),
            ("tag", sorted(args.tag or [])),
            ("limit", args.limit),
        )
        if v
    }
    if filters:
        config["_filters"] = filters
        sizes = [len(config.get(f) or []) for f in ("models", "prompts", "test_cases")]
        print(
            "Filtered: {} model(s) × {} prompt(s) × {} case(s)".format(*sizes),
            file=sys.stderr,
        )
    return config


def cmd_run(args):
    """Run a benchmark, write the report, and exit on the CI quality gate."""
    config = load_config(_config_path(args.config))

    # The whole config first, so a filter never hides one of its mistakes.
    errors, warnings = _validate_config(config)
    console = reporter.make_console()
    _print_findings(console, errors, warnings)
    if errors:
        sys.exit(1)
    config = _filtered(config, args)

    if args.dry_run:
        # Estimate cost from the pricing catalog only — no completion calls.
        models_meta = _fetch_models()
        est = estimate.estimate(config, models_meta)
        reporter.render_estimate(est)
        sys.exit(1 if est["over_budget"] or est["unknown_models"] else 0)

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
    gate = config.get("gate") or {}
    gated = " / ".join(gate[k] for k in ("model", "prompt") if gate.get(k))
    subject = f"{gated} score" if gated else "best score"
    if result["passed"]:
        print(f"\nPASS: {subject} meets quality_threshold ({threshold}).")
        sys.exit(0)
    if errors:
        print(
            f"\nFAIL: {subject} is below quality_threshold ({threshold}) once the"
            f" {errors} case(s) that could not be scored count as failures."
        )
        sys.exit(1)
    print(f"\nFAIL: {subject} is below quality_threshold ({threshold}).")
    sys.exit(1)


def _run_records():
    """Readable run records in ``results/``, newest first, as (path, record)."""
    try:
        names = os.listdir(RESULTS_DIR)
    except OSError:
        return []
    found = []
    for name in names:
        if name.endswith(".json"):
            path = os.path.join(RESULTS_DIR, name)
            try:
                found.append((path, compare.load_record(path)))
            except compare.RecordError:
                continue

    # Same-second reruns share a timestamp; the reporter numbers them
    # `<timestamp>_<slug>-2`, `-3`… (see reporter.write), which orders them
    # however the files were copied. The file's mtime is the last resort.
    def rerun(record):
        base = f"{record['timestamp']}_{reporter._slugify(record['benchmark'])}"
        suffix = str(record.get("run_id") or "").removeprefix(base)
        return int(suffix[1:]) if suffix[:1] == "-" and suffix[1:].isdigit() else 1

    found.sort(
        key=lambda pr: (
            pr[1]["timestamp"],
            rerun(pr[1]),
            os.path.getmtime(pr[0]),
            pr[0],
        ),
        reverse=True,
    )
    return found


def _record_path(arg):
    """A results file from a path, or from a run ID (or a unique prefix of one)."""
    if os.path.exists(arg) or os.sep in arg or arg.endswith(".json"):
        return arg
    try:
        names = sorted(os.listdir(RESULTS_DIR))
    except OSError:
        names = []
    if f"{arg}.json" in names:
        return os.path.join(RESULTS_DIR, f"{arg}.json")
    matches = [n for n in names if n.endswith(".json") and n.startswith(arg)]
    if len(matches) == 1:
        return os.path.join(RESULTS_DIR, matches[0])
    if matches:
        sys.exit(
            f"Run ID {arg!r} matches several runs:\n"
            + "\n".join(f"  {n[:-5]}" for n in matches)
        )
    return arg  # load_record reports it as not found


def _load(path):
    try:
        return compare.load_record(path)
    except compare.RecordError as exc:
        sys.exit(str(exc))


def cmd_compare(args):
    """Print the quality/cost delta between two persisted run records."""
    if args.file_b is not None:
        record_a = _load(_record_path(args.file_a))
        record_b = _load(_record_path(args.file_b))
    else:
        # One run: it is the baseline for the latest run of its benchmark.
        # None: the latest run against the previous one. Either way, only runs
        # of the same benchmark with the same `run` filters are paired.
        runs = _run_records()
        if args.file_a is not None:
            path_a = _record_path(args.file_a)
            record_a = _load(path_a)
            # Not the baseline itself, even when it is a copy of a results file.
            pool = [
                pr
                for pr in runs
                if os.path.abspath(pr[0]) != os.path.abspath(path_a)
                and pr[1].get("run_id") != record_a.get("run_id")
            ]
        else:
            if not runs:
                sys.exit(f"No run records in {RESULTS_DIR}/ — run a benchmark first.")
            path_a, record_a = None, None
            pool = runs
        key = record_a or (pool[0][1] if pool else {})
        same = [
            pr
            for pr in pool
            if pr[1]["benchmark"] == key.get("benchmark")
            and pr[1].get("filters") == key.get("filters")
        ]
        if record_a is None and len(same) >= 2:
            (path_b, record_b), (path_a, record_a) = same[0], same[1]
        elif record_a is not None and same:
            path_b, record_b = same[0]
        else:
            what = f"{key.get('benchmark')!r}" if key else "it"
            message = f"No other run of {what} in {RESULTS_DIR}/ to compare with."
            if args.fail_on_regression is not None:
                print(f"{message} Nothing to check.")
                sys.exit(0)
            sys.exit(message)
        print(f"Comparing {path_a} → {path_b}", file=sys.stderr)
    if args.detailed:
        reporter.render_detailed(compare.detailed_diff(record_a, record_b))
    else:
        reporter.render_comparison(compare.diff(record_a, record_b))
    if args.fail_on_regression is None:
        sys.exit(0)
    result = compare.diff(record_a, record_b)
    if not compare.compared(result):
        # Fail closed: a renamed model or an edited test set must not turn the
        # CI check into a silent pass.
        print(
            "FAIL: nothing to compare — the runs share no model × prompt combo"
            " with the same test cases. Refresh the baseline."
        )
        sys.exit(1)
    dropped = compare.regressions(result, args.fail_on_regression)
    for row in dropped:
        print(
            f"REGRESSION: {row['model']} / {row['prompt_id']} pass rate"
            f" {row['gate_delta']:+.1f} points (unscored cases count as failures)"
        )
    if dropped:
        sys.exit(1)
    print(f"No regression beyond {args.fail_on_regression:g} point(s).")
    sys.exit(0)


def cmd_report(args):
    """Regenerate the Markdown report from an existing JSON result file."""
    if args.file is None:
        runs = _run_records()
        if not runs:
            sys.exit(f"No run records in {RESULTS_DIR}/ — run a benchmark first.")
        path, record = runs[0]
        print(f"Using {path}", file=sys.stderr)
    else:
        record = _load(_record_path(args.file))
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
    table.add_column("Run ID", overflow="fold")
    table.add_column("Benchmark", overflow="fold")
    table.add_column("Timestamp", overflow="fold")
    table.add_column("Best Score", justify="right", overflow="fold")
    table.add_column("Total Cost", justify="right", overflow="fold")
    table.add_column("Errors", justify="right", overflow="fold")
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


def _positive_int(value) -> int:
    """An ``argparse`` type: a whole number ≥ 1."""
    try:
        number = int(value)
    except ValueError:
        number = 0
    if number < 1:
        raise argparse.ArgumentTypeError(f"invalid value {value!r} — use 1 or more")
    return number


def _points(value) -> float:
    """An ``argparse`` type: a finite number of points ≥ 0."""
    try:
        number = float(value)
    except ValueError:
        number = math.nan
    if not (math.isfinite(number) and number >= 0):
        raise argparse.ArgumentTypeError(f"invalid value {value!r} — use 0 or more")
    return number


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

    if args.text:
        rows = [(k, v) for k, v in rows if _is_text_model(v)]

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
    table.add_column("Model ID", overflow="fold")
    table.add_column("Context", justify="right", overflow="fold")
    table.add_column("Prompt $/1M", justify="right", overflow="fold")
    table.add_column("Completion $/1M", justify="right", overflow="fold")
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
    print(f"Cached responses : {s['entries']}")
    print(f"Responses size   : {size_kb:.1f} KB")
    print(f"Model catalog    : {'cached (24h)' if s.get('catalog') else 'not cached'}")
    sys.exit(0)


def cmd_cache_clear(args):
    """Delete all cached responses."""
    try:
        if not args.yes:
            s = cache.stats()
            print(
                f"{_plural(s['entries'], 'cached response')}."
                " Pass --yes to delete them (--all also drops the model catalog)."
            )
            sys.exit(0)
        removed = cache.clear(include_catalog=args.all)
    except OSError as exc:
        sys.exit(f"Cannot clear the cache: {exc.strerror or exc}")
    print(f"Cleared {_plural(removed, 'cache file')}.")
    sys.exit(0)


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


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


def _free_suggestions(models_meta, first=None):
    """Up to five free text models, by ID with ``first`` (the default) on top."""
    return sorted(
        (
            (k, v)
            for k, v in models_meta.items()
            if v.get("prompt") == 0 and v.get("completion") == 0 and _is_text_model(v)
        ),
        key=lambda kv: (kv[0] != first, kv[0]),
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
    default_model = starter.DEFAULT_MODEL
    suggestions = _free_suggestions(models_meta, first=default_model)
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
            table.add_column("Model ID", style="cyan", overflow="fold")
            table.add_column("Context", justify="right", overflow="fold")
            for mid, meta in suggestions:
                table.add_row(mid, _fmt_ctx(meta.get("context_length")))
            console.print(table)
            console.print("`evalblink models --free` lists them all.")
        paid = models_meta.get(starter.PAID_MODEL)
        if paid:
            console.print(
                "Small paid option, no free-tier rate limits:\n"
                f"  {starter.PAID_MODEL}"
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

    # Absolute: relative paths are under the directory init ran in, not the install.
    console.print(
        f"\n✓ Created {os.path.abspath(out_path)}\n", style="green", soft_wrap=True
    )
    console.print(f"Next: open {out_path} and make it yours —")
    console.print("  the prompt is under `prompts:`, the examples under `test_cases:`.")
    console.print("Then:")
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
    config = load_config(_config_path(args.benchmark))
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
  evalblink run                                  # the only YAML in benchmarks/
  evalblink run benchmarks/classification.yaml
  evalblink run my_benchmark.yaml -m openai/gpt-4o-mini -p v2
  evalblink run my_benchmark.yaml --tag edge_case --limit 5
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
  Notes           "only in A" / "only in B" when a combo is missing from one run;
                  "pipeline errors 2 → 0" when its error count changed (yellow);
                  "different test cases" when the runs ran other cases (yellow)

--fail-on-regression POINTS (for CI) exits 1 when a shared combo's pass rate —
unscored cases counted as failures — drops by more than POINTS, or when no
combo could be compared at all (refresh the baseline).

--detailed also shows per-combo case transitions and a global summary:
  regressed   pass → fail
  improved    fail → pass
  new error   scored → None  (pipeline error, not counted as a regression)
  recovered   None → scored

EXAMPLES
  evalblink compare                                   # two latest runs of a benchmark
  evalblink compare results/run_a.json results/run_b.json --detailed
  evalblink compare 2026-10-30_0814 2026-10-30_0815   # run ID prefixes
  evalblink compare baseline.json --fail-on-regression 2   # CI: latest run vs a kept baseline
"""


EXAMPLES_INIT = """\
EXAMPLES
  evalblink init                         # 3 questions, Enter = defaults
  evalblink init evals/support.yaml      # choose the file
  evalblink init --yes                   # no questions
"""


_ROOT_EPILOG = """\
QUICKSTART
  evalblink init              write a starter benchmark (3 questions, Enter = defaults)
  evalblink run --dry-run     estimate its cost (no API key needed)
  evalblink run               run it, then open results/<run>.md

ENVIRONMENT
  OPENROUTER_API_KEY          needed by `run`; get one at https://openrouter.ai/keys.
                              Also read from a .env file in this directory or a parent.

Run `evalblink <command> -h` for a command's options and examples.
"""


def build_parser():
    parser = argparse.ArgumentParser(
        prog="evalblink",
        description=__doc__,
        epilog=_ROOT_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=__version__)
    # Not required: a bare `evalblink` prints the help and the quickstart.
    subparsers = parser.add_subparsers(dest="command")

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
        help="path to a benchmark YAML (default: the only YAML in benchmarks/,"
        f" else {DEFAULT_CONFIG})",
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
        "-m",
        "--model",
        action="append",
        metavar="ID",
        help="only this model (repeatable)",
    )
    run_p.add_argument(
        "-p",
        "--prompt",
        action="append",
        metavar="ID",
        help="only this prompt id (repeatable)",
    )
    run_p.add_argument(
        "-t",
        "--tag",
        action="append",
        metavar="TAG",
        help="only test cases with this tag (repeatable: any of them)",
    )
    run_p.add_argument(
        "-n",
        "--limit",
        type=_positive_int,
        metavar="N",
        help="only the first N test cases (after --tag)",
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
        nargs="?",
        metavar="RUN_A",
        help="baseline run: a results JSON path or a run ID (prefix). Alone, it"
        " is compared with the latest run of its benchmark; with no runs, the"
        " two latest runs are compared",
    )
    compare_p.add_argument(
        "file_b",
        nargs="?",
        metavar="RUN_B",
        help="candidate run: a results JSON path or a run ID (prefix)",
    )
    compare_p.add_argument(
        "--fail-on-regression",
        type=_points,
        metavar="POINTS",
        help="exit 1 if a combo's pass rate (unscored cases counted as failures)"
        " drops by more than POINTS; 0 = any drop — for CI",
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
        epilog="""EXAMPLES
  evalblink report                       # the latest run
  evalblink report 2026-10-30_0814       # a run ID prefix
  evalblink report results/<run>.json
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    report_p.add_argument(
        "file",
        nargs="?",
        metavar="RUN",
        help="a results JSON path or a run ID (prefix); default: the latest run",
    )
    report_p.set_defaults(func=cmd_report)

    # --- history ---
    history_p = subparsers.add_parser(
        "history",
        help="list all past runs in the results/ directory",
        epilog="""EXAMPLES
  evalblink history
  evalblink report <run-id>    # regenerate one run's Markdown
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    history_p.set_defaults(func=cmd_history)

    # --- init ---
    init_p = subparsers.add_parser(
        "init",
        help="write a commented starter benchmark YAML (3 questions)",
        description="Ask a name, an evaluation mode and the models, then write a "
        "complete, commented benchmark that validates and runs as-is. Edit its "
        "example test cases in your editor.",
        epilog=EXAMPLES_INIT,
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        epilog="""EXAMPLES
  evalblink models --free --text         # free text models
  evalblink models --provider anthropic
  evalblink models --min-context 100k
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        "--text",
        action="store_true",
        help="show only text models (text in, text-only out)",
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
        epilog="""EXAMPLES
  evalblink validate                     # the only YAML in benchmarks/
  evalblink validate benchmarks/my-benchmark.yaml
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    validate_p.add_argument(
        "benchmark",
        nargs="?",
        metavar="FILE",
        help="path to a benchmark YAML (default: the only YAML in benchmarks/,"
        f" else {DEFAULT_CONFIG})",
    )
    validate_p.set_defaults(func=cmd_validate)

    # --- cache ---
    cache_p = subparsers.add_parser(
        "cache",
        help="manage the local response cache",
        epilog="""EXAMPLES
  evalblink cache stats
  evalblink cache clear --yes            # keep the model catalog
  evalblink cache clear --yes --all
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
    clear_p.add_argument(
        "--all",
        action="store_true",
        help="also delete the cached model catalog (refetched on next use)",
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
    if args.command is None:
        parser.print_help()
        sys.exit(0)
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
