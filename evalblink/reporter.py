"""All output for a benchmark run: terminal table, JSON, Markdown.

This module makes no decisions — it only formats and writes the raw results it
is handed. ``write`` is the single entry point.
"""

from __future__ import annotations

import json
import os
import re

from rich.console import Console
from rich.table import Table

from . import analysis
from .compare import CHANGED_TRANSITIONS
from .schemas import SCHEMA_VERSION

RESULTS_DIR = "results"


def _combo_label(combo) -> str:
    return f"{combo['model']} / {combo['prompt_id']}"


def _slugify(name: str) -> str:
    """Filename-safe slug: lowercase, non-alphanumeric runs → single hyphen."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _build_data(config, results, timestamp) -> dict:
    """The canonical run record — the exact shape persisted as JSON."""
    benchmark_name = config["name"]
    run_id = f"{timestamp}_{_slugify(benchmark_name)}"
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "benchmark": config["name"],
        "judge_model": config.get("evaluation", {}).get("judge_model"),
        "temperature": config["inference"]["temperature"]
        if "temperature" in config["inference"]
        else 0,
        "max_tokens": config["inference"]["max_tokens"]
        if "max_tokens" in config["inference"]
        else 4096,
        "quality_threshold": config.get("evaluation", {}).get(
            "quality_threshold", config.get("evaluation", {}).get("judge_threshold")
        ),
        "timestamp": timestamp,
        "results": results,
    }


def _write_json(data) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    file_path = f"{RESULTS_DIR}/{data['run_id']}.json"
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    return file_path


def _write_markdown(data) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    file_path = f"{RESULTS_DIR}/{data['run_id']}.md"
    lines = [
        f"# {data['benchmark']}",
        "",
        f"- **Run ID:** {data['run_id']}",
        f"- **Timestamp:** {data['timestamp']}",
        f"- **Judge model:** {data['judge_model']}",
        f"- **Temperature:** {data['temperature']}",
        f"- **Max tokens:** {data['max_tokens']}",
        f"- **Quality threshold:** {data['quality_threshold']}",
        "",
        "## Summary",
        "",
        "| Model | Prompt | Score | Pass/Scored | Errors | Prompt tok | Completion tok | Cost |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for run in data["results"]:
        scored = run.get("scored", run["total"])
        errors = run.get("errors", 0)
        lines.append(
            f"| {run['model']} | {run['prompt_id']} | {run['score']:.1f}% | "
            f"{run['success']}/{scored} | {errors} | {run['total_prompt_tokens']} | "
            f"{run['total_completion_tokens']} | ${run['total_cost']:.6f} |"
        )

    for run in data["results"]:
        lines += [
            "",
            f"## {run['model']} — prompt `{run['prompt_id']}`",
            "",
            "| Test case | Evaluation | Match | Score | Cost |",
            "| --- | --- | --- | --- | --- |",
        ]
        for tc in run["test_cases"]:
            score = "" if tc["match_score"] is None else f"{tc['match_score']:.4f}"
            lines.append(
                f"| {tc['id']} | {tc['evaluation']} | {tc['match_result']} | "
                f"{score} | ${tc['total_cost']:.6f} |"
            )

    summary = data.get("insights")
    if summary:
        top = summary["best_quality"]
        value = summary["best_value"]
        lines += [
            "",
            f"## Quality by tag — best combo ({_combo_label(top)})",
            "",
            "| Tag | Cases | Quality | Errors | |",
            "| --- | --- | --- | --- | --- |",
        ]
        for row in summary["tag_breakdown"]:
            warn = "⚠️" if row["warn"] else ""
            quality = "—" if row["quality"] is None else f"{row['quality']:.0f}%"
            lines.append(
                f"| {row['tag']} | {row['cases']} | {quality} | "
                f"{row['errors']} | {warn} |"
            )
        lines += [
            "",
            "## Recommendation",
            "",
            f"- **Best quality:** {_combo_label(top)} ({top['score']:.1f}%)",
            f"- **Best value:** {_combo_label(value)} "
            f"({value['score']:.1f}% · ${value['total_cost']:.4f})",
        ]
        for tag in summary["warnings"]:
            lines.append(f"- ⚠️ **Warning:** `{tag}` quality is below 70%.")
        if summary.get("errors"):
            lines.append(
                f"- ℹ️ **{summary['errors']} pipeline error(s)** excluded from scoring."
            )

    with open(file_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return file_path


def _print_table(data, console: Console) -> None:
    table = Table(title=f"{data['benchmark']} — {data['run_id']}")
    table.add_column("Model")
    table.add_column("Prompt")
    table.add_column("Score", justify="right")
    table.add_column("Pass/Scored", justify="right")
    table.add_column("Errors", justify="right")
    table.add_column("Prompt tok", justify="right")
    table.add_column("Completion tok", justify="right")
    table.add_column("Cost", justify="right")
    for run in data["results"]:
        scored = run.get("scored", run["total"])
        errors = run.get("errors", 0)
        table.add_row(
            run["model"],
            run["prompt_id"],
            f"{run['score']:.1f}%",
            f"{run['success']}/{scored}",
            str(errors),
            str(run["total_prompt_tokens"]),
            str(run["total_completion_tokens"]),
            f"${run['total_cost']:.6f}",
        )
    console.print(table)


def _print_insights(summary, console: Console) -> None:
    top = summary["best_quality"]
    value = summary["best_value"]

    tags = Table(title=f"QUALITY BY TAG — best combo ({_combo_label(top)})")
    tags.add_column("Tag")
    tags.add_column("Cases", justify="right")
    tags.add_column("Quality", justify="right")
    tags.add_column("Errors", justify="right")
    tags.add_column("", justify="left")
    for row in summary["tag_breakdown"]:
        warn = "⚠️" if row["warn"] else ""
        style = "yellow" if row["warn"] else None
        quality = "—" if row["quality"] is None else f"{row['quality']:.0f}%"
        tags.add_row(
            row["tag"],
            str(row["cases"]),
            quality,
            str(row["errors"]),
            warn,
            style=style,
        )
    console.print(tags)

    console.print("\n[bold]RECOMMENDATION[/bold]")
    console.print(f"Best quality : {_combo_label(top)} ({top['score']:.1f}%)")
    console.print(
        f"Best value   : {_combo_label(value)} "
        f"({value['score']:.1f}% · ${value['total_cost']:.4f}/run)"
    )
    if summary.get("errors"):
        console.print(
            f"[dim]{summary['errors']} pipeline error(s) excluded from scoring.[/dim]"
        )
    for tag in summary["warnings"]:
        console.print(f"[yellow]⚠️  Warning  : {tag} quality is below 70%.[/yellow]")


def write(config, results, timestamp) -> dict:
    """Render the terminal table + insights and write the JSON + Markdown files.

    Returns the written file paths and the CI/CD ``passed`` verdict.
    """
    data = _build_data(config, results, timestamp)
    # The CI gate is a top-level percentage (0-100), distinct from the per-case
    # `evaluation.quality_threshold`/`judge_threshold` (0-1) the scorers consume.
    summary = analysis.summarize(results, config.get("quality_threshold"))
    data["insights"] = summary
    json_path = _write_json(data)
    md_path = _write_markdown(data)
    console = Console()
    _print_table(data, console)
    if summary:
        _print_insights(summary, console)
    print(f"Results saved to {json_path}")
    print(f"Markdown report saved to {md_path}")
    passed = True if summary is None else summary["passed"]
    return {"json": json_path, "markdown": md_path, "passed": passed}


def write_from_record(record: dict) -> str:
    """Regenerate the Markdown report from a persisted JSON record.

    The record must be the dict previously written by ``write()`` — it already
    contains an ``insights`` key from ``analysis.summarize``. Old records that
    predate the ``run_id`` field get a derived fallback so the filename is stable.
    """
    if "run_id" not in record:
        record = {
            **record,
            "run_id": f"{record.get('timestamp', 'unknown')}_{_slugify(record.get('benchmark', 'run'))}",
        }
    md_path = _write_markdown(record)
    console = Console()
    _print_table(record, console)
    if record.get("insights"):
        _print_insights(record["insights"], console)
    print(f"Markdown report saved to {md_path}")
    return md_path


def render_estimate(est, console: Console | None = None) -> None:
    """Print the ``--dry-run`` cost estimate table, totals, and budget verdict.

    Formatting only — ``estimate.estimate`` produced the numbers. Costs are
    upper-bound estimates (completion tokens assume the configured max_tokens).
    """
    console = console or Console()
    table = Table(title="DRY RUN — estimated cost (no API calls)")
    table.add_column("Model")
    table.add_column("Prompt")
    table.add_column("Est prompt tok", justify="right")
    table.add_column("Est completion tok", justify="right")
    table.add_column("Est cost", justify="right")
    for row in est["rows"]:
        combo_cost = row["est_cost"] + row["est_judge_cost"]
        table.add_row(
            row["model"],
            row["prompt_id"],
            str(row["est_prompt_tokens"]),
            str(row["est_completion_tokens"]),
            f"${combo_cost:.6f}",
        )
    console.print(table)

    if est["judge_cost"]:
        console.print(f"Judge cost (llm_judge cases): ${est['judge_cost']:.6f}")
    console.print(f"[bold]Estimated total cost: ${est['total_cost']:.6f}[/bold]")
    console.print(
        "[dim]Estimate only — completion tokens assume max_tokens; "
        "prompt tokens ≈ chars/4.[/dim]"
    )

    for model in est["missing_pricing"]:
        console.print(
            f"[yellow]⚠️  No pricing for '{model}' — excluded from the estimate.[/yellow]"
        )

    budget = est["max_cost_usd"]
    if budget is not None:
        if est["over_budget"]:
            console.print(
                f"[red]✗ Over budget: estimate ${est['total_cost']:.6f} exceeds "
                f"max_cost_usd ${budget:.6f}.[/red]"
            )
        else:
            console.print(
                f"[green]✓ Within budget: estimate ${est['total_cost']:.6f} ≤ "
                f"max_cost_usd ${budget:.6f}.[/green]"
            )


# Cost deltas within this many dollars read as "stable" rather than a ±.
_COST_EPS = 1e-6


def _fmt_score_delta(delta) -> str:
    if delta is None:
        return "—"
    if abs(delta) < 0.05:
        return "stable"
    arrow = "↑" if delta > 0 else "↓"
    return f"{delta:+.1f}% {arrow}"


def _fmt_cost_delta(delta) -> str:
    if delta is None:
        return "—"
    if abs(delta) < _COST_EPS:
        return "stable"
    arrow = "↑" if delta > 0 else "↓"
    return f"{delta:+.4f} {arrow}"


def render_comparison(diff_result, console: Console | None = None) -> None:
    """Print the run-to-run delta table (quality + cost) for ``compare``.

    Terminal-only — mirrors ``_print_table``'s columns and styling. ``a_only`` /
    ``b_only`` combos are flagged in a Notes column; the verdict and any schema
    version note follow.
    """
    console = console or Console()
    table = Table(title=f"DELTA: {diff_result['run_a']} → {diff_result['run_b']}")
    table.add_column("Model")
    table.add_column("Prompt")
    table.add_column("Quality Δ", justify="right")
    table.add_column("Cost Δ", justify="right")
    table.add_column("Notes")
    note_for = {"a_only": "only in A", "b_only": "only in B", "both": ""}
    for row in diff_result["rows"]:
        delta = row["score_delta"]
        style = None
        if delta is not None:
            if delta > 0.05:
                style = "green"
            elif delta < -0.05:
                style = "red"
        table.add_row(
            row["model"],
            row["prompt_id"],
            _fmt_score_delta(delta),
            _fmt_cost_delta(row["cost_delta"]),
            note_for[row["present_in"]],
            style=style,
        )
    console.print(table)
    if diff_result.get("version_note"):
        console.print(f"[dim]{diff_result['version_note']}[/dim]")
    console.print(f"\n[bold]Verdict:[/bold] {diff_result['verdict']}")


# How each case transition is labelled and coloured in the detailed view.
_TRANSITION_STYLE = {
    "regressed": ("regressed", "red"),
    "improved": ("improved", "green"),
    "new_error": ("new error", "yellow"),
    "recovered": ("recovered", "green"),
    "added": ("only in B", "dim"),
    "removed": ("only in A", "dim"),
}


def _fmt_case_score_delta(delta) -> str:
    if delta is None:
        return "—"
    if abs(delta) < 1e-9:
        return "0"
    arrow = "↑" if delta > 0 else "↓"
    return f"{delta:+.2f} {arrow}"


def _render_case_combo(combo, console: Console) -> None:
    """One shared combo's changed test cases, with an unchanged-count footer."""
    changed = [r for r in combo["rows"] if r["transition"] in CHANGED_TRANSITIONS]
    label = f"{combo['model']} / {combo['prompt_id']}"
    if not changed:
        console.print(f"[dim]{label}: no case changes[/dim]")
        return
    table = Table(title=f"CASE CHANGES — {label}")
    table.add_column("Test case")
    table.add_column("Change")
    table.add_column("A → B")
    table.add_column("Score Δ", justify="right")
    for row in changed:
        text, style = _TRANSITION_STYLE[row["transition"]]
        a_to_b = f"{row['a_state'] or '—'} → {row['b_state'] or '—'}"
        table.add_row(
            row["id"],
            text,
            a_to_b,
            _fmt_case_score_delta(row["score_delta"]),
            style=style,
        )
    console.print(table)
    rollup = combo["rollup"]
    unchanged = rollup["unchanged"] + rollup["still_error"]
    console.print(f"[dim]{unchanged} unchanged[/dim]")


def render_detailed(detailed_result, console: Console | None = None) -> None:
    """Print the combo table, per-combo case changes, and the global summary."""
    console = console or Console()
    render_comparison(detailed_result, console)

    console.print("\n[bold]── TEST-LEVEL CHANGES ──[/bold]")
    if not detailed_result["case_combos"]:
        console.print("[dim]No combos shared between the two runs.[/dim]")
        return
    for combo in detailed_result["case_combos"]:
        console.print()
        _render_case_combo(combo, console)

    summary = detailed_result["summary"]
    console.print("\n[bold]GLOBAL SUMMARY[/bold]")
    console.print(
        f"{summary['improved']} improved · {summary['regressed']} regressed · "
        f"{summary['new_error']} new error(s) · {summary['recovered']} recovered"
    )
    console.print(f"[bold]Verdict:[/bold] {summary['verdict']}")

    if summary["worst_regressed"]:
        total_combos = len(detailed_result["case_combos"])
        worst = Table(title="WORST-REGRESSED CASES")
        worst.add_column("Test case")
        worst.add_column("Regressed in", justify="right")
        for item in summary["worst_regressed"]:
            worst.add_row(item["id"], f"{item['count']}/{total_combos} combos")
        console.print(worst)

    if summary["tag_net"]:
        tags = Table(title="PER-TAG CONCENTRATION (net case change)")
        tags.add_column("Tag")
        tags.add_column("Net", justify="right")
        for item in summary["tag_net"]:
            net = item["net"]
            style = "red" if net < 0 else "green" if net > 0 else None
            arrow = " ↓" if net < 0 else " ↑" if net > 0 else ""
            tags.add_row(item["tag"], f"{net:+d}{arrow}", style=style)
        console.print(tags)
