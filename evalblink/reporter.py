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
