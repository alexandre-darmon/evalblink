"""All output for a benchmark run: terminal table, JSON, Markdown.

This module makes no decisions — it only formats and writes the raw results it
is handed. ``write`` is the single entry point.
"""

from __future__ import annotations

import json
import os

from rich.console import Console
from rich.table import Table

RESULTS_DIR = "results"


def _build_data(config, results, timestamp) -> dict:
    """The canonical run record — the exact shape persisted as JSON."""
    benchmark_name = config["name"]
    run_id = f"{timestamp}_{benchmark_name}"
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
        "quality_threshold": config["evaluation"].get(
            "quality_threshold", config["evaluation"].get("judge_threshold")
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
        "| Model | Prompt | Score | Pass/Total | Prompt tok | Completion tok | Cost |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for run in data["results"]:
        lines.append(
            f"| {run['model']} | {run['prompt_id']} | {run['score']:.1f}% | "
            f"{run['success']}/{run['total']} | {run['total_prompt_tokens']} | "
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

    with open(file_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return file_path


def _print_table(data, console: Console) -> None:
    table = Table(title=f"{data['benchmark']} — {data['run_id']}")
    table.add_column("Model")
    table.add_column("Prompt")
    table.add_column("Score", justify="right")
    table.add_column("Pass/Total", justify="right")
    table.add_column("Prompt tok", justify="right")
    table.add_column("Completion tok", justify="right")
    table.add_column("Cost", justify="right")
    for run in data["results"]:
        table.add_row(
            run["model"],
            run["prompt_id"],
            f"{run['score']:.1f}%",
            f"{run['success']}/{run['total']}",
            str(run["total_prompt_tokens"]),
            str(run["total_completion_tokens"]),
            f"${run['total_cost']:.6f}",
        )
    console.print(table)


def write(config, results, timestamp) -> dict:
    """Render the terminal table and write the JSON + Markdown files.

    Returns the paths of the files written.
    """
    data = _build_data(config, results, timestamp)
    json_path = _write_json(data)
    md_path = _write_markdown(data)
    _print_table(data, Console())
    print(f"Results saved to {json_path}")
    print(f"Markdown report saved to {md_path}")
    return {"json": json_path, "markdown": md_path}
