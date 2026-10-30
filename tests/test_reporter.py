"""Tests for reporter helpers: slugified run ids and the run-record builder."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest

from evalblink import reporter
from evalblink.reporter import _build_data, _slugify, write


def test_slugify_lowercases_and_hyphenates():
    assert (
        _slugify("Customer Support Classification") == "customer-support-classification"
    )
    assert _slugify("a/b  c!") == "a-b-c"


def test_build_data_without_evaluation_block():
    # exact_match configs have no `evaluation:` block — must not KeyError.
    config = {"name": "X Y", "inference": {"temperature": 0, "max_tokens": 50}}
    data = _build_data(config, [], "2026-01-01_000000")
    assert data["run_id"] == "2026-01-01_000000_x-y"
    assert data["judge_model"] is None
    assert data["quality_threshold"] is None


def test_build_data_falls_back_to_judge_threshold():
    config = {
        "name": "X",
        "inference": {"temperature": 0, "max_tokens": 100},
        "evaluation": {"judge_model": "j/model", "judge_threshold": 0.7},
    }
    data = _build_data(config, [], "ts")
    assert data["judge_model"] == "j/model"
    assert data["quality_threshold"] == 0.7


def _results():
    return [
        {
            "model": "m/cheap",
            "prompt_id": "v1",
            "success": 1,
            "scored": 2,
            "errors": 0,
            "total": 2,
            "score": 50.0,
            "total_prompt_tokens": 10,
            "total_completion_tokens": 5,
            "total_cost": 0.0,
            "test_cases": [
                {
                    "id": "c1",
                    "tags": ["easy"],
                    "evaluation": "exact_match",
                    "status": "scored",
                    "match_result": True,
                    "match_score": 1.0,
                    "total_cost": 0.0,
                },
                {
                    "id": "c2",
                    "tags": ["edge_case"],
                    "evaluation": "exact_match",
                    "status": "scored",
                    "match_result": False,
                    "match_score": 0.0,
                    "total_cost": 0.0,
                },
            ],
        }
    ]


def test_write_embeds_insights_in_json_and_markdown(tmp_path, monkeypatch):
    monkeypatch.setattr(reporter, "RESULTS_DIR", str(tmp_path))
    config = {
        "name": "Gate Demo",
        "inference": {"temperature": 0, "max_tokens": 50},
        "quality_threshold": 80,  # top-level CI gate (0-100 %)
    }
    out = write(config, _results(), "2026-01-01_000000")

    # Best combo scores 50% < 80% threshold → gate fails.
    assert out["passed"] is False

    data = json.loads(Path(out["json"]).read_text())
    assert data["schema_version"] == 1
    insights = data["insights"]
    assert insights["best_quality"]["model"] == "m/cheap"
    assert insights["warnings"] == ["edge_case"]
    assert insights["passed"] is False

    md = Path(out["markdown"]).read_text()
    assert "Quality by tag" in md
    assert "- **CI gate (quality_threshold):** 80" in md
    assert "- **Per-case threshold:** None" in md
    assert "Recommendation" in md
    assert "edge_case" in md


def test_write_passes_gate_when_no_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr(reporter, "RESULTS_DIR", str(tmp_path))
    config = {"name": "No Gate", "inference": {"temperature": 0, "max_tokens": 50}}
    out = write(config, _results(), "ts")
    assert out["passed"] is True


@pytest.mark.parametrize(
    "name, slug",
    [("!!!", "benchmark"), ("日本語ベンチ", "benchmark"), ("", "benchmark")],
)
def test_slugify_falls_back_when_empty(name, slug):
    assert _slugify(name) == slug


def test_slugify_caps_length_without_trailing_hyphen():
    slug = _slugify("a" * 59 + " " + "b" * 5000)
    assert len(slug) <= 60 and not slug.endswith("-")


def test_any_name_writes_results(tmp_path, monkeypatch):
    monkeypatch.setattr(reporter, "RESULTS_DIR", str(tmp_path))
    for name in ("x" * 5000, "!!!"):
        out = write({"name": name}, _results(), "ts")
        assert Path(out["json"]).exists() and Path(out["markdown"]).exists()


@pytest.mark.parametrize(
    "status, label",
    [
        ("api_error", "API error"),
        ("judge_api_error", "judge API error"),
        ("judge_error", "judge error"),
        ("internal_error", "internal error"),
    ],
)
def test_markdown_shows_pipeline_error_status(tmp_path, monkeypatch, status, label):
    monkeypatch.setattr(reporter, "RESULTS_DIR", str(tmp_path))
    results = _results()
    results[0]["test_cases"][1].update(status=status, match_score=None)
    out = write({"name": "E"}, results, "ts")
    assert (
        f"| c2 | exact_match | False | {label} |" in Path(out["markdown"]).read_text()
    )


def test_yaml_dates_in_results_are_json_serialized(tmp_path, monkeypatch):
    # An unquoted `expected_output: 2024-01-01` loads as a datetime.date.
    monkeypatch.setattr(reporter, "RESULTS_DIR", str(tmp_path))
    results = _results()
    results[0]["test_cases"][0]["expected"] = datetime.date(2024, 1, 1)
    out = write({"name": "D"}, results, "ts")
    data = json.loads(Path(out["json"]).read_text())
    assert data["results"][0]["test_cases"][0]["expected"] == "2024-01-01"


def test_unserializable_record_leaves_no_truncated_json(tmp_path, monkeypatch):
    monkeypatch.setattr(reporter, "RESULTS_DIR", str(tmp_path))
    results = _results()
    loop = []
    loop.append(loop)
    results[0]["test_cases"][0]["expected"] = loop
    with pytest.raises(ValueError):
        write({"name": "L"}, results, "ts")
    assert list(tmp_path.iterdir()) == []


def _rendered(render, data):
    from rich.console import Console

    console = Console(record=True, width=200, markup=False, emoji=False)
    render(data, console=console)
    return console.export_text()


def test_estimate_warns_about_a_budget_above_the_output_limit():
    est = {
        "rows": [],
        "total_cost": 0.0,
        "judge_cost": 0.0,
        "missing_pricing": [],
        "max_cost_usd": None,
        "over_budget": False,
        "over_output_limit": [
            {
                "model": "j/x",
                "setting": "evaluation.judge_max_tokens",
                "budget": 4096,
                "limit": 2048,
            }
        ],
    }
    out = _rendered(reporter.render_estimate, est)
    assert "evaluation.judge_max_tokens (4096) is above j/x's output limit (2048" in out


def test_comparison_notes_pipeline_error_counts():
    row = {
        "model": "m/x",
        "prompt_id": "v1",
        "score_delta": 50.0,
        "cost_delta": 0.0,
        "present_in": "both",
        "errors_a": 0,
        "errors_b": 3,
    }
    out = _rendered(
        reporter.render_comparison,
        {"run_a": "a", "run_b": "b", "rows": [row], "verdict": "v"},
    )
    assert "pipeline errors 0 → 3" in out


@pytest.mark.parametrize(
    "row, note, style",
    [
        (
            {"errors_a": 0, "errors_b": 3, "score_delta": 50.0},
            "pipeline errors 0 → 3",
            "yellow",
        ),
        (
            {"errors_a": 2, "errors_b": 2, "score_delta": 50.0},
            "pipeline errors 2 → 2",
            "green",
        ),
        ({"errors_a": 0, "errors_b": 0, "score_delta": -9.0}, "", "red"),
        ({"errors_a": 0, "errors_b": 0, "score_delta": 0.0}, "", None),
        ({"present_in": "a_only", "errors_a": 2, "errors_b": None}, "only in A", None),
        ({"present_in": "b_only", "errors_a": None, "errors_b": 1}, "only in B", None),
    ],
)
def test_comparison_row_note_and_style(row, note, style):
    row = {"present_in": "both", "score_delta": None, **row}
    assert reporter._comparison_note_and_style(row) == (note, style)
