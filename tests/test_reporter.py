"""Tests for reporter helpers: slugified run ids and the run-record builder."""

from __future__ import annotations

import json

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

    data = json.loads(open(out["json"]).read())
    insights = data["insights"]
    assert insights["best_quality"]["model"] == "m/cheap"
    assert insights["warnings"] == ["edge_case"]
    assert insights["passed"] is False

    md = open(out["markdown"]).read()
    assert "Quality by tag" in md
    assert "Recommendation" in md
    assert "edge_case" in md


def test_write_passes_gate_when_no_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr(reporter, "RESULTS_DIR", str(tmp_path))
    config = {"name": "No Gate", "inference": {"temperature": 0, "max_tokens": 50}}
    out = write(config, _results(), "ts")
    assert out["passed"] is True
