"""Tests for reporter helpers: slugified run ids and the run-record builder."""

from __future__ import annotations

from evalblink.reporter import _build_data, _slugify


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
