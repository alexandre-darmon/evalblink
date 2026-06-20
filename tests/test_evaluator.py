"""Tests for the three evaluators and the shared JSON-extraction helper."""

from __future__ import annotations

import json

import pytest

from evalblink.evaluator import (
    _extract_json,
    evaluate_llm_judge,
    exact_match,
    judge_vendor_warning,
    weighted_match,
)

WEIGHTED_PARAMS = {
    "variables": [
        {"name": "use_case", "weight": 0.5},
        {"name": "percent", "weight": 0.25, "tolerance": 0.2},
        {"name": "order", "weight": 0.25},
    ]
}
EXPECTED = [
    {"use_case": "A", "percent": 0.7, "order": 1},
    {"use_case": "B", "percent": 0.3, "order": 2},
]


# --- exact_match ------------------------------------------------------------


def test_exact_match_is_case_and_whitespace_insensitive():
    assert exact_match("Billing", "billing")
    assert exact_match("  billing\n", "billing")
    assert not exact_match("order_issue", "billing")


# --- _extract_json ----------------------------------------------------------


def test_extract_json_plain_and_fenced():
    assert _extract_json('{"a": 1}') == {"a": 1}
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _extract_json('```\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_raises_on_garbage():
    with pytest.raises(json.JSONDecodeError):
        _extract_json("not json at all")


# --- judge_vendor_warning ---------------------------------------------------


def test_vendor_warning_fires_on_shared_vendor():
    msg = judge_vendor_warning(
        "anthropic/claude-sonnet-4-6", ["anthropic/claude-haiku"]
    )
    assert msg is not None
    assert "anthropic" in msg
    assert "anthropic/claude-haiku" in msg


def test_vendor_warning_silent_when_no_overlap():
    assert (
        judge_vendor_warning("anthropic/claude-sonnet", ["openai/gpt-4o", "m/x"])
        is None
    )


def test_vendor_warning_silent_without_judge():
    assert judge_vendor_warning(None, ["anthropic/claude"]) is None


# --- weighted_match ---------------------------------------------------------


def test_weighted_match_perfect_score():
    response = json.dumps(EXPECTED)
    assert weighted_match(WEIGHTED_PARAMS, response, EXPECTED) == pytest.approx(1.0)


def test_weighted_match_accepts_fenced_json():
    response = "```json\n" + json.dumps(EXPECTED) + "\n```"
    assert weighted_match(WEIGHTED_PARAMS, response, EXPECTED) == pytest.approx(1.0)


def test_weighted_match_malformed_returns_zero():
    assert weighted_match(WEIGHTED_PARAMS, "not json", EXPECTED) == 0.0


def test_weighted_match_wrong_shape_returns_zero():
    # Valid JSON, but not a list of objects with the scored keys → 0.0, not a crash.
    assert weighted_match(WEIGHTED_PARAMS, '{"use_case": "A"}', EXPECTED) == 0.0
    assert weighted_match(WEIGHTED_PARAMS, '["A", "B"]', EXPECTED) == 0.0
    assert (
        weighted_match(WEIGHTED_PARAMS, '[{"use_case": "A"}]', EXPECTED) == 0.0
    )  # missing percent/order


def test_weighted_match_partial_percent_out_of_tolerance():
    # B's percent is off by 0.6 (> 0.2 tolerance): labels F1=1, percent=1/2, order=1.
    response = json.dumps(
        [
            {"use_case": "A", "percent": 0.7, "order": 1},
            {"use_case": "B", "percent": 0.9, "order": 2},
        ]
    )
    score = weighted_match(WEIGHTED_PARAMS, response, EXPECTED)
    assert score == pytest.approx(0.5 * 1.0 + 0.25 * 0.5 + 0.25 * 1.0)


# --- evaluate_llm_judge -----------------------------------------------------

JUDGE_PARAMS = {"judge_model": "judge/model", "judge_threshold": 0.7}


def _judge(client_factory, completion, score, reasoning="because"):
    client = client_factory(
        completion(json.dumps({"reasoning": reasoning, "score": score}))
    )
    return evaluate_llm_judge(client, JUDGE_PARAMS, "answer", "task", "criteria")


def test_judge_success_above_threshold(disable_cache, client_factory, completion):
    result = _judge(client_factory, completion, 5)
    assert result["status"] == "success"
    assert result["match_result"] is True
    assert result["score_raw"] == 5
    assert result["score_normalized"] == pytest.approx(1.0)
    assert result["reasoning"] == "because"


def test_judge_fail_below_threshold(disable_cache, client_factory, completion):
    result = _judge(client_factory, completion, 2)
    assert result["status"] == "fail"
    assert result["match_result"] is False
    assert result["score_normalized"] == pytest.approx(0.25)


def test_judge_clamps_out_of_range_scores(disable_cache, client_factory, completion):
    assert _judge(client_factory, completion, 9)["score_raw"] == 5
    assert _judge(client_factory, completion, 0)["score_raw"] == 1


def test_judge_accepts_fenced_json(disable_cache, client_factory, completion):
    client = client_factory(completion('```json\n{"reasoning": "ok", "score": 4}\n```'))
    result = evaluate_llm_judge(client, JUDGE_PARAMS, "answer", "task", "criteria")
    assert result["status"] == "success"
    assert result["score_raw"] == 4


def test_judge_malformed_json_is_pipeline_error(
    disable_cache, client_factory, completion
):
    client = client_factory(completion("totally not json"))
    result = evaluate_llm_judge(client, JUDGE_PARAMS, "answer", "task", "criteria")
    assert result["status"] == "judge_error"
    assert result["score_normalized"] is None


def test_judge_api_error_is_pipeline_error(disable_cache, client_factory, error_body):
    # 500 is non-retryable: openrouter raises, the judge catches it as an API error.
    client = client_factory(error_body(500, "server blew up"))
    result = evaluate_llm_judge(client, JUDGE_PARAMS, "answer", "task", "criteria")
    assert result["status"] == "judge_api_error"
    assert result["score_normalized"] is None


def test_judge_requires_judge_model(client_factory, completion):
    client = client_factory(completion("{}"))
    with pytest.raises(ValueError, match="Judge model"):
        evaluate_llm_judge(client, {}, "answer", "task", "criteria")
