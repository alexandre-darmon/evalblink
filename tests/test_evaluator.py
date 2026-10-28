"""Tests for the three evaluators and the shared JSON-extraction helper."""

from __future__ import annotations

import json

import pytest

from evalblink import evaluator
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


@pytest.mark.parametrize(
    "response",
    [
        '[{"use_case": "A", "percent": "50%", "order": 1}]',  # non-numeric percent
        '[{"use_case": ["A"], "percent": 0.5, "order": 1}]',  # unhashable label
    ],
)
def test_weighted_match_wrong_value_types_return_zero(response):
    # Would raise TypeError mid-run without the guard — a task failure instead.
    assert weighted_match(WEIGHTED_PARAMS, response, EXPECTED) == 0.0


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


@pytest.mark.parametrize(
    "reply",
    [
        '{"score": Infinity}',
        '{"score": NaN}',
        '{"score": "four"}',
        '{"score": null}',
        '{"score": [4]}',
        "[4]",
        "[" * 100_000 + "]" * 100_000,
    ],
)
def test_judge_unusable_score_is_pipeline_error(
    disable_cache, client_factory, completion, reply
):
    client = client_factory(completion(reply))
    result = evaluate_llm_judge(client, JUDGE_PARAMS, "answer", "task", "criteria")
    assert result["status"] == "judge_error"
    assert result["score_normalized"] is None


def test_judge_out_of_range_score_is_clamped(disable_cache, client_factory, completion):
    client = client_factory(completion('{"score": 99}'))
    result = evaluate_llm_judge(client, JUDGE_PARAMS, "answer", "task", "criteria")
    assert result["score_raw"] == 5 and result["score_normalized"] == 1.0


def test_weighted_match_deeply_nested_json_scores_zero():
    deep = "[" * 100_000 + "]" * 100_000
    assert weighted_match(WEIGHTED_PARAMS, deep, EXPECTED) == 0.0


def test_extract_json_huge_int_is_a_decode_error():
    # Past Python's 4300-digit int-conversion limit json.loads raises a plain
    # ValueError; it must surface as JSONDecodeError like any other bad JSON.
    with pytest.raises(json.JSONDecodeError):
        _extract_json("[" + "1" * 5000 + "]")


@pytest.mark.parametrize(
    "percent",
    ["1" * 5000, "1" + "0" * 400, "1e400", "NaN", "-Infinity"],
)
def test_weighted_match_non_finite_percent_scores_zero(percent):
    response = f'[{{"use_case": "x", "percent": {percent}, "order": 1}}]'
    expected = [{"use_case": "x", "percent": 0.5, "order": 1}]
    assert weighted_match(WEIGHTED_PARAMS, response, expected) == 0.0


def test_judge_huge_int_score_is_pipeline_error(
    disable_cache, client_factory, completion
):
    client = client_factory(completion('{"score": ' + "9" * 5000 + "}"))
    result = evaluate_llm_judge(client, {"judge_model": "j/m"}, "r", "t", "c")
    assert result["status"] == "judge_error" and result["score_normalized"] is None


def test_judge_use_cache_is_passed_through(monkeypatch):
    seen = {}

    def fake(client, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop")

    monkeypatch.setattr(evaluator, "openrouter_request", fake)
    evaluate_llm_judge(None, {"judge_model": "j/m"}, "r", "t", "c", use_cache=False)
    assert seen["use_cache"] is False


def test_judge_null_threshold_uses_the_default(
    disable_cache, client_factory, completion
):
    # A bare `judge_threshold:` loads as None and means "default 0.70".
    client = client_factory(completion(json.dumps({"reasoning": "r", "score": 4})))
    params = {**JUDGE_PARAMS, "judge_threshold": None}
    result = evaluate_llm_judge(client, params, "answer", "task", "criteria")
    assert result["match_result"] is True  # 0.75 >= 0.70


def test_judge_budget_defaults_high_and_is_configurable(
    disable_cache, client_factory, completion
):
    from evalblink.schemas import DEFAULT_JUDGE_MAX_TOKENS

    body = completion(json.dumps({"reasoning": "r", "score": 4}))
    client = client_factory(body)
    evaluate_llm_judge(client, JUDGE_PARAMS, "answer", "task", "criteria")
    assert client.calls[0]["json"]["max_tokens"] == DEFAULT_JUDGE_MAX_TOKENS >= 4096
    client = client_factory(body)
    params = {**JUDGE_PARAMS, "judge_max_tokens": 777}
    evaluate_llm_judge(client, params, "answer", "task", "criteria")
    assert client.calls[0]["json"]["max_tokens"] == 777


def test_judge_api_error_keeps_the_reason(disable_cache, client_factory, error_body):
    client = client_factory(error_body(400, "model not found"))
    result = evaluate_llm_judge(client, JUDGE_PARAMS, "answer", "task", "criteria")
    assert result["status"] == "judge_api_error"
    assert "model not found" in result["error"]


def test_judge_errors_name_the_judge_budget(disable_cache, client_factory):
    null_reply = {
        "choices": [{"message": {"content": None}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 9, "cost": 0.0},
    }
    result = evaluate_llm_judge(
        client_factory(null_reply), JUDGE_PARAMS, "answer", "task", "criteria"
    )
    assert result["status"] == "judge_api_error"
    assert "evaluation.judge_max_tokens" in result["error"]

    cut_off = {
        "choices": [{"message": {"content": '{"reasoning": "long and'}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 16, "cost": 0.0},
    }
    result = evaluate_llm_judge(
        client_factory(cut_off), JUDGE_PARAMS, "answer", "task", "criteria"
    )
    assert result["status"] == "judge_error"
    assert "16 completion tokens" in result["error"]
    assert "evaluation.judge_max_tokens" in result["error"]
