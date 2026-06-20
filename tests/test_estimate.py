"""Unit tests for the offline cost estimator (``evalblink run --dry-run``).

Pure logic — no network. Each test feeds a config plus a fake pricing catalog
(the shape ``openrouter.fetch_models`` returns) and asserts the estimate.
"""

from __future__ import annotations

from evalblink import estimate
from evalblink.estimate import JUDGE_MAX_TOKENS, _est_tokens


def _meta(prompt=0.0, completion=0.0, context_length=8000):
    return {
        "prompt": prompt,
        "completion": completion,
        "context_length": context_length,
    }


def _config(test_cases, evaluation=None, max_tokens=100, max_cost_usd=None):
    config = {
        "name": "Estimate Test",
        "inference": {"temperature": 0, "max_tokens": max_tokens},
        "models": ["m/x"],
        "prompts": [{"id": "v1", "template": "{{ q }}"}],
        "variables": {},
        "test_cases": test_cases,
    }
    if evaluation is not None:
        config["evaluation"] = evaluation
    if max_cost_usd is not None:
        config["max_cost_usd"] = max_cost_usd
    return config


def test_est_tokens_heuristic():
    assert _est_tokens("") == 0
    assert _est_tokens("a") == 1  # ceil(1/4) floored to min 1
    assert _est_tokens("a" * 8) == 2  # 8 chars / 4


def test_exact_match_cost_uses_prompt_and_completion_pricing():
    config = _config(
        [
            {"id": "c1", "variables": {"q": "abcd"}, "evaluation": "exact_match"},
            {"id": "c2", "variables": {"q": "efgh"}, "evaluation": "exact_match"},
        ],
        max_tokens=10,
    )
    models_meta = {"m/x": _meta(prompt=2.0, completion=3.0)}
    est = estimate.estimate(config, models_meta)

    [row] = est["rows"]
    # Each rendered prompt is 4 chars → 1 prompt token; 2 cases → 2 prompt tokens.
    assert row["est_prompt_tokens"] == 2
    assert row["est_completion_tokens"] == 20  # 2 cases × max_tokens 10
    # cost = 2 prompt_tok * 2.0 + 20 completion_tok * 3.0
    assert row["est_cost"] == 2 * 2.0 + 20 * 3.0
    assert row["est_judge_cost"] == 0.0
    assert est["judge_cost"] == 0.0
    assert est["total_cost"] == row["est_cost"]


def test_llm_judge_adds_judge_cost():
    evaluation = {"judge_model": "j/m", "judge_threshold": 0.7}
    config = _config(
        [
            {
                "id": "c1",
                "variables": {"q": "abcd"},
                "evaluation": "llm_judge",
                "criteria": "be good",
            }
        ],
        evaluation=evaluation,
        max_tokens=10,
    )
    models_meta = {
        "m/x": _meta(prompt=1.0, completion=1.0),
        "j/m": _meta(prompt=2.0, completion=4.0),
    }
    est = estimate.estimate(config, models_meta)
    [row] = est["rows"]

    assert row["est_judge_cost"] > 0
    # Judge completion is the fixed JUDGE_MAX_TOKENS at the judge's completion price.
    assert est["judge_cost"] == row["est_judge_cost"]
    # Total includes both candidate and judge cost.
    assert est["total_cost"] == row["est_cost"] + row["est_judge_cost"]
    # Sanity: judge cost includes the fixed completion budget.
    assert row["est_judge_cost"] >= JUDGE_MAX_TOKENS * 4.0


def test_missing_pricing_is_recorded_and_zero_cost():
    config = _config(
        [{"id": "c1", "variables": {"q": "abcd"}, "evaluation": "exact_match"}],
    )
    est = estimate.estimate(config, models_meta={})  # no pricing for m/x
    assert est["missing_pricing"] == ["m/x"]
    assert est["total_cost"] == 0.0


def test_over_budget_flag():
    config = _config(
        [{"id": "c1", "variables": {"q": "abcd"}, "evaluation": "exact_match"}],
        max_tokens=10,
        max_cost_usd=0.0,
    )
    models_meta = {"m/x": _meta(prompt=1.0, completion=1.0)}
    est = estimate.estimate(config, models_meta)
    assert est["max_cost_usd"] == 0.0
    assert est["total_cost"] > 0
    assert est["over_budget"] is True


def test_within_budget_flag():
    config = _config(
        [{"id": "c1", "variables": {"q": "abcd"}, "evaluation": "exact_match"}],
        max_tokens=10,
        max_cost_usd=1000.0,
    )
    models_meta = {"m/x": _meta(prompt=1.0, completion=1.0)}
    est = estimate.estimate(config, models_meta)
    assert est["over_budget"] is False


def test_no_budget_means_not_over_budget():
    config = _config(
        [{"id": "c1", "variables": {"q": "abcd"}, "evaluation": "exact_match"}],
    )
    models_meta = {"m/x": _meta(prompt=1.0, completion=1.0)}
    est = estimate.estimate(config, models_meta)
    assert est["max_cost_usd"] is None
    assert est["over_budget"] is False
