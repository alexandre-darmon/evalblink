"""End-to-end pipeline tests: drive the real ``runner.run`` for every eval mode.

No network — the two API seams (``runner.openrouter_request`` for candidate calls,
``evaluator.openrouter_request`` for judge calls) are monkeypatched with queued
canned bodies. Each test runs a config through ``runner.run`` and, where relevant,
on through ``analysis.summarize`` and ``reporter.write`` to prove the report is
correct for that mode.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pytest

from evalblink import analysis, estimate, evaluator, reporter, runner
from evalblink.validator import validate


def _resp(content, cost=0.0):
    """A candidate/judge completion as ``openrouter_request`` returns it."""
    return {
        "response": content,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "cost": cost,
        "from_cache": False,
    }


def _run(monkeypatch, config, candidate_responses, judge_responses=None):
    cand = iter(candidate_responses)

    def fake_candidate(client, prompt, model, *a, **k):
        return next(cand)

    monkeypatch.setattr(runner, "openrouter_request", fake_candidate)

    if judge_responses is not None:
        jud = iter(judge_responses)

        def fake_judge(client, *a, **k):
            item = next(jud)
            if isinstance(item, Exception):
                raise item
            return item

        monkeypatch.setattr(evaluator, "openrouter_request", fake_judge)

    results, _ = runner.run(config)
    return results


def _config(test_cases, evaluation=None, prompt=None):
    config = {
        "name": "Pipeline Test",
        "inference": {"temperature": 0, "max_tokens": 50},
        "models": ["m/x"],
        "prompts": [prompt or {"id": "v1", "template": "{{ q }}"}],
        "variables": {},
        "test_cases": test_cases,
        # Force sequential execution so iterator-based mocks remain deterministic.
        "concurrency": 1,
    }
    if evaluation is not None:
        config["evaluation"] = evaluation
    return config


# --- exact_match ----------------------------------------------------------


def test_exact_match_pipeline(monkeypatch):
    config = _config(
        [
            {
                "id": "c1",
                "variables": {"q": "a"},
                "expected_output": "order",
                "evaluation": "exact_match",
                "tags": ["easy"],
            },
            {
                "id": "c2",
                "variables": {"q": "b"},
                "expected_output": "order",
                "evaluation": "exact_match",
                "tags": ["edge"],
            },
        ]
    )
    [combo] = _run(monkeypatch, config, [_resp("order"), _resp("nope")])
    assert combo["success"] == 1
    assert combo["scored"] == 2
    assert combo["errors"] == 0
    assert combo["score"] == 50.0
    statuses = {tc["id"]: tc["status"] for tc in combo["test_cases"]}
    assert statuses == {"c1": "scored", "c2": "scored"}


# Absent key, and a bare `inference:` line in YAML (parsed as None).
@pytest.mark.parametrize("inference", ["absent", None])
def test_missing_inference_block_uses_defaults(monkeypatch, inference):
    config = _config(
        [
            {
                "id": "c1",
                "variables": {"q": "a"},
                "expected_output": "order",
                "evaluation": "exact_match",
            }
        ]
    )
    if inference == "absent":
        del config["inference"]
    else:
        config["inference"] = inference
    seen = []

    def fake_candidate(client, prompt, model, temperature, max_tokens, *a, **k):
        seen.append((temperature, max_tokens))
        return _resp("order")

    monkeypatch.setattr(runner, "openrouter_request", fake_candidate)
    results, _ = runner.run(config)
    assert seen == [(0, 100)]
    assert results[0]["score"] == 100.0

    data = reporter._build_data(config, results, "2026-01-01_00-00")
    assert (data["temperature"], data["max_tokens"]) == (0, 100)


# --- weighted_match -------------------------------------------------------


def test_weighted_match_pipeline(monkeypatch):
    evaluation = {
        "quality_threshold": 0.5,
        "variables": [
            {"name": "use_case", "weight": 0.5},
            {"name": "percent", "weight": 0.25, "tolerance": 0.2},
            {"name": "order", "weight": 0.25},
        ],
    }
    expected = [{"use_case": "A", "percent": 0.5, "order": 1}]
    config = _config(
        [
            {
                "id": "c1",
                "variables": {"q": "x"},
                "expected_output": expected,
                "evaluation": "weighted_match",
                "tags": ["w"],
            }
        ],
        evaluation=evaluation,
    )
    # Candidate echoes the expected array exactly → perfect score of 1.0.
    [combo] = _run(monkeypatch, config, [_resp(json.dumps(expected))])
    tc = combo["test_cases"][0]
    assert tc["match_score"] == 1.0
    assert tc["match_result"] is True
    assert combo["scored"] == 1
    assert combo["errors"] == 0


# --- llm_judge: success ---------------------------------------------------


def test_llm_judge_pipeline_pass_and_fail(monkeypatch):
    evaluation = {"judge_model": "j/m", "judge_threshold": 0.7}
    config = _config(
        [
            {
                "id": "c1",
                "variables": {"q": "x"},
                "evaluation": "llm_judge",
                "criteria": "be good",
                "tags": ["j"],
            },
            {
                "id": "c2",
                "variables": {"q": "y"},
                "evaluation": "llm_judge",
                "criteria": "be good",
                "tags": ["j"],
            },
        ],
        evaluation=evaluation,
    )
    [combo] = _run(
        monkeypatch,
        config,
        candidate_responses=[_resp("answer one"), _resp("answer two")],
        judge_responses=[
            _resp(json.dumps({"reasoning": "great", "score": 5})),
            _resp(json.dumps({"reasoning": "poor", "score": 1})),
        ],
    )
    assert combo["success"] == 1  # score 5 passes, score 1 fails
    assert combo["scored"] == 2
    assert combo["errors"] == 0
    assert combo["score"] == 50.0


# --- llm_judge: pipeline failure (the conflation fix) ---------------------


def test_llm_judge_pipeline_failure_excluded_from_scoring(monkeypatch):
    evaluation = {"judge_model": "j/m", "judge_threshold": 0.7}
    config = _config(
        [
            {
                "id": "c1",
                "variables": {"q": "x"},
                "evaluation": "llm_judge",
                "criteria": "be good",
                "tags": ["edge_case"],
            }
        ],
        evaluation=evaluation,
    )
    # Judge returns malformed JSON → pipeline error, not a real 0.
    [combo] = _run(
        monkeypatch,
        config,
        candidate_responses=[_resp("an answer")],
        judge_responses=[_resp("{not json")],
    )
    tc = combo["test_cases"][0]
    assert tc["status"] == "judge_error"
    assert tc["match_score"] is None
    assert combo["scored"] == 0
    assert combo["errors"] == 1
    assert combo["score"] == 0.0  # no scored cases, not a 0% real failure

    # The error must NOT trigger a tag warning or count in the tag denominator.
    summary = analysis.summarize([combo], None)
    assert summary["errors"] == 1
    assert summary["warnings"] == []
    row = summary["tag_breakdown"][0]
    assert row["scored"] == 0 and row["errors"] == 1 and row["quality"] is None


# --- candidate API failure: recorded, run continues -----------------------


@pytest.mark.parametrize(
    "exc",
    [RuntimeError("OpenRouter error 400: bad model"), httpx.ConnectError("refused")],
)
def test_candidate_api_error_is_recorded_and_run_continues(
    tmp_path, monkeypatch, capsys, exc
):
    monkeypatch.setattr(reporter, "RESULTS_DIR", str(tmp_path))
    cases = [
        {
            "id": f"c{i}",
            "variables": {"q": str(i)},
            "expected_output": "ok",
            "evaluation": "exact_match",
            "tags": ["t"],
        }
        for i in range(3)
    ]
    config = _config(cases)
    responses = iter([_resp("ok", cost=0.1), exc, _resp("ok", cost=0.1)])

    def fake_candidate(client, prompt, model, *a, **k):
        item = next(responses)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(runner, "openrouter_request", fake_candidate)
    results, ts = runner.run(config)
    assert f"API error on m/x / v1 / c1: {exc}" in capsys.readouterr().out
    [combo] = results
    assert (combo["success"], combo["scored"], combo["errors"]) == (2, 2, 1)
    assert combo["score"] == 100.0  # the error is excluded, not a real fail
    failed = combo["test_cases"][1]
    assert failed["status"] == "api_error"
    assert failed["match_score"] is None and failed["match_result"] is False
    assert failed["response"] == "" and failed["error"] == str(exc)
    assert failed["total_cost"] == 0.0 and failed["prompt_tokens"] == 0
    assert "error" not in combo["test_cases"][0]

    out = reporter.write(config, results, ts)
    data = json.loads(Path(out["json"]).read_text())
    assert data["results"][0]["test_cases"][1]["status"] == "api_error"
    assert data["insights"]["errors"] == 1
    assert "| c1 | exact_match | False |  |" in Path(out["markdown"]).read_text()


# A bare `evaluation:` line in YAML parses as None — must read as absent.
def test_null_evaluation_block_through_runner_estimate_reporter(tmp_path, monkeypatch):
    monkeypatch.setattr(reporter, "RESULTS_DIR", str(tmp_path))
    config = _config(
        [
            {
                "id": "c1",
                "variables": {"q": "a"},
                "expected_output": "ok",
                "evaluation": "exact_match",
            }
        ]
    )
    config["evaluation"] = None
    assert validate(config) == ([], [])
    est = estimate.estimate(config, {"m/x": {"prompt": 0.0, "completion": 0.0}})
    assert est["judge_cost"] == 0.0
    results = _run(monkeypatch, config, [_resp("ok")])
    out = reporter.write(config, results, "2026-01-01_000000")
    data = json.loads(Path(out["json"]).read_text())
    assert data["judge_model"] is None and data["quality_threshold"] is None


# --- robustness -----------------------------------------------------------


def test_missing_tags_and_variables_do_not_crash(monkeypatch):
    config = _config(
        [
            # No `tags`, no `variables` keys at all.
            {"id": "c1", "expected_output": "ok", "evaluation": "exact_match"}
        ],
        prompt={"id": "v1", "template": "static prompt"},
    )
    [combo] = _run(monkeypatch, config, [_resp("ok")])
    assert combo["success"] == 1
    assert combo["test_cases"][0]["tags"] == []


def test_unknown_evaluation_type_raises(monkeypatch):
    config = _config(
        [
            {
                "id": "c1",
                "variables": {"q": "x"},
                "evaluation": "exactmatch",  # typo
                "expected_output": "x",
            }
        ]
    )
    try:
        _run(monkeypatch, config, [_resp("x")])
        raised = False
    except ValueError as exc:
        raised = "exactmatch" in str(exc)
    assert raised


# --- mixed config: all three modes in one run -----------------------------


def test_mixed_modes_report_renders(tmp_path, monkeypatch):
    monkeypatch.setattr(reporter, "RESULTS_DIR", str(tmp_path))
    expected = [{"use_case": "A", "percent": 0.5, "order": 1}]
    evaluation = {
        "judge_model": "j/m",
        "judge_threshold": 0.7,
        "quality_threshold": 0.5,
        "variables": [
            {"name": "use_case", "weight": 0.5},
            {"name": "percent", "weight": 0.25, "tolerance": 0.2},
            {"name": "order", "weight": 0.25},
        ],
    }
    config = _config(
        [
            {
                "id": "ex",
                "variables": {"q": "x"},
                "expected_output": "ok",
                "evaluation": "exact_match",
                "tags": ["exact"],
            },
            {
                "id": "wt",
                "variables": {"q": "y"},
                "expected_output": expected,
                "evaluation": "weighted_match",
                "tags": ["weighted"],
            },
            {
                "id": "jg",
                "variables": {"q": "z"},
                "evaluation": "llm_judge",
                "criteria": "be good",
                "tags": ["judge"],
            },
        ],
        evaluation=evaluation,
    )
    results = _run(
        monkeypatch,
        config,
        candidate_responses=[_resp("ok"), _resp(json.dumps(expected)), _resp("ans")],
        judge_responses=[_resp(json.dumps({"score": 5}))],
    )
    out = reporter.write(config, results, "2026-01-01_000000")
    data = json.loads(Path(out["json"]).read_text())
    # All three eval modes present and well-formed in the persisted insights.
    assert data["insights"] is not None
    evals = {tc["evaluation"] for r in data["results"] for tc in r["test_cases"]}
    assert evals == {"exact_match", "weighted_match", "llm_judge"}


# --- concurrency ----------------------------------------------------------


def test_concurrency_preserves_config_order(monkeypatch):
    """With workers > 1 and completion order reversed, results stay in config order.

    Responses are keyed by rendered prompt (not a shared iterator) so the mock is
    thread-safe, and each worker sleeps inversely to its index so earlier-submitted
    cases finish *last* — proving the runner collects by submission order, not by
    whichever future resolves first.
    """
    n = 6
    cases = [
        {
            "id": f"c{i}",
            "variables": {"q": str(i)},
            "expected_output": "ok",
            "evaluation": "exact_match",
        }
        for i in range(n)
    ]
    config = _config(cases)
    config["concurrency"] = 3

    # Even-indexed cases match ("ok"); odd ones miss. Keyed by rendered prompt.
    responses = {str(i): _resp("ok" if i % 2 == 0 else "miss") for i in range(n)}

    def fake_candidate(client, prompt, model, *a, **k):
        # Earlier index → longer sleep → finishes later. Reverses completion order.
        time.sleep((n - int(prompt)) * 0.01)
        return responses[prompt]

    monkeypatch.setattr(runner, "openrouter_request", fake_candidate)

    [combo] = runner.run(config)[0]
    ids = [tc["id"] for tc in combo["test_cases"]]
    assert ids == [f"c{i}" for i in range(n)]  # submission/config order preserved
    assert combo["success"] == 3  # i = 0, 2, 4 matched "ok"
    assert combo["scored"] == n


@pytest.mark.parametrize("concurrency", [0, None])
def test_concurrency_clamped_to_minimum_one(monkeypatch, concurrency):
    """A non-positive ``concurrency`` is clamped to 1 instead of crashing.

    ``ThreadPoolExecutor(max_workers=0)`` raises ``ValueError`` — the runner must
    guard against a config that sets ``concurrency: 0``. A bare ``concurrency:``
    (None) falls back to the default.
    """
    config = _config(
        [
            {
                "id": "c1",
                "variables": {"q": "x"},
                "expected_output": "ok",
                "evaluation": "exact_match",
            }
        ]
    )
    config["concurrency"] = concurrency
    [combo] = _run(monkeypatch, config, [_resp("ok")])
    assert combo["success"] == 1
    assert combo["scored"] == 1
