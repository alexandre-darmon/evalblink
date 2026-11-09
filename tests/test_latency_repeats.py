"""Latency measurement, repeats and the noise statistics (intervals, flaky cases)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evalblink import analysis, cache, compare, estimate, openrouter, reporter, runner
from evalblink.validator import validate


def _resp(content, latency=0.5):
    return {
        "response": content,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "cost": 0.0,
        "latency_s": latency,
        "from_cache": False,
    }


def _config(n_cases=2, **extra):
    return {
        "name": "Stats",
        "models": ["m/x"],
        "prompts": [{"id": "v1", "template": "{{ q }}"}],
        "test_cases": [
            {
                "id": f"c{i}",
                "evaluation": "exact_match",
                "variables": {"q": "x"},
                "expected_output": "yes",
            }
            for i in range(n_cases)
        ],
        "concurrency": 1,
        **extra,
    }


def _run(monkeypatch, config, replies):
    calls = []
    replies = iter(replies)

    def fake(client, prompt, model, *a, **k):
        calls.append(k)
        return next(replies)

    monkeypatch.setattr(runner, "openrouter_request", fake)
    results, _ = runner.run(config)
    return results, calls


# ── latency ───────────────────────────────────────────────────────────────────


def test_request_measures_and_caches_latency(
    monkeypatch, tmp_path, client_factory, completion
):
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path))
    ticks = iter([10.0, 11.25])
    monkeypatch.setattr(openrouter.time, "perf_counter", lambda: next(ticks))
    client = client_factory(completion("hi"))
    fresh = openrouter.openrouter_request(client, "p", "m")
    assert fresh["latency_s"] == 1.25
    # A cache hit reports the latency measured when the reply was fetched.
    cached = openrouter.openrouter_request(client, "p", "m")
    assert cached["from_cache"] is True
    assert cached["latency_s"] == 1.25
    assert len(client.calls) == 1


def test_latency_excludes_failed_attempts(
    monkeypatch, client_factory, completion, error_body
):
    ticks = iter([0.0, 30.0, 100.0, 100.5])  # a 30 s 504, then a 0.5 s success
    monkeypatch.setattr(openrouter.time, "perf_counter", lambda: next(ticks))
    client = client_factory([error_body(504), completion("ok")])
    result = openrouter.openrouter_request(client, "p", "m", use_cache=False)
    assert result["latency_s"] == 0.5


@pytest.mark.parametrize("stored", [None, -1, "2", True, float("nan")])
def test_old_or_bad_cached_latency_reads_as_unknown(
    monkeypatch, tmp_path, client_factory, stored
):
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path))
    payload = {
        "model": "m",
        "temperature": 0,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": "p"}],
    }
    entry = {"response": "x", "prompt_tokens": 1, "completion_tokens": 1, "cost": 0}
    if stored is not None:
        entry["latency_s"] = stored
    cache.set(cache.sha256_key(payload), entry)
    result = openrouter.openrouter_request(client_factory([]), "p", "m")
    assert result["from_cache"] is True
    assert result["latency_s"] is None


def test_run_records_latency_per_case_and_percentiles(monkeypatch):
    replies = [_resp("yes", latency) for latency in (1.0, 2.0, 3.0, 4.0)]
    results, _ = _run(monkeypatch, _config(4), replies)
    combo = results[0]
    assert [tc["latency_s"] for tc in combo["test_cases"]] == [1.0, 2.0, 3.0, 4.0]
    assert combo["latency_p50_s"] == 2.5
    assert combo["latency_p95_s"] == 3.85
    assert combo["latency_mean_s"] == 2.5


def test_api_error_has_no_latency_and_is_left_out(monkeypatch):
    def fake(client, prompt, model, *a, **k):
        if prompt == "boom":
            raise RuntimeError("down")
        return _resp("yes", 2.0)

    monkeypatch.setattr(runner, "openrouter_request", fake)
    config = _config(2)
    config["test_cases"][1]["variables"]["q"] = "boom"
    results, _ = runner.run(config)
    cases = results[0]["test_cases"]
    assert cases[1]["latency_s"] is None
    assert results[0]["latency_p50_s"] == 2.0


def test_latency_stats_without_data():
    assert analysis.latency_stats([{"latency_s": None}, {}]) == {
        "latency_p50_s": None,
        "latency_p95_s": None,
        "latency_mean_s": None,
    }


def test_progress_line_shows_latency():
    line = runner._progress_line(
        1,
        1,
        {
            "model": "m",
            "prompt_id": "v1",
            "test_case": {
                "id": "c",
                "evaluation": "exact_match",
                "match_score": 1.0,
                "match_result": True,
                "latency_s": 0.84,
            },
        },
    )
    assert line == "[1/1] m / v1 / c  pass  0.84s"


def test_fastest_combo_and_report_columns(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(reporter, "RESULTS_DIR", str(tmp_path))
    config = _config(2, models=["m/slow", "m/fast"])
    replies = [
        _resp("yes", 3.0),
        _resp("yes", 3.0),
        _resp("no", 0.5),
        _resp("yes", 0.7),
    ]
    results, _ = _run(monkeypatch, config, replies)
    summary = analysis.summarize(results, None)
    assert summary["fastest"]["model"] == "m/fast"
    out = reporter.write(config, results, "2026-01-01_000000")
    printed = capsys.readouterr().out
    assert "Fastest      : m/fast / v1 (p50 0.60 s · 50.0%)" in printed
    assert "3.00 s" in printed
    md = Path(out["markdown"]).read_text()
    assert "| p50 latency | p95 latency |" in md
    assert "| c0 | exact_match | True | 1.0000 | $0.000000 | 3.00 s |" in md


def test_compare_shows_latency_delta_and_tolerates_old_records(capsys):
    def record(run_id, latency):
        case = {"id": "c", "match_score": 1.0, "match_result": True, "total_cost": 0}
        if latency is not None:
            case["latency_s"] = latency
        return {
            "run_id": run_id,
            "results": [
                {
                    "model": "m",
                    "prompt_id": "v1",
                    "score": 100.0,
                    "total_cost": 0.0,
                    "success": 1,
                    "scored": 1,
                    "total": 1,
                    "test_cases": [case],
                }
            ],
        }

    result = compare.diff(record("a", 1.0), record("b", 1.5))
    assert result["rows"][0]["latency_delta"] == 0.5
    reporter.render_comparison(result)
    assert "+0.50 s ↑" in capsys.readouterr().out
    legacy = compare.diff(record("a", None), record("b", 1.5))
    assert legacy["rows"][0]["latency_delta"] is None


# ── repeats ───────────────────────────────────────────────────────────────────


def test_repeats_run_every_case_n_times_with_their_own_cache_entries(monkeypatch):
    replies = [_resp(r) for r in ("yes", "no", "yes", "yes", "yes", "yes")]
    results, calls = _run(monkeypatch, _config(2, repeats=3), replies)
    combo = results[0]
    assert [k["sample"] for k in calls] == [0, 1, 2, 0, 1, 2]
    assert [analysis.case_label(tc) for tc in combo["test_cases"]] == [
        "c0 #1",
        "c0 #2",
        "c0 #3",
        "c1 #1",
        "c1 #2",
        "c1 #3",
    ]
    assert combo["total"] == 6 and combo["success"] == 5
    assert combo["repeats"] == 3
    assert combo["flaky"] == ["c0"]
    assert analysis.summarize(results, None)["flaky"] == 1


def test_without_repeats_cases_carry_no_sample(monkeypatch):
    results, calls = _run(monkeypatch, _config(1), [_resp("yes")])
    assert "sample" not in results[0]["test_cases"][0]
    assert calls[0]["sample"] == 0
    assert results[0]["flaky"] == []


def test_repeat_cache_keys(monkeypatch, tmp_path, client_factory, completion):
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path))
    client = client_factory([completion("a"), completion("b")])
    first = openrouter.openrouter_request(client, "p", "m", sample=0)
    second = openrouter.openrouter_request(client, "p", "m", sample=1)
    assert (first["response"], second["response"]) == ("a", "b")
    # Sample 0 keeps the plain request key, so existing caches still hit.
    again = openrouter.openrouter_request(client, "p", "m")
    assert again["from_cache"] and again["response"] == "a"
    assert len(client.calls) == 2


@pytest.mark.parametrize("repeats", [0, 101, 2.0, True, "3", -1])
def test_validate_rejects_bad_repeats(repeats):
    errors, _ = validate(_config(1, repeats=repeats))
    assert any("repeats must be a whole number from 1 to 100" in e for e in errors)


def test_validate_accepts_repeats():
    assert validate(_config(1, repeats=5)) == ([], [])


def test_estimate_scales_with_repeats():
    meta = {"m/x": {"prompt": 1e-6, "completion": 2e-6, "max_output": None}}
    one = estimate.estimate(_config(2), meta)
    three = estimate.estimate(_config(2, repeats=3), meta)
    assert three["total_cost"] == pytest.approx(3 * one["total_cost"])
    assert three["rows"][0]["est_completion_tokens"] == 3 * 2 * 100


def test_compare_keys_cases_by_repeat():
    def combo(results):
        return {
            "model": "m",
            "prompt_id": "v1",
            "test_cases": [
                {"id": "c", "sample": i, "match_score": float(r), "match_result": r}
                for i, r in enumerate(results)
            ],
        }

    diff = compare.case_diff(combo([True, True]), combo([True, False]))
    changed = [r for r in diff["rows"] if r["transition"] == "regressed"]
    assert [r["id"] for r in changed] == ["c #2"]


# ── intervals and noise ───────────────────────────────────────────────────────


def test_wilson_interval_known_values():
    low, high = analysis.wilson(8, 10)
    assert (round(low, 4), round(high, 4)) == (0.4902, 0.9433)
    assert analysis.wilson(0, 0) is None
    low, high = analysis.wilson(10, 10)
    assert high == pytest.approx(1.0) and round(low, 4) == 0.7225


def _combo(passed, n, model="m"):
    return {
        "model": model,
        "prompt_id": "v1",
        "score": passed / n * 100,
        "success": passed,
        "scored": n,
        "total": n,
        "total_cost": 0.0,
        "test_cases": [
            {"id": f"c{i}", "match_score": 1.0, "match_result": i < passed}
            for i in range(n)
        ],
    }


def test_interval_counts_test_cases_not_repeats():
    combo = _combo(8, 10)
    repeated = {
        **combo,
        "test_cases": [
            dict(tc, sample=s) for tc in combo["test_cases"] for s in (0, 1)
        ],
        "success": 16,
        "scored": 20,
        "total": 20,
    }
    assert analysis.score_interval(repeated) == analysis.score_interval(combo)


def test_within_noise():
    assert analysis.within_noise(_combo(9, 10, "a"), _combo(8, 10, "b"))
    assert not analysis.within_noise(_combo(98, 100, "a"), _combo(60, 100, "b"))
    nothing = {**_combo(0, 1), "scored": 0, "success": 0, "test_cases": []}
    assert not analysis.within_noise(_combo(9, 10), nothing)


def test_summary_and_report_name_combos_within_noise(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(reporter, "RESULTS_DIR", str(tmp_path))
    results = [_combo(9, 10, "m/a"), _combo(8, 10, "m/b"), _combo(1, 10, "m/c")]
    summary = analysis.summarize(results, None)
    assert [c["model"] for c in summary["within_noise"]] == ["m/b"]
    reporter._print_insights(summary, reporter.make_console())
    out = capsys.readouterr().out
    assert "Best quality : m/a / v1 (90.0%, 95% CI 60–98%)" in out
    assert "Within noise : m/b / v1 (80.0%)" in out


def test_compare_flags_a_gain_within_noise(capsys):
    def record(run_id, passed):
        return {"run_id": run_id, "results": [_combo(passed, 10)]}

    result = compare.diff(record("a", 8), record("b", 9))
    assert result["rows"][0]["noise"] is True
    assert "None of these gains is beyond noise" in result["verdict"]
    reporter.render_comparison(result)
    assert "within noise" in capsys.readouterr().out
    big = compare.diff(
        {"run_id": "a", "results": [_combo(20, 100)]},
        {"run_id": "b", "results": [_combo(90, 100)]},
    )
    assert big["rows"][0]["noise"] is False
    assert "noise" not in big["verdict"]


def test_record_with_v2_fields_round_trips(monkeypatch, tmp_path):
    monkeypatch.setattr(reporter, "RESULTS_DIR", str(tmp_path))
    results, _ = _run(monkeypatch, _config(1, repeats=2), [_resp("yes"), _resp("no")])
    out = reporter.write(_config(1, repeats=2), results, "2026-01-01_000000")
    record = compare.load_record(out["json"])
    combo = record["results"][0]
    assert record["schema_version"] == 2
    low, high = combo["score_ci"]
    assert low < combo["score"] == 50.0 < high
    assert combo["flaky"] == ["c0"]
    json.dumps(record)


def test_no_fastest_line_with_a_single_combo(monkeypatch):
    results, _ = _run(monkeypatch, _config(1), [_resp("yes", 0.4)])
    assert analysis.summarize(results, None)["fastest"] is None
