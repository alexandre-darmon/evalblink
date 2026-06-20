"""Tests for the insight layer: tag breakdown, winners, and the CI gate.

All pure — no API, no I/O.
"""

from __future__ import annotations

from evalblink.analysis import (
    best_quality,
    best_value,
    summarize,
    tag_breakdown,
)


_AUTO = object()


def _case(tags, match_result, match_score=_AUTO):
    # A real (scored) case derives match_score from match_result; pass
    # match_score=None to model a pipeline error (judge API/parse failure).
    if match_score is _AUTO:
        match_score = 1.0 if match_result else 0.0
    return {
        "id": "c",
        "tags": tags,
        "match_result": match_result,
        "match_score": match_score,
    }


def _error_case(tags):
    return _case(tags, match_result=False, match_score=None)


def _combo(model, prompt_id, score, total_cost, cases):
    return {
        "model": model,
        "prompt_id": prompt_id,
        "score": score,
        "total_cost": total_cost,
        "test_cases": cases,
    }


def test_tag_breakdown_counts_multi_tag_cases_toward_each_tag():
    combo = _combo(
        "m",
        "p",
        50.0,
        0.0,
        [
            _case(["easy", "order"], True),
            _case(["easy"], True),
            _case(["edge_case", "order"], False),
        ],
    )
    rows = {r["tag"]: r for r in tag_breakdown(combo)}
    assert rows["easy"] == {
        "tag": "easy",
        "cases": 2,
        "scored": 2,
        "errors": 0,
        "quality": 100.0,
        "warn": False,
    }
    # "order" appears on a passing and a failing case → 50%.
    assert rows["order"]["cases"] == 2
    assert rows["order"]["quality"] == 50.0
    assert rows["order"]["warn"] is True
    assert rows["edge_case"]["quality"] == 0.0


def test_tag_breakdown_excludes_pipeline_errors_from_denominator():
    # One pass + one judge error on the same tag → 100% over 1 scored case,
    # not 50%. The error is tallied separately and never warns.
    combo = _combo("m", "p", 100.0, 0.0, [_case(["t"], True), _error_case(["t"])])
    row = tag_breakdown(combo)[0]
    assert row == {
        "tag": "t",
        "cases": 2,
        "scored": 1,
        "errors": 1,
        "quality": 100.0,
        "warn": False,
    }


def test_tag_breakdown_all_errors_tag_has_no_quality_and_no_warn():
    combo = _combo("m", "p", 0.0, 0.0, [_error_case(["t"])])
    row = tag_breakdown(combo)[0]
    assert row["scored"] == 0
    assert row["errors"] == 1
    assert row["quality"] is None
    assert row["warn"] is False


def test_tag_breakdown_warn_boundary_at_70():
    # 7/10 = exactly 70% → not a warning; 69% would warn.
    at_70 = _combo("m", "p", 0, 0, [_case(["t"], i < 7) for i in range(10)])
    below = _combo("m", "p", 0, 0, [_case(["t"], i < 6) for i in range(10)])
    assert tag_breakdown(at_70)[0]["warn"] is False
    assert tag_breakdown(below)[0]["warn"] is True


def test_tag_breakdown_sorts_warnings_first_then_by_name():
    combo = _combo(
        "m",
        "p",
        0,
        0,
        [
            _case(["zebra"], True),  # 100% no warn
            _case(["alpha"], False),  # 0% warn
            _case(["beta"], False),  # 0% warn
        ],
    )
    assert [r["tag"] for r in tag_breakdown(combo)] == ["alpha", "beta", "zebra"]


def test_tag_breakdown_omits_untagged_cases():
    combo = _combo("m", "p", 0, 0, [_case([], True), _case(None, True)])
    assert tag_breakdown(combo) == []


def test_best_quality_breaks_ties_by_lower_cost():
    a = _combo("a", "p", 90.0, 0.50, [])
    b = _combo("b", "p", 90.0, 0.10, [])
    assert best_quality([a, b]) is b


def test_best_value_prefers_score_per_dollar():
    pricey = _combo("pricey", "p", 96.0, 0.45, [])
    cheap = _combo("cheap", "p", 88.0, 0.04, [])
    assert best_value([pricey, cheap]) is cheap


def test_best_value_falls_back_to_top_score_when_all_free():
    low = _combo("low", "p", 70.0, 0.0, [])
    high = _combo("high", "p", 95.0, 0.0, [])
    assert best_value([low, high]) is high


def test_summarize_gate_above_below_and_at_threshold():
    top = _combo("m", "p", 85.0, 0.0, [_case(["t"], True)])
    assert summarize([top], 80)["passed"] is True
    assert summarize([top], 85)["passed"] is True  # at threshold passes
    assert summarize([top], 90)["passed"] is False


def test_summarize_no_threshold_passes():
    top = _combo("m", "p", 10.0, 0.0, [_case(["t"], False)])
    s = summarize([top], None)
    assert s["passed"] is True
    assert s["warnings"] == ["t"]


def test_summarize_reports_best_combo_error_total():
    top = {
        "model": "m",
        "prompt_id": "p",
        "score": 100.0,
        "total_cost": 0.0,
        "errors": 2,
        "test_cases": [_case(["t"], True)],
    }
    assert summarize([top], None)["errors"] == 2


def test_summarize_empty_results_is_none():
    assert summarize([], 90) is None
