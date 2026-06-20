"""Tests for the run-to-run diff: matching, deltas, missing combos, schema."""

from __future__ import annotations

import json

from evalblink import compare


def _combo(model, prompt_id, score, cost):
    return {"model": model, "prompt_id": prompt_id, "score": score, "total_cost": cost}


def _record(run_id, combos, schema_version=1):
    return {
        "schema_version": schema_version,
        "run_id": run_id,
        "results": combos,
    }


def test_diff_computes_score_and_cost_deltas_for_matched_combos():
    a = _record("run_a", [_combo("m/x", "v1", 80.0, 0.10)])
    b = _record("run_b", [_combo("m/x", "v1", 86.0, 0.12)])
    result = compare.diff(a, b)

    [row] = result["rows"]
    assert row["present_in"] == "both"
    assert row["score_delta"] == 6.0
    assert round(row["cost_delta"], 4) == 0.02
    assert result["run_a"] == "run_a" and result["run_b"] == "run_b"
    assert "1/1" in result["verdict"] or "all 1" in result["verdict"]


def test_diff_flags_combos_present_in_only_one_run():
    a = _record("run_a", [_combo("m/x", "v1", 80.0, 0.10)])
    b = _record(
        "run_b",
        [_combo("m/x", "v1", 80.0, 0.10), _combo("m/y", "v1", 90.0, 0.20)],
    )
    rows = {(r["model"], r["prompt_id"]): r for r in compare.diff(a, b)["rows"]}

    assert rows[("m/x", "v1")]["present_in"] == "both"
    only = rows[("m/y", "v1")]
    assert only["present_in"] == "b_only"
    assert only["score_a"] is None
    assert only["score_delta"] is None


def test_load_record_defaults_missing_schema_version_to_zero(tmp_path):
    path = tmp_path / "legacy.json"
    # A legacy record written before schema_version existed.
    path.write_text(json.dumps({"run_id": "old", "results": []}))
    record = compare.load_record(str(path))
    assert record["schema_version"] == 0


def test_diff_notes_version_mismatch_but_still_compares():
    a = _record("run_a", [_combo("m/x", "v1", 80.0, 0.10)], schema_version=0)
    b = _record("run_b", [_combo("m/x", "v1", 90.0, 0.10)], schema_version=1)
    result = compare.diff(a, b)

    assert result["version_note"] is not None
    assert "v0" in result["version_note"] and "v1" in result["version_note"]
    # Mismatch must not stop the comparison.
    assert result["rows"][0]["score_delta"] == 10.0


def test_diff_with_no_common_combos():
    a = _record("run_a", [_combo("m/x", "v1", 80.0, 0.10)])
    b = _record("run_b", [_combo("m/y", "v2", 90.0, 0.10)])
    result = compare.diff(a, b)
    assert "No combos in common" in result["verdict"]
    assert {r["present_in"] for r in result["rows"]} == {"a_only", "b_only"}


# --- test-level diff (--detailed) -------------------------------------------


def _case(case_id, result, score, tags=None):
    """A test-case result dict; ``score=None`` models a pipeline (judge) error."""
    return {
        "id": case_id,
        "tags": tags or [],
        "match_result": result,
        "match_score": score,
    }


def _combo_tc(model, prompt_id, cases, score=0.0, cost=0.0):
    return {
        "model": model,
        "prompt_id": prompt_id,
        "score": score,
        "total_cost": cost,
        "test_cases": cases,
    }


def test_classify_case_covers_every_transition():
    cl = compare._classify_case
    assert cl(_case("c", True, 1.0), _case("c", False, 0.0)) == "regressed"
    assert cl(_case("c", False, 0.0), _case("c", True, 1.0)) == "improved"
    assert cl(_case("c", True, 1.0), _case("c", True, 1.0)) == "unchanged"
    assert cl(_case("c", True, 1.0), _case("c", False, None)) == "new_error"
    assert cl(_case("c", False, None), _case("c", True, 1.0)) == "recovered"
    assert cl(_case("c", False, None), _case("c", False, None)) == "still_error"


def test_new_error_is_not_counted_as_a_regression():
    # A pass that becomes a judge error is a pipeline failure, never a real fail.
    a = _combo_tc("m/x", "v1", [_case("c1", True, 1.0)])
    b = _combo_tc("m/x", "v1", [_case("c1", False, None)])
    cd = compare.case_diff(a, b)
    assert cd["rollup"]["new_error"] == 1
    assert cd["rollup"]["regressed"] == 0


def test_case_diff_handles_added_and_removed_cases():
    a = _combo_tc("m/x", "v1", [_case("c1", True, 1.0)])
    b = _combo_tc("m/x", "v1", [_case("c2", True, 1.0)])
    cd = compare.case_diff(a, b)
    by_id = {r["id"]: r for r in cd["rows"]}
    assert by_id["c1"]["transition"] == "removed"
    assert by_id["c2"]["transition"] == "added"
    assert cd["rollup"]["removed"] == 1 and cd["rollup"]["added"] == 1


def test_detailed_diff_aggregates_totals_worst_and_tags():
    # conv_002 regresses in BOTH combos; conv_001 improves in one.
    a = _record(
        "run_a",
        [
            _combo_tc(
                "m/x",
                "v1",
                [
                    _case("conv_001", False, 0.0, ["easy"]),
                    _case("conv_002", True, 1.0, ["edge_case"]),
                ],
            ),
            _combo_tc(
                "m/y",
                "v1",
                [_case("conv_002", True, 1.0, ["edge_case"])],
            ),
        ],
    )
    b = _record(
        "run_b",
        [
            _combo_tc(
                "m/x",
                "v1",
                [
                    _case("conv_001", True, 1.0, ["easy"]),
                    _case("conv_002", False, 0.0, ["edge_case"]),
                ],
            ),
            _combo_tc(
                "m/y",
                "v1",
                [_case("conv_002", False, 0.0, ["edge_case"])],
            ),
        ],
    )
    result = compare.detailed_diff(a, b)
    summary = result["summary"]

    assert summary["improved"] == 1  # conv_001 in m/x
    assert summary["regressed"] == 2  # conv_002 in both combos
    assert summary["net"] == -1

    # conv_002 is the systematic regression: 2 of 2 combos.
    worst = {item["id"]: item["count"] for item in summary["worst_regressed"]}
    assert worst["conv_002"] == 2

    # Tag net: edge_case -2 (two regressions), easy +1 (one improvement).
    tag_net = {item["tag"]: item["net"] for item in summary["tag_net"]}
    assert tag_net["edge_case"] == -2
    assert tag_net["easy"] == 1
    # Worst-first ordering puts edge_case ahead of easy.
    assert summary["tag_net"][0]["tag"] == "edge_case"


def test_detailed_diff_only_includes_shared_combos():
    a = _record("run_a", [_combo_tc("m/x", "v1", [_case("c1", True, 1.0)])])
    b = _record(
        "run_b",
        [
            _combo_tc("m/x", "v1", [_case("c1", True, 1.0)]),
            _combo_tc("m/y", "v1", [_case("c1", True, 1.0)]),
        ],
    )
    result = compare.detailed_diff(a, b)
    keys = {(c["model"], c["prompt_id"]) for c in result["case_combos"]}
    assert keys == {("m/x", "v1")}  # m/y is b_only, gets no case diff
