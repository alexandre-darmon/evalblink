"""CLI dispatch: argparse routes subcommands to the right handler."""

from __future__ import annotations

import json

import pytest

from evalblink import main


def test_compare_subcommand_dispatches(monkeypatch):
    seen = {}

    def fake_render(diff_result, console=None):
        seen["rendered"] = diff_result

    monkeypatch.setattr(
        main.compare, "load_record", lambda p: {"run_id": p, "results": []}
    )
    monkeypatch.setattr(
        main.compare, "diff", lambda a, b: {"a": a["run_id"], "b": b["run_id"]}
    )
    monkeypatch.setattr(main.reporter, "render_comparison", fake_render)
    monkeypatch.setattr(main.sys, "argv", ["evalblink", "compare", "a.json", "b.json"])

    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 0
    assert seen["rendered"] == {"a": "a.json", "b": "b.json"}


def test_compare_detailed_flag_routes_to_detailed(monkeypatch):
    seen = {}

    monkeypatch.setattr(
        main.compare, "load_record", lambda p: {"run_id": p, "results": []}
    )
    monkeypatch.setattr(main.compare, "detailed_diff", lambda a, b: {"detailed": True})
    # The plain path must NOT be taken when --detailed is given.
    monkeypatch.setattr(
        main.compare,
        "diff",
        lambda a, b: pytest.fail("diff called for --detailed run"),
    )
    monkeypatch.setattr(
        main.reporter,
        "render_detailed",
        lambda result, console=None: seen.update(result=result),
    )
    monkeypatch.setattr(
        main.sys, "argv", ["evalblink", "compare", "a.json", "b.json", "--detailed"]
    )

    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 0
    assert seen["result"] == {"detailed": True}


def test_run_subcommand_dispatches(monkeypatch):
    calls = {}

    monkeypatch.setattr(main, "load_config", lambda p: {"name": "X", "config_path": p})
    monkeypatch.setattr(
        main.runner, "run", lambda config, verbose=False, use_cache=True: ([], "ts")
    )

    def fake_write(config, results, timestamp):
        calls["config"] = config
        return {"passed": True}

    monkeypatch.setattr(main.reporter, "write", fake_write)
    monkeypatch.setattr(main.sys, "argv", ["evalblink", "run", "bench.yaml"])

    with pytest.raises(SystemExit) as exc:
        main.main()
    # No quality_threshold → gate not enforced, exits 0.
    assert exc.value.code == 0
    assert calls["config"]["config_path"] == "bench.yaml"


def test_no_subcommand_errors(monkeypatch):
    monkeypatch.setattr(main.sys, "argv", ["evalblink"])
    with pytest.raises(SystemExit) as exc:
        main.main()
    # argparse exits 2 on a required-subcommand parse error.
    assert exc.value.code == 2


# ── cache stats ──────────────────────────────────────────────────────────────


def test_cache_stats_command(monkeypatch, capsys):
    monkeypatch.setattr(main.cache, "stats", lambda: {"entries": 7, "size_bytes": 3072})
    monkeypatch.setattr(main.sys, "argv", ["evalblink", "cache", "stats"])
    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "7" in out
    assert "3.0 KB" in out


# ── cache clear ──────────────────────────────────────────────────────────────


def test_cache_clear_without_yes_shows_count(monkeypatch, capsys):
    monkeypatch.setattr(main.cache, "stats", lambda: {"entries": 12, "size_bytes": 0})
    monkeypatch.setattr(main.sys, "argv", ["evalblink", "cache", "clear"])
    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "12" in out
    assert "--yes" in out


def test_cache_clear_with_yes_deletes(monkeypatch, capsys):
    cleared = {}
    monkeypatch.setattr(main.cache, "clear", lambda: cleared.update(called=True) or 5)
    monkeypatch.setattr(main.sys, "argv", ["evalblink", "cache", "clear", "--yes"])
    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 0
    assert cleared.get("called")
    assert "5" in capsys.readouterr().out


# ── report ───────────────────────────────────────────────────────────────────


def test_report_subcommand_dispatches(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        main.compare, "load_record", lambda p: {"run_id": p, "results": []}
    )
    monkeypatch.setattr(
        main.reporter, "write_from_record", lambda r: seen.update(record=r)
    )
    monkeypatch.setattr(main.sys, "argv", ["evalblink", "report", "results/run.json"])
    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 0
    assert seen["record"]["run_id"] == "results/run.json"


# ── history ───────────────────────────────────────────────────────────────────


def test_history_no_results_dir(monkeypatch, capsys):
    monkeypatch.setattr(main, "RESULTS_DIR", "/nonexistent/path/xyz")
    monkeypatch.setattr(main.sys, "argv", ["evalblink", "history"])
    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 0
    assert "No results directory" in capsys.readouterr().out


def test_history_shows_runs(monkeypatch, tmp_path):
    record = {
        "run_id": "2026-01-01_test",
        "benchmark": "Test Bench",
        "timestamp": "2026-01-01_120000",
        "results": [{"total_cost": 0.01}],
        "insights": {
            "best_quality": {"score": 90.0, "model": "m", "prompt_id": "p"},
            "errors": 0,
        },
    }
    (tmp_path / "run.json").write_text(json.dumps(record))
    monkeypatch.setattr(main, "RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(main.sys, "argv", ["evalblink", "history"])
    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 0


def test_history_skips_invalid_json(monkeypatch, tmp_path):
    (tmp_path / "bad.json").write_text("not valid json {{")
    monkeypatch.setattr(main, "RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(main.sys, "argv", ["evalblink", "history"])
    with pytest.raises(SystemExit) as exc:
        main.main()
    # Graceful exit (empty table message), not a crash.
    assert exc.value.code == 0
