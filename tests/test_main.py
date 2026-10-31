"""CLI dispatch: argparse routes subcommands to the right handler."""

from __future__ import annotations

import datetime
import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import dotenv
import httpx
import pytest
import yaml

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
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    monkeypatch.setattr(main, "load_config", lambda p: {"name": "X", "config_path": p})
    monkeypatch.setattr(main, "_validate_config", lambda c: ([], []))
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


@pytest.mark.parametrize("key", [None, "", "   "])
def test_run_without_api_key_exits_1_before_any_call(monkeypatch, capsys, key):
    # main() calls load_dotenv(), which would pick up a real .env — neutralise it.
    monkeypatch.setattr(main, "load_dotenv", lambda *a, **k: None)
    if key is None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENROUTER_API_KEY", key)
    monkeypatch.setattr(main, "load_config", lambda p: {"name": "X"})
    monkeypatch.setattr(main, "_validate_config", lambda c: ([], []))
    monkeypatch.setattr(
        main.runner, "run", lambda *a, **k: pytest.fail("runner.run called")
    )
    monkeypatch.setattr(
        main.openrouter,
        "openrouter_request",
        lambda *a, **k: pytest.fail("API called"),
    )
    monkeypatch.setattr(main.sys, "argv", ["evalblink", "run", "bench.yaml"])
    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 1
    assert "OPENROUTER_API_KEY is not set" in capsys.readouterr().err


def test_no_subcommand_prints_help_and_the_quickstart(monkeypatch, capsys):
    monkeypatch.setattr(main.sys, "argv", ["evalblink"])
    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "QUICKSTART" in out and "evalblink init" in out
    assert "OPENROUTER_API_KEY" in out


# --- run --dry-run --------------------------------------------------------


def test_dry_run_estimates_without_running(monkeypatch):
    seen = {}
    monkeypatch.setattr(main, "load_config", lambda p: {"name": "X"})
    monkeypatch.setattr(main, "_validate_config", lambda c: ([], []))
    monkeypatch.setattr(main.openrouter, "fetch_models", lambda client: {"m": {}})
    monkeypatch.setattr(
        main.estimate,
        "estimate",
        lambda config, meta: {"over_budget": False, "unknown_models": []},
    )
    monkeypatch.setattr(
        main.reporter, "render_estimate", lambda est: seen.update(rendered=est)
    )
    # The benchmark itself must never run in a dry-run.
    monkeypatch.setattr(
        main.runner,
        "run",
        lambda *a, **k: pytest.fail("runner.run called during --dry-run"),
    )
    monkeypatch.setattr(
        main.sys, "argv", ["evalblink", "run", "bench.yaml", "--dry-run"]
    )
    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 0
    assert seen["rendered"] == {"over_budget": False, "unknown_models": []}


def test_dry_run_exits_1_when_over_budget(monkeypatch):
    monkeypatch.setattr(main, "load_config", lambda p: {"name": "X"})
    monkeypatch.setattr(main, "_validate_config", lambda c: ([], []))
    monkeypatch.setattr(main.openrouter, "fetch_models", lambda client: {})
    monkeypatch.setattr(
        main.estimate, "estimate", lambda config, meta: {"over_budget": True}
    )
    monkeypatch.setattr(main.reporter, "render_estimate", lambda est: None)
    monkeypatch.setattr(
        main.sys, "argv", ["evalblink", "run", "bench.yaml", "--dry-run"]
    )
    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 1


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
    monkeypatch.setattr(
        main.cache, "clear", lambda include_catalog: cleared.update(called=True) or 5
    )
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


def test_history_shows_runs(monkeypatch, tmp_path, capsys):
    combo = {
        "model": "m",
        "prompt_id": "p",
        "score": 90.0,
        "success": 9,
        "total": 10,
        "total_prompt_tokens": 1,
        "total_completion_tokens": 1,
        "total_cost": 0.01,
        "test_cases": [],
    }
    record = {
        "run_id": "2026-01-01_test",
        "benchmark": "Test Bench",
        "timestamp": "2026-01-01_120000",
        "results": [combo],
    }
    (tmp_path / "run.json").write_text(json.dumps(record))
    monkeypatch.setattr(main, "RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(main.sys, "argv", ["evalblink", "history"])
    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Test Bench" in out and "90.0%" in out and "$0.010000" in out


def test_history_skips_invalid_json(monkeypatch, tmp_path):
    (tmp_path / "bad.json").write_text("not valid json {{")
    monkeypatch.setattr(main, "RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(main.sys, "argv", ["evalblink", "history"])
    with pytest.raises(SystemExit) as exc:
        main.main()
    # Graceful exit (empty table message), not a crash.
    assert exc.value.code == 0


# ── models ────────────────────────────────────────────────────────────────────

_FAKE_MODELS = {
    "anthropic/claude-3-haiku": {
        "prompt": 0.00000025,
        "completion": 0.00000125,
        "context_length": 200000,
    },
    "openai/gpt-4o": {
        "prompt": 0.000005,
        "completion": 0.000015,
        "context_length": 128000,
    },
    "mistralai/mistral-7b-instruct:free": {
        "prompt": 0.0,
        "completion": 0.0,
        "context_length": 32768,
    },
}


def test_models_lists_all(monkeypatch):
    monkeypatch.setattr(
        main.openrouter, "fetch_models", lambda client, use_cache=True: _FAKE_MODELS
    )
    monkeypatch.setattr(main.sys, "argv", ["evalblink", "models"])
    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 0


def test_models_provider_filter(monkeypatch, capsys):
    monkeypatch.setattr(
        main.openrouter, "fetch_models", lambda client, use_cache=True: _FAKE_MODELS
    )
    monkeypatch.setattr(
        main.sys, "argv", ["evalblink", "models", "--provider", "anthropic"]
    )
    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "anthropic/claude-3-haiku" in out
    assert "openai/gpt-4o" not in out


def test_models_free_filter(monkeypatch, capsys):
    monkeypatch.setattr(
        main.openrouter, "fetch_models", lambda client, use_cache=True: _FAKE_MODELS
    )
    monkeypatch.setattr(main.sys, "argv", ["evalblink", "models", "--free"])
    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "mistralai/mistral-7b-instruct:free" in out
    assert "anthropic/claude-3-haiku" not in out


def test_models_min_context_filter(monkeypatch, capsys):
    monkeypatch.setattr(
        main.openrouter, "fetch_models", lambda client, use_cache=True: _FAKE_MODELS
    )
    monkeypatch.setattr(
        main.sys, "argv", ["evalblink", "models", "--min-context", "100k"]
    )
    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "anthropic/claude-3-haiku" in out  # 200k passes
    assert "openai/gpt-4o" in out  # 128k passes
    assert "mistralai/mistral-7b-instruct:free" not in out  # 32k excluded


def test_models_no_match_exits_gracefully(monkeypatch, capsys):
    monkeypatch.setattr(
        main.openrouter, "fetch_models", lambda client, use_cache=True: _FAKE_MODELS
    )
    monkeypatch.setattr(
        main.sys, "argv", ["evalblink", "models", "--provider", "nonexistent"]
    )
    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 0
    assert "No models match" in capsys.readouterr().out


def test_models_no_cache_flag(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        main.openrouter,
        "fetch_models",
        lambda client, use_cache=True: seen.update(use_cache=use_cache) or _FAKE_MODELS,
    )
    monkeypatch.setattr(main.sys, "argv", ["evalblink", "models", "--no-cache"])
    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 0
    assert seen["use_cache"] is False


@pytest.mark.parametrize("value", ["notanumber", "xk", "1e400k", "nanm"])
def test_models_invalid_min_context_exits_cleanly(monkeypatch, capsys, value):
    monkeypatch.setattr(
        main.openrouter, "fetch_models", lambda client, use_cache=True: _FAKE_MODELS
    )
    monkeypatch.setattr(
        main.sys, "argv", ["evalblink", "models", "--min-context", value]
    )
    with pytest.raises(SystemExit) as exc:
        main.main()
    # An argparse usage error: exit 2 with a message, no traceback.
    assert exc.value.code == 2
    assert f"invalid value {value!r}" in capsys.readouterr().err


def test_parse_context_helper():
    assert main._parse_context("100k") == 100_000
    assert main._parse_context("8K") == 8_000
    assert main._parse_context("1m") == 1_000_000
    assert main._parse_context("4096") == 4096


# ── validate ──────────────────────────────────────────────────────────────────


def test_validate_subcommand_clean_config(monkeypatch, tmp_path):
    cfg = {
        "name": "T",
        "models": ["m"],
        "prompts": [{"id": "v1", "template": "Hi"}],
        "test_cases": [
            {"id": "tc1", "evaluation": "exact_match", "expected_output": "y"}
        ],
    }
    f = tmp_path / "ok.yaml"
    f.write_text(yaml.dump(cfg))
    monkeypatch.setattr(main.sys, "argv", ["evalblink", "validate", str(f)])
    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 0


def test_validate_subcommand_errors_exit_1(monkeypatch, tmp_path):
    cfg = {"name": "T", "models": ["m"], "prompts": [{"id": "v1", "template": "Hi"}]}
    f = tmp_path / "bad.yaml"
    f.write_text(yaml.dump(cfg))
    monkeypatch.setattr(main.sys, "argv", ["evalblink", "validate", str(f)])
    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 1


@pytest.mark.parametrize(
    "text, message", [("", "Config file is empty"), ("- a\n- b\n", "YAML mapping")]
)
@pytest.mark.parametrize("command", ["validate", "run"])
def test_empty_or_non_mapping_yaml_is_a_clean_error(
    monkeypatch, tmp_path, capsys, text, message, command
):
    f = tmp_path / "cfg.yaml"
    f.write_text(text)
    monkeypatch.setattr(
        main.runner, "run", lambda *a, **k: pytest.fail("runner.run called")
    )
    monkeypatch.setattr(main.sys, "argv", ["evalblink", command, str(f)])
    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 1
    out, err = capsys.readouterr()
    # validate's findings are its result; for run they are diagnostics.
    assert message in (out if command == "validate" else err)


# ── CLI-level runs through the real pipeline, fake HTTP (round 2 contract) ────


def _write_cli_config(tmp_path, **overrides) -> Path:
    cfg = {
        "name": "CLI",
        "models": ["m/a"],
        "concurrency": 1,  # deterministic order for the queued responses
        "prompts": [{"id": "v1", "template": "Q: {{ q }}"}],
        "test_cases": [
            {
                "id": f"c{i}",
                "evaluation": "exact_match",
                "expected_output": "yes",
                "variables": {"q": str(i)},
            }
            for i in (1, 2)
        ],
    }
    cfg.update(overrides)
    path = tmp_path / "bench.yaml"
    path.write_text(yaml.dump(cfg))
    return path


@pytest.fixture
def fake_http(monkeypatch, tmp_path):
    """Run from ``tmp_path`` with a key set and ``httpx.Client.post`` faked.

    ``install(*responses)`` queues ``httpx.Response``s / exceptions; the last
    one repeats.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    def install(*responses):
        calls = []

        def post(self, **kwargs):
            calls.append(kwargs)
            item = responses[min(len(calls), len(responses)) - 1]
            if isinstance(item, BaseException):
                raise item
            return item

        monkeypatch.setattr(httpx.Client, "post", post)
        return calls

    return install


def _ok(content):
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0},
        },
    )


def _run_cli(argv):
    with pytest.raises(SystemExit) as exc:
        main.main(argv)
    return exc.value.code


def test_run_prints_results_on_stdout_and_progress_on_stderr(
    fake_http, tmp_path, capsys
):
    fake_http(_ok("yes"), _ok("no"))
    cfg = _write_cli_config(tmp_path, quality_threshold=40)
    assert _run_cli(["run", str(cfg), "--no-cache"]) == 0
    out, err = capsys.readouterr()
    assert "[1/2] m/a / v1 / c1  pass" in err and "[2/2] m/a / v1 / c2  fail" in err
    # stdout: the table once, the saved files and the verdict — no diagnostics,
    # and no second per-combo summary.
    assert "[1/2]" not in out and "Running 2 cases" not in out
    assert "Quality score" not in out and "Total cost" not in out
    assert "PASS: best score meets quality_threshold (40)." in out


def test_run_prints_validation_warnings_on_stderr(fake_http, tmp_path, capsys):
    fake_http(_ok("yes"))
    cfg = _write_cli_config(
        tmp_path, prompts=[{"id": "v1", "template": "Q: {q} {{ q }}"}]
    )
    assert _run_cli(["run", str(cfg), "--no-cache"]) == 0
    out, err = capsys.readouterr()
    assert "single-brace" in err and "single-brace" not in out


@pytest.mark.parametrize(
    "failure",
    [
        httpx.Response(502, text="<html>502 Bad Gateway</html>"),
        httpx.Response(401, json={"error": {"code": 401, "message": "bad key"}}),
        httpx.Response(200, json={"choices": []}),
        httpx.ConnectError("refused"),
    ],
)
def test_run_where_nothing_could_be_scored_exits_1(
    fake_http, tmp_path, capsys, failure
):
    fake_http(failure)
    assert _run_cli(["run", str(_write_cli_config(tmp_path))]) == 1
    out, err = capsys.readouterr()
    assert "2 of 2 case(s) hit a pipeline error" in err
    assert "no case could be scored" in out
    # The run still completed and wrote its records.
    [md] = (tmp_path / "results").glob("*.md")
    assert "| c1 | exact_match | False | API error |" in md.read_text()


def test_run_with_some_pipeline_errors_warns_and_uses_the_gate(
    fake_http, tmp_path, capsys
):
    fake_http(_ok("yes"), httpx.Response(400, text="<html>bad</html>"))
    path = _write_cli_config(tmp_path)
    assert _run_cli(["run", str(path), "--no-cache"]) == 0  # no gate set
    out, err = capsys.readouterr()
    assert "1 of 2 case(s) hit a pipeline error" in err
    assert "no case could be scored" not in out

    # The scored case passes (100% of scored), but the gate is fail-safe: the
    # unscored case counts as a failure, 1/2 = 50% < 90.
    fake_http(_ok("yes"), httpx.Response(400, text="<html>bad</html>"))
    path = _write_cli_config(tmp_path, quality_threshold=90)
    assert _run_cli(["run", str(path), "--no-cache"]) == 1
    assert "1 case(s) that could not be scored count as failures" in (
        capsys.readouterr().out
    )

    # A gate the worst case still clears passes despite the error.
    fake_http(_ok("yes"), httpx.Response(400, text="<html>bad</html>"))
    path = _write_cli_config(tmp_path, quality_threshold=50)
    assert _run_cli(["run", str(path), "--no-cache"]) == 0


def test_run_without_pipeline_errors_prints_no_error_warning(
    fake_http, tmp_path, capsys
):
    fake_http(_ok("yes"))
    assert _run_cli(["run", str(_write_cli_config(tmp_path))]) == 0
    assert "pipeline error" not in capsys.readouterr().out


def test_validate_cli_rejects_template_that_fails_to_render(tmp_path, capsys):
    path = _write_cli_config(
        tmp_path, prompts=[{"id": "v1", "template": "Q: {{ q.a.b }}"}]
    )
    assert _run_cli(["validate", str(path)]) == 1
    out = capsys.readouterr().out
    assert "prompts['v1'] × test_cases['c1']" in out and "UndefinedError" in out


def test_dotenv_is_loaded_from_the_working_directory(monkeypatch, tmp_path):
    # Restore the real loader (conftest neutralises it) and start key-less.
    monkeypatch.setattr(main, "load_dotenv", dotenv.load_dotenv)
    monkeypatch.setenv("OPENROUTER_API_KEY", "placeholder")  # restored on teardown
    monkeypatch.delenv("OPENROUTER_API_KEY")
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=from-cwd-dotenv\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main.cache, "stats", lambda: {"entries": 0, "size_bytes": 0})
    assert _run_cli(["cache", "stats"]) == 0
    # Compare without echoing the value: a wrong .env could hold a real key.
    loaded_from_cwd = os.environ.get("OPENROUTER_API_KEY") == "from-cwd-dotenv"
    assert loaded_from_cwd, ".env in the working directory was not loaded"


@pytest.mark.parametrize("command", ["validate", "run"])
def test_unreadable_config_is_a_clean_error(tmp_path, command):
    bad = tmp_path / "latin1.yaml"
    bad.write_bytes(b"name: caf\xe9\n")
    for path, message in ((bad, "Invalid YAML"), (tmp_path, "Cannot read config")):
        code = _run_cli([command, str(path)])
        assert isinstance(code, str) and message in code


def test_dry_run_catalog_failure_is_a_clean_error(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no cached models.json here
    monkeypatch.setattr(
        httpx.Client, "get", lambda self, **k: httpx.Response(502, text="<html>")
    )
    code = _run_cli(["run", str(_write_cli_config(tmp_path)), "--dry-run"])
    assert isinstance(code, str) and "Could not fetch the OpenRouter model" in code


# ── round 3 ──────────────────────────────────────────────────────────────────


def test_markup_in_user_strings_is_printed_literally(fake_http, tmp_path, capsys):
    fake_http(_ok("yes"))
    path = _write_cli_config(
        tmp_path,
        name="[bold]v2 :smile:",
        models=["m/[/x]"],
        prompts=[{"id": "[/p]", "template": "Q: {{ q }}"}],
    )
    assert _run_cli(["run", str(path)]) == 0
    out = capsys.readouterr().out
    assert "[bold]v2 :smile:" in out and "[/p]" in out and "m/[/x]" in out


def test_validate_prints_markup_in_errors_literally(tmp_path, capsys):
    path = _write_cli_config(
        tmp_path,
        test_cases=[
            {"id": "[/x]", "evaluation": "exact_match", "expected_output": "y"},
            {"id": "[/x]", "evaluation": "exact_match", "expected_output": "y"},
        ],
    )
    assert _run_cli(["validate", str(path)]) == 1
    assert "Duplicate test_case id: '[/x]'" in capsys.readouterr().out


def test_surrogates_in_config_and_output_do_not_crash_run(fake_http, tmp_path, capsys):
    body = (
        b'{"choices":[{"message":{"content":"y\\ud800"}}],'
        b'"usage":{"prompt_tokens":1,"completion_tokens":1,"cost":0}}'
    )
    fake_http(httpx.Response(200, content=body))
    path = tmp_path / "s.yaml"
    path.write_text(
        _write_cli_config(tmp_path)
        .read_text()
        .replace("name: CLI", 'name: "CLI \\ud800"')
        .replace("id: c1", 'id: "c1\\udfff"')
    )
    assert "\\ud800" in path.read_text()
    assert _run_cli(["run", str(path)]) in (0, 1)
    [record] = (tmp_path / "results").glob("*.json")
    assert _run_cli(["report", str(record)]) == 0


def _run_two_prompts_one_failing(fake_http, tmp_path):
    # concurrency 1: v1/c1, v1/c2 succeed; v2/c1, v2/c2 hit an API error.
    fake_http(_ok("yes"), _ok("yes"), httpx.Response(400, text="<html>bad</html>"))
    path = _write_cli_config(
        tmp_path,
        prompts=[
            {"id": "v1", "template": "Q: {{ q }}"},
            {"id": "v2", "template": "Q2: {{ q }}"},
        ],
    )
    return _run_cli(["run", str(path)])


def test_insights_errors_report_the_run_total(fake_http, tmp_path, capsys):
    assert _run_two_prompts_one_failing(fake_http, tmp_path) == 0
    out, err = capsys.readouterr()
    assert "2 of 4 case(s) hit a pipeline error" in err
    assert "2 pipeline error(s) excluded from scoring." in out  # not 0 (best combo)
    [md] = (tmp_path / "results").glob("*.md")
    assert "**2 pipeline error(s)** excluded" in md.read_text()


def test_history_errors_column_is_the_run_total(fake_http, tmp_path, capsys):
    _run_two_prompts_one_failing(fake_http, tmp_path)
    capsys.readouterr()
    [record] = (tmp_path / "results").glob("*.json")
    data = json.loads(record.read_text())
    data["insights"]["errors"] = 0  # an old record's best-combo count
    record.write_text(json.dumps(data))
    assert _run_cli(["history"]) == 0
    row = [line for line in capsys.readouterr().out.splitlines() if "$" in line]
    assert row and row[0].rstrip(" │").endswith("2")


@pytest.mark.parametrize(
    "content, reason",
    [
        ("not json", "not a valid JSON run record"),
        ("# Report\n", "not a valid JSON run record"),
        ("[1]", "expected a JSON object"),
        ("{}", "'benchmark' is missing"),
        (b"\xff\xfe", "not a valid JSON run record"),
        (
            json.dumps(
                {
                    "benchmark": "b",
                    "timestamp": "t",
                    "results": [
                        {
                            "model": "m",
                            "prompt_id": "p",
                            "score": 1,
                            "success": 1,
                            "total": 1,
                            "total_prompt_tokens": 1,
                            "total_completion_tokens": 1,
                            "total_cost": None,
                            "test_cases": [],
                        }
                    ],
                }
            ),
            "total_cost is missing or not a finite number",
        ),
        (None, "Cannot read run record"),  # a directory
    ],
)
def test_bad_results_file_is_a_clean_error(
    monkeypatch, tmp_path, capsys, content, reason
):
    path = tmp_path / "r.json"
    if content is None:
        path.mkdir()
    elif isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content)
    for argv in (["report", str(path)], ["compare", str(path), str(path)]):
        code = _run_cli(argv)
        assert isinstance(code, str) and reason in code, (argv, code)
    # history skips it with a one-line warning instead of failing.
    (tmp_path / "results").mkdir()
    os.replace(path, tmp_path / "results" / "r.json")
    monkeypatch.chdir(tmp_path)
    assert _run_cli(["history"]) == 0
    out = capsys.readouterr().out
    assert "Skipping" in out and reason in out


def test_missing_results_file_is_a_clean_error(tmp_path):
    code = _run_cli(["report", str(tmp_path / "nope.json")])
    assert isinstance(code, str) and "Run record not found" in code


def test_models_shows_unknown_pricing_as_unknown(monkeypatch, capsys):
    catalog = {
        "openrouter/auto": {"prompt": None, "completion": None, "context_length": None}
    }
    monkeypatch.setattr(
        main.openrouter, "fetch_models", lambda client, use_cache=True: catalog
    )
    assert _run_cli(["models"]) == 0
    row = [line for line in capsys.readouterr().out.splitlines() if "auto" in line]
    assert "?" in row[0] and "free" not in row[0]
    assert _run_cli(["models", "--free"]) == 0  # unknown is not free
    assert "No models match" in capsys.readouterr().out


def test_report_recomputes_insights_from_results(fake_http, tmp_path, capsys):
    _run_two_prompts_one_failing(fake_http, tmp_path)
    [record] = (tmp_path / "results").glob("*.json")
    data = json.loads(record.read_text())
    data["insights"] = {"errors": 0}  # an old/partial record's stored insights
    record.write_text(json.dumps(data))
    for md in (tmp_path / "results").glob("*.md"):
        md.unlink()
    assert _run_cli(["report", str(record)]) == 0
    [md] = (tmp_path / "results").glob("*.md")
    assert "**2 pipeline error(s)** excluded" in md.read_text()


# ── round 4: environment, filesystem and signal paths ─────────────────────────


def test_non_ascii_api_key_is_a_clean_api_error(fake_http, tmp_path, monkeypatch):
    calls = fake_http(_ok("yes"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-abc…")  # pasted "…"
    assert _run_cli(["run", str(_write_cli_config(tmp_path))]) == 1
    assert calls == []  # rejected before httpx builds the header
    [record] = (tmp_path / "results").glob("*.json")
    case = json.loads(record.read_text())["results"][0]["test_cases"][0]
    assert case["status"] == "api_error" and "non-ASCII" in case["error"]
    code = _run_cli(["models", "--no-cache"])
    assert isinstance(code, str) and "non-ASCII" in code


@pytest.mark.parametrize(
    "raw", ["OPENROUTER_API_KEY=x\r\n".encode("utf-16"), b"# caf\xe9\nA=1\n"]
)
def test_non_utf8_dotenv_warns_instead_of_crashing(monkeypatch, tmp_path, capsys, raw):
    monkeypatch.setattr(main, "load_dotenv", dotenv.load_dotenv)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_bytes(raw)
    assert _run_cli(["cache", "stats"]) == 0
    assert "Ignoring .env" in capsys.readouterr().err


def test_dotenv_with_bom_still_sets_the_key(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "load_dotenv", dotenv.load_dotenv)
    monkeypatch.setenv("OPENROUTER_API_KEY", "placeholder")
    monkeypatch.delenv("OPENROUTER_API_KEY")
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_bytes(b"\xef\xbb\xbfOPENROUTER_API_KEY=bom-key\r\n")
    assert _run_cli(["cache", "stats"]) == 0
    assert os.environ.get("OPENROUTER_API_KEY") == "bom-key"


def test_ctrl_c_cancels_queued_calls_and_exits_130(fake_http, tmp_path, monkeypatch):
    fake_http(_ok("y"))
    calls = []

    def post(self, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise KeyboardInterrupt  # what the main thread sees on Ctrl-C
        threading.Event().wait(2)  # in flight while the run stops (sleep is stubbed)
        return _ok("y")

    monkeypatch.setattr(httpx.Client, "post", post)

    def hard_exit(code):  # main() uses os._exit so it never joins the workers
        raise SystemExit(code)

    monkeypatch.setattr(main.os, "_exit", hard_exit)
    cases = [
        {
            "id": f"c{i}",
            "evaluation": "exact_match",
            "expected_output": "y",
            "variables": {"q": str(i)},  # distinct: no cache hits,
        }
        for i in range(20)
    ]
    path = _write_cli_config(tmp_path, test_cases=cases)
    started = time.monotonic()
    assert _run_cli(["run", str(path)]) == 130
    assert time.monotonic() - started < 1.5  # did not wait for the in-flight call
    assert len(calls) <= 2  # queued calls were dropped (the worker may race in one)
    assert list((tmp_path / "results").glob("*.json")) == []


def test_unwritable_results_dir_fails_before_any_call(fake_http, tmp_path):
    calls = fake_http(_ok("yes"))
    (tmp_path / "results").write_text("a file, not a directory")
    code = _run_cli(["run", str(_write_cli_config(tmp_path))])
    assert isinstance(code, str) and "results/" in code
    assert calls == []


def test_two_runs_in_the_same_second_keep_both_records(
    fake_http, tmp_path, monkeypatch
):
    fake_http(_ok("yes"))
    fixed = datetime.datetime(2026, 1, 1, 12, 0, 0)
    fake = SimpleNamespace(datetime=SimpleNamespace(now=lambda: fixed))
    monkeypatch.setattr(main.runner, "datetime", fake)
    path = _write_cli_config(tmp_path)
    assert _run_cli(["run", str(path)]) == 0
    assert _run_cli(["run", str(path)]) == 0
    names = sorted(p.name for p in (tmp_path / "results").glob("*.json"))
    assert names == ["2026-01-01_120000_cli-2.json", "2026-01-01_120000_cli.json"]


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="POSIX, non-root")
def test_read_only_results_dir_fails_before_any_call(fake_http, tmp_path):
    calls = fake_http(_ok("yes"))
    (tmp_path / "results").mkdir()
    (tmp_path / "results").chmod(0o555)
    try:
        code = _run_cli(["run", str(_write_cli_config(tmp_path))])
    finally:
        (tmp_path / "results").chmod(0o755)
    assert isinstance(code, str) and "Cannot write to results/" in code
    assert calls == []


@pytest.mark.parametrize(
    "command", [["history"], ["cache", "stats"], ["cache", "clear", "--yes"]]
)
def test_unreadable_directories_are_clean_errors(monkeypatch, tmp_path, command):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    (tmp_path / ".evalblink_cache").mkdir()

    def denied(path):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(os, "listdir", denied)
    code = _run_cli(command)
    assert isinstance(code, str) and "Permission denied" in code


def test_cache_clear_skips_a_directory_named_like_an_entry(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".evalblink_cache" / "sub.json").mkdir(parents=True)
    (tmp_path / ".evalblink_cache" / "k.json").write_text("{}")
    assert _run_cli(["cache", "clear", "--yes"]) == 0
    assert [p.name for p in (tmp_path / ".evalblink_cache").iterdir()] == ["sub.json"]


def test_cache_keeps_the_model_catalog_apart_from_responses(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.chdir(tmp_path)
    cache_dir = tmp_path / ".evalblink_cache"
    cache_dir.mkdir()
    (cache_dir / "k.json").write_text("{}")
    (cache_dir / "models.json").write_text("{}")
    assert _run_cli(["cache", "stats"]) == 0
    out = capsys.readouterr().out
    assert "Cached responses : 1" in out and "Model catalog    : cached" in out
    assert _run_cli(["cache", "clear"]) == 0
    assert "1 cached response." in capsys.readouterr().out
    assert _run_cli(["cache", "clear", "--yes"]) == 0
    assert "Cleared 1 cache file." in capsys.readouterr().out
    assert [p.name for p in cache_dir.iterdir()] == ["models.json"]
    assert _run_cli(["cache", "clear", "--yes", "--all"]) == 0
    assert list(cache_dir.iterdir()) == []


def _good_record(**case):
    tc = {"id": "c1", "match_score": 1.0, "match_result": True, "total_cost": 0.0}
    tc.update(case)
    return {
        "benchmark": "B",
        "timestamp": "2026-01-01_000000",
        "results": [
            {
                "model": "a/b",
                "prompt_id": "v1",
                "score": 100.0,
                "total_cost": 0.0,
                "success": 1,
                "total": 1,
                "total_prompt_tokens": 1,
                "total_completion_tokens": 1,
                "test_cases": [tc],
            }
        ],
    }


@pytest.mark.parametrize(
    "record",
    [
        {k: v for k, v in _good_record().items()} | {"run_id": "a\x00b"},
        {k: v for k, v in _good_record().items()} | {"run_id": "x" * 300},
    ],
)
def test_report_with_unusable_run_id_is_a_clean_error(monkeypatch, tmp_path, record):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "r.json"
    path.write_text(json.dumps(record))
    code = _run_cli(["report", str(path)])
    assert isinstance(code, str) and "Could not write the report" in code


@pytest.mark.parametrize("drop", ["match_score", "match_result"])
def test_record_missing_case_fields_is_rejected_by_report_and_history(
    monkeypatch, tmp_path, capsys, drop
):
    monkeypatch.chdir(tmp_path)
    record = _good_record(tags=["t"])
    del record["results"][0]["test_cases"][0][drop]
    (tmp_path / "results").mkdir()
    path = tmp_path / "results" / "x.json"
    path.write_text(json.dumps(record))
    code = _run_cli(["report", str(path)])
    assert isinstance(code, str) and drop in code
    assert _run_cli(["history"]) == 0
    assert "Skipping" in capsys.readouterr().out


def test_deeply_nested_yaml_is_invalid_yaml(tmp_path):
    path = tmp_path / "deep.yaml"
    path.write_text("name: " + "[" * 5000 + "]" * 5000 + "\n")
    code = _run_cli(["validate", str(path)])
    assert isinstance(code, str) and "nested too deeply" in code


def test_judge_api_error_is_warned_and_recorded(fake_http, tmp_path, capsys):
    cfg = _write_cli_config(
        tmp_path,
        evaluation={"judge_model": "j/x"},
        test_cases=[
            {
                "id": "c1",
                "evaluation": "llm_judge",
                "criteria": "good",
                "variables": {"q": "1"},
            }
        ],
    )
    fake_http(
        _ok("an answer"),
        httpx.Response(404, json={"error": {"code": 404, "message": "no judge"}}),
    )
    assert _run_cli(["run", str(cfg), "--no-cache"]) == 1  # nothing scored
    assert "[1/1] m/a / v1 / c1  ⚠ judge API error: " in capsys.readouterr().err
    [record] = (tmp_path / "results").glob("*.json")
    case = json.loads(record.read_text())["results"][0]["test_cases"][0]
    assert case["status"] == "judge_api_error" and "no judge" in case["error"]


# ── run / validate without a path ─────────────────────────────────────────────


def _bench_dir(tmp_path, *names):
    (tmp_path / "benchmarks").mkdir(exist_ok=True)
    for name in names:
        path = _write_cli_config(tmp_path)
        path.rename(tmp_path / "benchmarks" / name)


@pytest.mark.parametrize("command", ["validate", "run"])
def test_no_path_uses_the_only_yaml_in_benchmarks(
    monkeypatch, tmp_path, capsys, command
):
    monkeypatch.chdir(tmp_path)
    _bench_dir(tmp_path, "mine.yaml")
    (tmp_path / "benchmarks" / "notes.txt").write_text("not a config")
    (tmp_path / "benchmarks" / "dir.yaml").mkdir()  # not a file
    (tmp_path / "benchmarks" / "._mine.yaml").write_text("")  # macOS junk
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    _run_cli([command])  # run stops at the missing key, after loading the config
    captured = capsys.readouterr()
    assert f"Using {os.path.join('benchmarks', 'mine.yaml')}" in captured.err
    assert "Config not found" not in captured.out + captured.err


def test_no_path_with_several_configs_lists_them(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _bench_dir(tmp_path, "a.yaml", "b.yml")
    code = _run_cli(["validate"])
    assert isinstance(code, str) and "Several configs" in code
    assert "benchmarks/a.yaml" in code and "benchmarks/b.yml" in code


def test_no_path_in_the_repo_keeps_the_example_default(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    _bench_dir(tmp_path, "a.yaml", "exact_match_classification.yaml")
    assert _run_cli(["validate"]) == 0
    assert f"Using {main.DEFAULT_CONFIG}" in capsys.readouterr().err


def test_no_path_and_no_benchmarks_suggests_init(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    code = _run_cli(["validate"])
    assert isinstance(code, str) and "evalblink init" in code


# ── models --text ─────────────────────────────────────────────────────────────


def test_models_text_drops_only_known_non_text_models(monkeypatch, capsys):
    catalog = {
        "t/known": {"prompt": 0.0, "completion": 0.0, "text": True},
        "t/unknown": {"prompt": 0.0, "completion": 0.0, "text": None},
        "a/music": {"prompt": 0.0, "completion": 0.0, "text": False},
    }
    monkeypatch.setattr(
        main.openrouter, "fetch_models", lambda client, use_cache=True: catalog
    )
    assert _run_cli(["models", "--free", "--text"]) == 0
    out = capsys.readouterr().out
    assert "t/known" in out and "t/unknown" in out and "a/music" not in out


# ── run filters ───────────────────────────────────────────────────────────────


def _filter_config(tmp_path):
    return _write_cli_config(
        tmp_path,
        models=["m/a", "m/b"],
        prompts=[
            {"id": "v1", "template": "Q: {{ q }}"},
            {"id": "v2", "template": "{{ q }}?"},
        ],
        test_cases=[
            {
                "id": f"c{i}",
                "evaluation": "exact_match",
                "expected_output": "yes",
                "variables": {"q": str(i)},
                "tags": ["odd" if i % 2 else "even"],
            }
            for i in range(1, 6)
        ],
        quality_threshold=50,
        gate={"model": "m/b"},
    )


def test_run_filters_narrow_the_matrix(fake_http, tmp_path, capsys):
    calls = fake_http(_ok("yes"))
    path = _filter_config(tmp_path)
    argv = ["run", str(path), "-m", "m/a", "-p", "v2", "--tag", "odd", "-n", "2"]
    assert _run_cli(argv) == 0
    [record] = (tmp_path / "results").glob("*.json")
    assert json.loads(record.read_text())["filters"] == {
        "model": ["m/a"],
        "prompt": ["v2"],
        "tag": ["odd"],
        "limit": 2,
    }
    assert len(calls) == 2  # 1 model × 1 prompt × 2 cases
    assert {c["json"]["model"] for c in calls} == {"m/a"}
    err = capsys.readouterr().err
    assert "Filtered: 1 model(s) × 1 prompt(s) × 2 case(s)" in err
    assert "gate ignored" in err  # the gate's m/b is filtered out


@pytest.mark.parametrize(
    "argv, message",
    [
        (["-m", "m/zzz"], "No model 'm/zzz' in the config. Available: m/a, m/b"),
        (["-p", "v9"], "No prompt 'v9' in the config. Available: v1, v2"),
        (["--tag", "nope"], "No test case has tag 'nope'."),
    ],
)
def test_run_filters_name_what_exists(fake_http, tmp_path, argv, message):
    fake_http(_ok("yes"))
    code = _run_cli(["run", str(_filter_config(tmp_path)), *argv])
    assert code == message


@pytest.mark.parametrize("value", ["0", "-1", "two"])
def test_limit_must_be_positive(tmp_path, value, capsys):
    assert _run_cli(["run", "x.yaml", "--limit", value]) == 2
    assert "use 1 or more" in capsys.readouterr().err


def test_gate_names_its_combo_in_the_verdict(fake_http, tmp_path, capsys):
    fake_http(_ok("yes"))
    path = _write_cli_config(
        tmp_path, models=["m/a", "m/b"], quality_threshold=50, gate={"model": "m/b"}
    )
    assert _run_cli(["run", str(path)]) == 0
    assert "PASS: m/b score meets quality_threshold (50)." in capsys.readouterr().out


# ── dry-run: unknown model ────────────────────────────────────────────────────


def test_dry_run_fails_on_a_model_missing_from_the_catalog(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.chdir(tmp_path)
    catalog = {"openai/gpt-4o-mini": {"prompt": 1e-7, "completion": 1e-7}}
    monkeypatch.setattr(
        main.openrouter, "fetch_models", lambda client, use_cache=True: catalog
    )
    path = _write_cli_config(tmp_path, models=["openai/gpt-4o-mni"])
    assert _run_cli(["run", str(path), "--dry-run"]) == 1
    out = capsys.readouterr().out
    assert "Unknown model 'openai/gpt-4o-mni'" in out
    assert "did you mean 'openai/gpt-4o-mini'?" in out
    path = _write_cli_config(tmp_path, models=["openai/gpt-4o-mini"])
    assert _run_cli(["run", str(path), "--dry-run"]) == 0


# ── compare / report without paths ────────────────────────────────────────────


def _results_dir(tmp_path, *runs):
    """Write run records: (run_id, benchmark, passed, errors[, filters]) each,
    over the same 10 cases; the run_id's first 17 characters are its timestamp."""
    (tmp_path / "results").mkdir(exist_ok=True)
    for i, (run_id, benchmark, passed, errors, *filters) in enumerate(runs):
        cases = [
            {
                "id": f"c{n}",
                "match_score": None if n < errors else float(n < errors + passed),
                "match_result": errors <= n < errors + passed,
                "total_cost": 0.0,
            }
            for n in range(10)
        ]
        scored = 10 - errors
        record = {
            "run_id": run_id,
            "benchmark": benchmark,
            "timestamp": run_id[:17],
            "filters": filters[0] if filters else None,
            "results": [
                {
                    "model": "a/b",
                    "prompt_id": "v1",
                    "score": passed / scored * 100 if scored else 0.0,
                    "success": passed,
                    "total": 10,
                    "errors": errors,
                    "total_cost": 0.0,
                    "total_prompt_tokens": 1,
                    "total_completion_tokens": 1,
                    "test_cases": cases,
                }
            ],
        }
        path = tmp_path / "results" / f"{run_id}.json"
        path.write_text(json.dumps(record))
        os.utime(path, (1_000_000 + i, 1_000_000 + i))  # written in this order


def test_compare_without_paths_diffs_the_two_latest_runs_of_a_benchmark(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.chdir(tmp_path)
    _results_dir(
        tmp_path,
        ("2026-01-01_000000_b", "B", 9, 0),
        ("2026-01-02_000000_b", "B", 8, 0),
        ("2026-01-03_000000_other", "Other", 1, 0),
        ("2026-01-04_000000_b", "B", 7, 0),
    )
    # Latest run is B's 01-04; it is compared with B's previous one, 01-02.
    assert _run_cli(["compare"]) == 0
    assert "2026-01-02_000000_b.json → results" in capsys.readouterr().err
    assert _run_cli(["compare", "--fail-on-regression", "0"]) == 1
    assert "REGRESSION: a/b / v1 pass rate -10.0 points" in capsys.readouterr().out
    assert (
        _run_cli(["compare", "--fail-on-regression", "10"]) == 0
    )  # not *more* than 10
    assert "No regression beyond 10 point(s)." in capsys.readouterr().out


def test_one_run_is_the_baseline_for_the_latest_run(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "kept").mkdir()
    _results_dir(
        tmp_path,
        ("2026-01-01_000000_b", "B", 9, 0),
        ("2026-01-02_000000_b", "B", 3, 0),
    )
    baseline = tmp_path / "kept" / "baseline.json"
    (tmp_path / "results" / "2026-01-01_000000_b.json").rename(baseline)
    assert _run_cli(["compare", str(baseline), "--fail-on-regression", "5"]) == 1
    assert "baseline.json → results/2026-01-02_000000_b.json" in capsys.readouterr().err


def test_a_run_that_breaks_with_new_errors_is_a_regression(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _results_dir(
        tmp_path,
        ("2026-01-01_000000_b", "B", 5, 0),  # 5/10
        ("2026-01-02_000000_b", "B", 5, 3),  # 5/7 scored (better), 5/10 fail-safe
        ("2026-01-03_000000_b", "B", 2, 6),  # 2/4 scored (same), 2/10 fail-safe
    )
    assert (
        _run_cli(["compare", "2026-01-01", "2026-01-02", "--fail-on-regression", "0"])
        == 0
    )
    assert _run_cli(["compare", "--fail-on-regression", "0"]) == 1


def test_the_regression_line_shows_the_rate_the_check_uses(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.chdir(tmp_path)
    _results_dir(
        tmp_path,
        ("2026-01-01_000000_b", "B", 5, 0),  # 50 % scored, 50 % fail-safe
        ("2026-01-02_000000_b", "B", 2, 6),  # 50 % scored, 20 % fail-safe
    )
    assert _run_cli(["compare", "--fail-on-regression", "0"]) == 1
    assert "REGRESSION: a/b / v1 pass rate -30.0 points" in capsys.readouterr().out


def test_slices_are_only_compared_with_the_same_slice(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    _results_dir(
        tmp_path,
        ("2026-01-01_000000_b", "B", 9, 0),
        ("2026-01-02_000000_b", "B", 2, 0, {"tag": ["hard"]}),
        ("2026-01-03_000000_b", "B", 9, 0),
        ("2026-01-04_000000_b", "B", 1, 0, {"tag": ["hard"]}),
    )
    assert _run_cli(["compare"]) == 0  # the two "hard" slices
    assert "2026-01-02_000000_b.json → results/2026-01-04" in capsys.readouterr().err
    # Explicitly comparing a slice with a full run: the cases differ.
    path = tmp_path / "results" / "2026-01-04_000000_b.json"
    record = json.loads(path.read_text())
    record["results"][0]["test_cases"] = record["results"][0]["test_cases"][:3]
    path.write_text(json.dumps(record))
    argv = ["compare", "2026-01-03", "2026-01-04", "--fail-on-regression", "0"]
    assert _run_cli(argv) == 1  # nothing comparable: fail closed
    out = " ".join(capsys.readouterr().out.split())  # Rich wraps the verdict
    assert "│ different test cases" in out and "they ran different test cases" in out
    assert "FAIL: nothing to compare" in out


def test_same_second_reruns_order_by_their_number(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    _results_dir(
        tmp_path,
        ("2026-01-01_000000_b-2", "B", 3, 0),  # same second, the later run…
        ("2026-01-01_000000_b", "B", 9, 0),  # …though restored (copied) last
    )
    assert _run_cli(["compare", "--fail-on-regression", "0"]) == 1
    assert (
        "000000_b.json → results/2026-01-01_000000_b-2.json" in capsys.readouterr().err
    )
    assert _run_cli(["report", "2026-01-01_000000_b"]) == 0  # exact id, not ambiguous


def test_compare_accepts_run_id_prefixes(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _results_dir(
        tmp_path,
        ("2026-01-01_000000_b", "B", 5, 0),
        ("2026-01-02_000000_b", "B", 9, 0),
        ("2026-01-02_120000_b", "B", 9, 0),
    )
    argv = ["compare", "2026-01-01", "2026-01-02_12", "--fail-on-regression", "0"]
    assert _run_cli(argv) == 0
    code = _run_cli(["compare", "2026-01-01", "2026-01-02"])
    assert isinstance(code, str) and "matches several runs" in code
    for missing in ("nope", "_b"):  # a prefix, not a substring
        code = _run_cli(["compare", "2026-01-01", missing])
        assert isinstance(code, str) and "not found" in code.lower()


def test_fail_on_regression_compares_only_shared_combos(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _results_dir(
        tmp_path,
        ("2026-01-01_000000_b", "B", 9, 0),
        ("2026-01-02_000000_b", "B", 9, 0),
    )
    path = tmp_path / "results" / "2026-01-02_000000_b.json"
    record = json.loads(path.read_text())
    record["results"][0]["model"] = "c/d"  # a_only + b_only, no shared combo
    path.write_text(json.dumps(record))
    assert _run_cli(["compare", "--fail-on-regression", "0"]) == 1  # fail closed
    record["results"].append({**record["results"][0], "model": "a/b"})
    path.write_text(json.dumps(record))
    assert _run_cli(["compare", "--fail-on-regression", "0"]) == 0  # a/b shared


@pytest.mark.parametrize("points", ["nan", "inf", "-1", "two"])
def test_fail_on_regression_needs_finite_points(points, capsys):
    assert _run_cli(["compare", "--fail-on-regression", points]) == 2
    assert "use 0 or more" in capsys.readouterr().err


@pytest.mark.parametrize(
    "runs, message",
    [
        ((), "No run records"),
        ((("2026-01-01_000000_b", "B", 9, 0),), "No other run of 'B'"),
    ],
)
def test_compare_without_paths_explains_what_is_missing(
    monkeypatch, tmp_path, runs, message, capsys
):
    monkeypatch.chdir(tmp_path)
    _results_dir(tmp_path, *runs)
    code = _run_cli(["compare"])
    assert isinstance(code, str) and message in code


def test_a_first_run_passes_the_regression_check_with_a_note(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.chdir(tmp_path)
    _results_dir(tmp_path, ("2026-01-01_000000_b", "B", 9, 0))
    assert _run_cli(["compare", "--fail-on-regression", "0"]) == 0
    assert "Nothing to check." in capsys.readouterr().out


def test_report_without_a_path_uses_the_latest_run(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    assert isinstance(_run_cli(["report"]), str)  # no results/ yet
    _results_dir(
        tmp_path,
        ("2026-01-01_000000_b", "B", 9, 0),
        ("2026-01-02_000000_b", "B", 9, 0),
    )
    assert _run_cli(["report"]) == 0
    assert "Using results/2026-01-02_000000_b.json" in capsys.readouterr().err
    assert (tmp_path / "results" / "2026-01-02_000000_b.md").exists()
    assert not (tmp_path / "results" / "2026-01-01_000000_b.md").exists()


# ── tables never truncate ids ─────────────────────────────────────────────────


def test_tables_wrap_long_ids_instead_of_cutting_them(monkeypatch, capsys):
    long_id = "vendor/a-very-long-model-identifier-that-will-not-fit:free"
    catalog = {long_id: {"prompt": 0.0, "completion": 0.0, "context_length": 1}}
    monkeypatch.setattr(
        main.openrouter, "fetch_models", lambda client, use_cache=True: catalog
    )
    monkeypatch.setenv("COLUMNS", "60")
    assert _run_cli(["models"]) == 0
    out = capsys.readouterr().out
    assert "…" not in out
    first_column = [
        line.split("│")[1].strip() for line in out.splitlines() if line.startswith("│")
    ]
    assert "".join(first_column) == long_id


def test_the_gate_fails_when_its_combo_fails_although_another_passes(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    def post(self, **kwargs):  # m/a always right, m/b always wrong
        return _ok("yes" if kwargs["json"]["model"] == "m/a" else "no")

    monkeypatch.setattr(httpx.Client, "post", post)
    for gated, code in (("m/b", 1), ("m/a", 0)):
        path = _write_cli_config(
            tmp_path, models=["m/a", "m/b"], quality_threshold=50, gate={"model": gated}
        )
        assert _run_cli(["run", str(path), "--no-cache"]) == code
        out = capsys.readouterr().out
        verdict = "FAIL: m/b score is below" if code else "PASS: m/a score meets"
        assert verdict in out


def test_a_prompt_filter_drops_a_gate_on_another_prompt(fake_http, tmp_path, capsys):
    fake_http(_ok("yes"))
    path = _write_cli_config(
        tmp_path,
        prompts=[
            {"id": "v1", "template": "Q: {{ q }}"},
            {"id": "v2", "template": "{{ q }}?"},
        ],
        quality_threshold=50,
        gate={"prompt": "v2"},
    )
    assert _run_cli(["run", str(path), "-p", "v1"]) == 0
    captured = capsys.readouterr()
    assert "gate ignored" in captured.err
    assert "PASS: best score meets" in captured.out


def test_filters_never_hide_a_config_mistake(fake_http, tmp_path, capsys):
    fake_http(_ok("yes"))
    path = _write_cli_config(tmp_path, quality_threshold=50, gate={"model": "m/typo"})
    assert _run_cli(["run", str(path), "-m", "m/a", "--dry-run"]) == 1
    assert (
        "gate.model 'm/typo' is not one of the config's models"
        in capsys.readouterr().err
    )


def test_tag_filter_skips_cases_without_tags(fake_http, tmp_path):
    calls = fake_http(_ok("yes"))
    case = {"id": "c1", "evaluation": "exact_match", "expected_output": "yes"}
    path = _write_cli_config(
        tmp_path, test_cases=[{**case, "variables": {"q": "1"}, "tags": None}]
    )
    assert _run_cli(["run", str(path), "--tag", "x"]) == "No test case has tag 'x'."
    assert calls == []


def test_python_dash_m_runs_the_cli():
    import subprocess

    out = subprocess.run(
        [sys.executable, "-m", "evalblink", "--version"],
        capture_output=True,
        text=True,
        cwd=Path(main.__file__).resolve().parent.parent,
        check=True,
    ).stdout
    assert out.strip() == main.__version__


def test_dry_run_reports_an_unknown_model_once(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    catalog = {"m/b": {"prompt": 0.0, "completion": 0.0}}
    monkeypatch.setattr(
        main.openrouter, "fetch_models", lambda client, use_cache=True: catalog
    )
    path = _write_cli_config(tmp_path, models=["m/a"])
    assert _run_cli(["run", str(path), "--dry-run"]) == 1
    out = capsys.readouterr().out
    assert "Unknown model 'm/a'" in out and "No pricing for" not in out


def test_a_baseline_copied_from_results_is_not_compared_with_itself(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.chdir(tmp_path)
    _results_dir(tmp_path, ("2026-01-01_000000_b", "B", 9, 0))
    baseline = tmp_path / "baseline.json"
    baseline.write_text((tmp_path / "results" / "2026-01-01_000000_b.json").read_text())
    assert _run_cli(["compare", str(baseline), "--fail-on-regression", "0"]) == 0
    assert "No other run of 'B'" in capsys.readouterr().out


def test_tag_order_does_not_split_a_slice(fake_http, tmp_path):
    fake_http(_ok("yes"))
    path = _filter_config(tmp_path)
    assert _run_cli(["run", str(path), "--tag", "odd", "--tag", "even"]) == 0
    assert _run_cli(["run", str(path), "--tag", "even", "--tag", "odd"]) == 0
    filters = {
        json.dumps(json.loads(p.read_text())["filters"])
        for p in (tmp_path / "results").glob("*.json")
    }
    assert len(filters) == 1
