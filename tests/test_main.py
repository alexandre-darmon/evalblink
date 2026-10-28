"""CLI dispatch: argparse routes subcommands to the right handler."""

from __future__ import annotations

import datetime
import json
import os
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import dotenv
import httpx
import pytest
import yaml
from rich.text import Text

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
    assert "OPENROUTER_API_KEY is not set" in capsys.readouterr().out


def test_no_subcommand_errors(monkeypatch):
    monkeypatch.setattr(main.sys, "argv", ["evalblink"])
    with pytest.raises(SystemExit) as exc:
        main.main()
    # argparse exits 2 on a required-subcommand parse error.
    assert exc.value.code == 2


# --- run --dry-run --------------------------------------------------------


def test_dry_run_estimates_without_running(monkeypatch):
    seen = {}
    monkeypatch.setattr(main, "load_config", lambda p: {"name": "X"})
    monkeypatch.setattr(main, "_validate_config", lambda c: ([], []))
    monkeypatch.setattr(main.openrouter, "fetch_models", lambda client: {"m": {}})
    monkeypatch.setattr(
        main.estimate, "estimate", lambda config, meta: {"over_budget": False}
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
    assert seen["rendered"] == {"over_budget": False}


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


# ── init ──────────────────────────────────────────────────────────────────────


def _make_fake_prompt(answers: deque):
    class FakePrompt:
        @classmethod
        def ask(cls, question, **kwargs):
            return answers.popleft()

    return FakePrompt


def _make_fake_confirm(answers: deque):
    class FakeConfirm:
        @classmethod
        def ask(cls, question, **kwargs):
            return answers.popleft()

    return FakeConfirm


def test_init_exact_match_creates_file(monkeypatch, tmp_path):
    monkeypatch.setattr(
        main.openrouter, "fetch_models", lambda client, use_cache=True: _FAKE_MODELS
    )
    monkeypatch.setattr(main.sys, "argv", ["evalblink", "init"])

    prompt_answers = deque(
        [
            "My Test",  # name
            "exact_match",  # mode
            "anthropic/claude-3-haiku",  # model 1
            "",  # blank → stop
            "Classify: {{ text }}",  # template
            "",  # system (skip)
            "p",  # 'text' is per-case
            "0",  # temperature
            "100",  # max_tokens
            "tc_001",  # test case id
            "some text here",  # {{ text }}
            "positive",  # expected_output
            "",  # tags (skip)
            "",  # quality threshold (skip)
            str(tmp_path / "my-test.yaml"),  # output path
        ]
    )
    confirm_answers = deque([False])  # "Add another test case?" → No

    monkeypatch.setattr(main, "Prompt", _make_fake_prompt(prompt_answers))
    monkeypatch.setattr(main, "Confirm", _make_fake_confirm(confirm_answers))

    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 0

    out_file = tmp_path / "my-test.yaml"
    assert out_file.exists()
    config = yaml.safe_load(out_file.read_text())
    assert config["name"] == "My Test"
    assert config["models"] == ["anthropic/claude-3-haiku"]
    assert config["prompts"][0]["template"] == "Classify: {{ text }}"
    assert config["test_cases"][0]["expected_output"] == "positive"
    assert config["test_cases"][0]["evaluation"] == "exact_match"


def test_init_llm_judge_creates_file(monkeypatch, tmp_path):
    monkeypatch.setattr(
        main.openrouter, "fetch_models", lambda client, use_cache=True: _FAKE_MODELS
    )
    monkeypatch.setattr(main.sys, "argv", ["evalblink", "init"])

    prompt_answers = deque(
        [
            "LLM Judge Test",  # name
            "llm_judge",  # mode
            "openai/gpt-4o",  # model
            "",  # stop
            "Summarize: {{ text }}",  # template
            "",  # system (skip)
            "p",  # 'text' is per-case
            "openai/gpt-4o",  # judge_model
            "0.70",  # judge_threshold
            "0",  # temperature
            "1024",  # max_tokens
            "tc_001",  # test case id
            "A mountain scene",  # {{ text }}
            "Must be concise and accurate",  # criteria
            "",  # tags (skip)
            "",  # quality threshold (skip)
            str(tmp_path / "llm-judge-test.yaml"),  # output path
        ]
    )
    confirm_answers = deque([False])

    monkeypatch.setattr(main, "Prompt", _make_fake_prompt(prompt_answers))
    monkeypatch.setattr(main, "Confirm", _make_fake_confirm(confirm_answers))

    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 0

    config = yaml.safe_load((tmp_path / "llm-judge-test.yaml").read_text())
    assert config["evaluation"]["judge_model"] == "openai/gpt-4o"
    assert config["test_cases"][0]["criteria"] == "Must be concise and accurate"
    assert config["test_cases"][0]["evaluation"] == "llm_judge"


def test_init_global_variable(monkeypatch, tmp_path):
    monkeypatch.setattr(
        main.openrouter, "fetch_models", lambda client, use_cache=True: _FAKE_MODELS
    )
    monkeypatch.setattr(main.sys, "argv", ["evalblink", "init"])

    out_file = tmp_path / "global-var-test.yaml"
    prompt_answers = deque(
        [
            "Global Var Test",  # name
            "exact_match",  # mode
            "anthropic/claude-3-haiku",  # model
            "",  # stop
            "Classify: {{ text }}. Labels: {{ labels }}",  # template (2 vars)
            "",  # system (skip)
            "p",  # 'text' is per-case
            "g",  # 'labels' is global
            "yes, no, maybe",  # value for labels
            "0",  # temperature
            "100",  # max_tokens
            "tc_001",  # test case id
            "some text",  # {{ text }}
            "yes",  # expected_output
            "",  # tags (skip)
            "",  # quality threshold (skip)
            str(out_file),  # output path
        ]
    )
    confirm_answers = deque([False])

    monkeypatch.setattr(main, "Prompt", _make_fake_prompt(prompt_answers))
    monkeypatch.setattr(main, "Confirm", _make_fake_confirm(confirm_answers))

    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 0

    config = yaml.safe_load(out_file.read_text())
    assert config["variables"] == {"labels": "yes, no, maybe"}
    assert config["test_cases"][0]["variables"] == {"text": "some text"}


def test_init_no_template_variables(monkeypatch, tmp_path):
    monkeypatch.setattr(
        main.openrouter, "fetch_models", lambda client, use_cache=True: _FAKE_MODELS
    )
    monkeypatch.setattr(main.sys, "argv", ["evalblink", "init"])

    out_file = tmp_path / "static.yaml"
    prompt_answers = deque(
        [
            "Static Test",  # name
            "exact_match",  # mode
            "anthropic/claude-3-haiku",  # model
            "",  # stop
            "What is 2+2?",  # template — no {{ }} vars
            "",  # system (skip)
            # no variable-scoping prompts (all_vars is empty)
            "0",  # temperature
            "100",  # max_tokens
            "tc_001",  # test case id
            # no per-case variable prompts
            "4",  # expected_output
            "",  # tags (skip)
            "",  # quality threshold (skip)
            str(out_file),  # output path
        ]
    )
    confirm_answers = deque([False])

    monkeypatch.setattr(main, "Prompt", _make_fake_prompt(prompt_answers))
    monkeypatch.setattr(main, "Confirm", _make_fake_confirm(confirm_answers))

    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 0

    config = yaml.safe_load(out_file.read_text())
    assert config["prompts"][0]["template"] == "What is 2+2?"
    assert "variables" not in config
    assert "variables" not in config["test_cases"][0]


def test_jinja_vars_helper():
    assert main._jinja_vars("Hello {{ name }}, score {{ score }}") == ["name", "score"]
    assert main._jinja_vars("{{ x }} and {{ x }} again") == ["x"]
    assert main._jinja_vars("no variables here") == []


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
    assert message in capsys.readouterr().out


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
    out = capsys.readouterr().out
    assert "2 of 2 case(s) hit a pipeline error" in out
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
    out = capsys.readouterr().out
    assert "1 of 2 case(s) hit a pipeline error" in out
    assert "no case could be scored" not in out

    # The scored case passes (100%), so a gate of 90 passes too.
    fake_http(_ok("yes"), httpx.Response(400, text="<html>bad</html>"))
    path = _write_cli_config(tmp_path, quality_threshold=90)
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
    out = capsys.readouterr().out
    assert "2 of 4 case(s) hit a pipeline error" in out
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


def _scripted(answers):
    """A ``Prompt``/``Confirm`` stand-in that, like Rich, returns the default
    for an empty answer."""

    class Scripted:
        @classmethod
        def ask(cls, question, **kwargs):
            Text.from_markup(question)  # Rich renders the question as markup
            answer = answers.popleft()
            if answer == "" and "default" in kwargs:
                return kwargs["default"]
            return answer

    return Scripted


def test_init_rejects_bad_answers_and_writes_a_valid_config(monkeypatch, tmp_path):
    from evalblink.validator import validate

    monkeypatch.chdir(tmp_path)

    def no_catalog(client, use_cache=True):
        raise RuntimeError("catalog down")

    monkeypatch.setattr(main.openrouter, "fetch_models", no_catalog)
    prompts = deque(
        [
            "",  # name: empty → re-asked
            "!!!",  # name (punctuation only → slug fallback)
            "exact_match",
            "a/b",
            "a/b",  # duplicate model → re-asked
            "",  # finish models
            "{% bad",  # template syntax error → re-asked
            "Hi {{ text|upper }} from {{ user.name }}",
            "",  # system
            "p",  # text: per-case
            "g",  # user: global
            "bob",
            "nan",  # temperature: rejected
            "1e400",  # temperature: inf, rejected
            "0.2",
            "100.5",  # max_tokens: rejected
            "0",  # max_tokens: rejected
            "1" + "0" * 400,  # max_tokens: overflows a float, rejected
            "50",
            "",  # tc id → default tc_001
            "hello",  # {{ text }}
            "yes",
            "",  # tags
            "tc_001",  # duplicate id → re-asked
            "tc_002",
            "world",
            "no",
            "",  # tags
            "150",  # quality threshold: out of range → re-asked
            "80",
            "",  # output path → default
        ]
    )
    confirms = deque([True, False])
    monkeypatch.setattr(main, "Prompt", _scripted(prompts))
    monkeypatch.setattr(main, "Confirm", _scripted(confirms))
    assert _run_cli(["init"]) == 0
    assert not prompts and not confirms

    out = tmp_path / "benchmarks" / "benchmark.yaml"
    config = yaml.safe_load(out.read_text())
    assert validate(config) == ([], [])
    assert config["name"] == "!!!"
    assert config["models"] == ["a/b"]
    assert config["variables"] == {"user": "bob"}
    assert [tc["id"] for tc in config["test_cases"]] == ["tc_001", "tc_002"]
    assert config["test_cases"][0]["variables"] == {"text": "hello"}
    assert config["inference"] == {"temperature": 0.2, "max_tokens": 50}
    assert config["quality_threshold"] == 80


def test_init_never_writes_a_config_validate_rejects(monkeypatch, tmp_path):
    monkeypatch.setattr(
        main.openrouter, "fetch_models", lambda client, use_cache=True: _FAKE_MODELS
    )
    out = tmp_path / "x.yaml"
    prompts = deque(
        ["X", "exact_match", "a/b", "", "{{ text.a.b }}", "", "p"]
        + ["0", "100", "", "plain", "yes", "", "", str(out)]
    )
    monkeypatch.setattr(main, "Prompt", _scripted(prompts))
    monkeypatch.setattr(main, "Confirm", _scripted(deque([False])))
    assert _run_cli(["init"]) == 1  # renders with UndefinedError
    assert not out.exists()


def test_jinja_vars_uses_validator_detection():
    assert main._jinja_vars("{{ text|upper }} {{ user.name }} {{ text }}") == [
        "text",
        "user",
    ]


def test_init_unwritable_output_path_is_a_clean_error(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        main.openrouter, "fetch_models", lambda client, use_cache=True: _FAKE_MODELS
    )
    for out, reason in (
        (str(tmp_path), "Cannot write"),
        ("\ud800.yaml", "Cannot write"),
    ):
        prompts = deque(
            ["X", "exact_match", "a/b", "", "Hi", "", "0", "100", "", "y", "", "", out]
        )
        monkeypatch.setattr(main, "Prompt", _scripted(prompts))
        monkeypatch.setattr(main, "Confirm", _scripted(deque([False, True])))
        code = _run_cli(["init"])
        assert isinstance(code, str) and reason in code


def test_init_overwrite_prompt_escapes_markup_in_the_path(monkeypatch, tmp_path):
    monkeypatch.setattr(
        main.openrouter, "fetch_models", lambda client, use_cache=True: _FAKE_MODELS
    )
    (tmp_path / "[").mkdir()
    out = tmp_path / "[" / "x].yaml"  # the path contains "[/x]"
    out.write_text("old")
    prompts = deque(
        ["X", "exact_match", "a/b", "", "Hi", "", "0", "100", "", "y", "", "", str(out)]
    )
    monkeypatch.setattr(main, "Prompt", _scripted(prompts))
    monkeypatch.setattr(main, "Confirm", _scripted(deque([False, True])))
    assert _run_cli(["init"]) == 0
    assert yaml.safe_load(out.read_text())["name"] == "X"


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
    assert len(calls) == 2  # the 18 queued calls were dropped, not sent
    assert list((tmp_path / "results").glob("*.json")) == []


def test_eof_during_init_is_a_clean_exit(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    def eof(*a, **k):
        raise EOFError

    monkeypatch.setattr(main.Prompt, "ask", eof)
    code = _run_cli(["init"])
    assert isinstance(code, str) and "No input" in code


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
