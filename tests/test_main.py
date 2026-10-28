"""CLI dispatch: argparse routes subcommands to the right handler."""

from __future__ import annotations

import json
import os
from collections import deque
from pathlib import Path

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


def test_models_invalid_min_context_exits_cleanly(monkeypatch):
    monkeypatch.setattr(
        main.openrouter, "fetch_models", lambda client, use_cache=True: _FAKE_MODELS
    )
    monkeypatch.setattr(
        main.sys, "argv", ["evalblink", "models", "--min-context", "notanumber"]
    )
    with pytest.raises(SystemExit) as exc:
        main.main()
    # sys.exit(msg) stores the message as the exit code when it's a string.
    assert "Invalid" in str(exc.value.code)


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
            if isinstance(item, Exception):
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
