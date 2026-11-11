"""Local and self-hosted models: `ollama:llama3.2` and the `providers:` block."""

from __future__ import annotations

import threading
import time

import httpx
import pytest

from evalblink import cache, estimate, main, openrouter, providers, runner
from evalblink.openrouter import openrouter_request
from evalblink.providers import ProviderError
from evalblink.validator import validate

# ── which server serves a model id ────────────────────────────────────────────


@pytest.mark.parametrize(
    "model, provider, name",
    [
        ("ollama:llama3.2", "ollama", "llama3.2"),
        ("ollama:llama3.2:3b", "ollama", "llama3.2:3b"),  # tags keep their colon
        ("lmstudio:qwen/qwen3-8b", "lmstudio", "qwen/qwen3-8b"),
        ("openai/gpt-4o-mini", None, "openai/gpt-4o-mini"),
        # OpenRouter variants: a vendor/ comes before the colon.
        ("meta-llama/llama-3.3-70b-instruct:free", None, None),
        ("ollama:", None, None),  # nothing after the prefix
    ],
)
def test_split(model, provider, name):
    assert providers.split(model) == (provider, name or model)


def test_resolve_builtin_override_and_custom():
    ollama = providers.resolve("ollama:llama3.2")
    assert (ollama.base_url, ollama.model, ollama.local) == (
        "http://localhost:11434/v1",
        "llama3.2",
        True,
    )
    assert (ollama.concurrency, ollama.timeout) == (1, 600)

    config = {
        "ollama": {"base_url": "http://gpu-box:11434/v1/", "concurrency": 2},
        "lab": {"base_url": "http://10.0.0.5:8000/v1", "api_key_env": "LAB_KEY"},
    }
    moved = providers.resolve("ollama:llama3.2", config)
    assert (moved.base_url, moved.concurrency) == ("http://gpu-box:11434/v1", 2)
    assert moved.start == "ollama serve"  # the built-in hints survive an override
    lab = providers.resolve("lab:qwen2.5", config)
    assert (lab.base_url, lab.api_key_env, lab.model) == (
        "http://10.0.0.5:8000/v1",
        "LAB_KEY",
        "qwen2.5",
    )
    remote = providers.resolve("openai/gpt-4o-mini", config)
    assert not remote.local and remote.concurrency is None
    with pytest.raises(ProviderError, match="unknown provider 'olama'"):
        providers.resolve("olama:llama3.2")


def test_called_models_adds_the_judge_only_when_used():
    config = {
        "models": ["ollama:a"],
        "evaluation": {"judge_model": "ollama:judge"},
        "test_cases": [{"id": "x", "evaluation": "exact_match"}],
    }
    assert providers.called_models(config) == ["ollama:a"]
    config["test_cases"].append({"id": "y", "evaluation": "llm_judge"})
    assert providers.called_models(config) == ["ollama:a", "ollama:judge"]


# ── the request ───────────────────────────────────────────────────────────────


def test_a_local_request_goes_to_the_server_without_the_openrouter_key(
    client_factory, disable_cache, monkeypatch
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-secret")
    # Ollama's OpenAI endpoint: usage without a cost.
    body = {
        "choices": [{"message": {"content": "billing"}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 1},
    }
    client = client_factory(body)
    result = openrouter_request(client, "Classify", "ollama:llama3.2")
    call = client.calls[0]
    assert call["url"] == "http://localhost:11434/v1/chat/completions"
    assert call["json"]["model"] == "llama3.2"
    assert "Authorization" not in call["headers"]
    assert call["timeout"] == 600
    assert (result["response"], result["prompt_tokens"], result["cost"]) == (
        "billing",
        12,
        0,
    )
    assert result["latency_s"] is not None


def test_a_local_reply_without_usage_counts_zero_tokens(client_factory, disable_cache):
    client = client_factory({"choices": [{"message": {"content": "ok"}}]})
    result = openrouter_request(client, "Hi", "llamacpp:any")
    assert (result["prompt_tokens"], result["completion_tokens"], result["cost"]) == (
        0,
        0,
        0,
    )


def test_openrouter_still_requires_usage(client_factory, disable_cache):
    client = client_factory({"choices": [{"message": {"content": "ok"}}]})
    with pytest.raises(RuntimeError, match="OpenRouter returned a malformed response"):
        openrouter_request(client, "Hi", "openai/gpt-4o-mini")


def test_a_custom_provider_sends_its_own_key(client_factory, completion, monkeypatch):
    monkeypatch.setattr(cache, "get", lambda key: None)
    monkeypatch.setattr(cache, "set", lambda key, value: None)
    monkeypatch.setenv("LAB_KEY", "lab-secret")
    client = client_factory(completion("ok"))
    config = {"lab": {"base_url": "https://lab.example/v1", "api_key_env": "LAB_KEY"}}
    openrouter_request(client, "Hi", "lab:qwen", providers_config=config)
    assert client.calls[0]["url"] == "https://lab.example/v1/chat/completions"
    assert client.calls[0]["headers"] == {"Authorization": "Bearer lab-secret"}


def test_two_servers_with_the_same_model_name_never_share_cache_entries(
    client_factory, completion, monkeypatch
):
    store = {}
    monkeypatch.setattr(cache, "get", store.get)
    monkeypatch.setattr(cache, "set", store.__setitem__)
    client = client_factory([completion("from ollama"), completion("from vllm")])
    a = openrouter_request(client, "Hi", "ollama:qwen2.5")
    b = openrouter_request(client, "Hi", "vllm:qwen2.5")
    assert (a["response"], b["response"]) == ("from ollama", "from vllm")
    assert len(store) == 2
    again = openrouter_request(client, "Hi", "ollama:qwen2.5")
    assert again["from_cache"] and again["response"] == "from ollama"


def test_openrouter_cache_keys_are_unchanged(client_factory, completion, monkeypatch):
    seen = []
    monkeypatch.setattr(cache, "get", lambda key: seen.append(key))
    monkeypatch.setattr(cache, "set", lambda key, value: None)
    openrouter_request(client_factory(completion("x")), "Hi", "openai/gpt-4o-mini")
    payload = {
        "model": "openai/gpt-4o-mini",
        "temperature": 0,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": "Hi"}],
    }
    assert seen == [cache.sha256_key(payload)]


def test_a_server_that_is_down_says_how_to_start_it(client_factory, disable_cache):
    refused = httpx.ConnectError("Connection refused")
    client = client_factory([refused])
    with pytest.raises(RuntimeError) as exc:
        openrouter_request(client, "Hi", "ollama:llama3.2")
    message = str(exc.value)
    assert message.startswith("ollama (http://localhost:11434/v1) request failed")
    assert "Start it: ollama serve" in message
    assert len(client.calls) == openrouter.MAX_RETRIES


def test_a_local_error_is_named_after_the_server(client_factory, disable_cache):
    client = client_factory(
        [{"error": {"message": "model 'llama9' not found", "type": "api_error"}}]
    )
    with pytest.raises(
        RuntimeError, match=r"^ollama \(http://localhost:11434/v1\) error"
    ):
        openrouter_request(client, "Hi", "ollama:llama9")


def test_a_local_429_gets_no_openrouter_advice(client_factory, disable_cache):
    client = client_factory([httpx.Response(429, json={"error": "busy"})])
    with pytest.raises(RuntimeError) as exc:
        openrouter_request(client, "Hi", "vllm:m")
    assert "free models" not in str(exc.value)


def test_a_local_server_gets_one_request_at_a_time(monkeypatch, completion):
    monkeypatch.setattr(cache, "get", lambda key: None)
    monkeypatch.setattr(cache, "set", lambda key, value: None)
    monkeypatch.setattr(openrouter, "_slots", {})
    state = {"now": 0, "peak": 0}
    lock = threading.Lock()

    class SlowClient:
        def post(self, **kwargs):
            with lock:
                state["now"] += 1
                state["peak"] = max(state["peak"], state["now"])
            time.sleep(0.02)
            with lock:
                state["now"] -= 1
            return httpx.Response(200, json=completion("ok"))

    threads = [
        threading.Thread(
            target=openrouter_request, args=(SlowClient(), f"q{i}", "ollama:m")
        )
        for i in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert state["peak"] == 1


def test_fetch_local_models(client_factory):
    endpoint = providers.resolve("ollama:llama3.2")
    listed = {"data": [{"id": "llama3.2:latest"}, {"id": "qwen2.5:7b"}, {"x": 1}]}
    client = client_factory(listed)
    assert openrouter.fetch_local_models(client, endpoint) == [
        "llama3.2:latest",
        "qwen2.5:7b",
    ]
    assert client.calls[0]["url"] == "http://localhost:11434/v1/models"
    assert openrouter.fetch_local_models(client_factory("<html>"), endpoint) is None
    not_found = httpx.Response(404, json={"error": "no"})
    assert openrouter.fetch_local_models(client_factory([not_found]), endpoint) is None


# ── validate ──────────────────────────────────────────────────────────────────


def _config(**extra):
    return {
        "name": "Local",
        "models": ["ollama:llama3.2"],
        "prompts": [{"id": "v1", "template": "Classify"}],
        "test_cases": [
            {"id": "a", "evaluation": "exact_match", "expected_output": "x"}
        ],
        **extra,
    }


def test_validate_accepts_local_models_and_providers():
    config = _config(
        models=["ollama:llama3.2", "lab:qwen", "openai/gpt-4o-mini"],
        providers={
            "ollama": None,
            "lab": {
                "base_url": "http://localhost:8000/v1",
                "api_key_env": "LAB_KEY",
                "concurrency": 2,
                "timeout": 120,
            },
        },
    )
    assert validate(config) == ([], [])


@pytest.mark.parametrize(
    "extra, message",
    [
        (
            {"models": ["olama:llama3.2"]},
            "unknown provider 'olama' — did you mean 'ollama:'?",
        ),
        ({"models": ["mylab:qwen"]}, "add others under `providers:`"),
        ({"providers": ["ollama"]}, "providers must be a mapping"),
        (
            {"providers": {"Lab Box": {"base_url": "http://x"}}},
            "a provider name is lowercase",
        ),
        (
            {"providers": {"openrouter": {"base_url": "http://x"}}},
            "'openrouter' is reserved",
        ),
        ({"providers": {"lab": {}}}, "providers.lab: base_url is required"),
        (
            {"providers": {"lab": {"base_url": "localhost:8000"}}},
            "must be an http(s) URL",
        ),
        ({"providers": {"ollama": {"concurrency": 0}}}, "providers.ollama.concurrency"),
        ({"providers": {"ollama": {"timeout": -1}}}, "providers.ollama.timeout"),
        ({"providers": {"ollama": {"api_key_env": 3}}}, "api_key_env must be the name"),
    ],
)
def test_validate_rejects(extra, message):
    errors, _ = validate(_config(**extra))
    assert any(message in e for e in errors), errors


def test_validate_checks_the_judge_prefix_and_warns_on_unknown_keys():
    config = _config(
        evaluation={"judge_model": "olama:qwen2.5"},
        providers={"ollama": {"base_ur": "http://x"}},
    )
    config["test_cases"] = [{"id": "j", "evaluation": "llm_judge", "criteria": "ok"}]
    errors, warnings = validate(config)
    assert any(e.startswith("evaluation.judge_model: 'olama:qwen2.5'") for e in errors)
    assert any("providers.ollama: unknown key 'base_ur'" in w for w in warnings)


# ── run: key, servers, dry run ────────────────────────────────────────────────


def _run(monkeypatch, config, listed=None, down=False):
    """``evalblink run`` on ``config``; the servers list ``listed``."""
    monkeypatch.setattr(main, "load_config", lambda p: config)
    monkeypatch.setattr(main.sys, "argv", ["evalblink", "run", "bench.yaml"])
    asked = []

    def fake_list(client, endpoint):
        asked.append(endpoint.base_url)
        if down:
            raise httpx.ConnectError("refused")
        return listed

    monkeypatch.setattr(main.openrouter, "fetch_local_models", fake_list)
    ran = []
    monkeypatch.setattr(main.runner, "run", lambda *a, **k: ran.append(1) or ([], "t"))
    monkeypatch.setattr(main.reporter, "write", lambda *a, **k: {"passed": True})
    with pytest.raises(SystemExit) as exc:
        main.main()
    return exc.value.code, asked, bool(ran)


def test_a_local_only_run_needs_no_openrouter_key(monkeypatch, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    code, asked, ran = _run(monkeypatch, _config(), listed=["llama3.2:latest"])
    assert (code, asked, ran) == (0, ["http://localhost:11434/v1"], True)
    assert (
        "✓  ollama at http://localhost:11434/v1: llama3.2 ready"
        in capsys.readouterr().err
    )


def test_a_mixed_run_still_needs_the_key(monkeypatch, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    config = _config(models=["ollama:llama3.2", "openai/gpt-4o-mini"])
    code, _, ran = _run(monkeypatch, config, listed=["llama3.2"])
    assert (code, ran) == (1, False)
    assert "OPENROUTER_API_KEY is not set" in capsys.readouterr().err


def test_a_server_that_is_down_stops_the_run(monkeypatch):
    message, _, ran = _run(monkeypatch, _config(), down=True)
    assert not ran
    assert "Can't reach ollama at http://localhost:11434/v1" in message
    assert "Start it: ollama serve" in message


def test_a_missing_model_stops_the_run_with_the_pull_command(monkeypatch):
    message, _, ran = _run(monkeypatch, _config(), listed=["qwen2.5:7b"])
    assert not ran
    assert "ollama at http://localhost:11434/v1 has no model 'llama3.2'" in message
    assert "Get it: ollama pull llama3.2." in message
    assert "Available: qwen2.5:7b" in message


def test_llamacpp_serves_whatever_it_loaded(monkeypatch):
    code, _, ran = _run(
        monkeypatch, _config(models=["llamacpp:any"]), listed=["/m.gguf"]
    )
    assert (code, ran) == (0, True)


def test_a_server_without_a_model_list_is_trusted(monkeypatch):
    code, _, ran = _run(monkeypatch, _config(), listed=None)
    assert (code, ran) == (0, True)


def test_a_missing_provider_key_variable_stops_the_run(monkeypatch):
    monkeypatch.delenv("LAB_KEY", raising=False)
    config = _config(
        models=["lab:qwen"],
        providers={"lab": {"base_url": "http://x/v1", "api_key_env": "LAB_KEY"}},
    )
    message, asked, ran = _run(monkeypatch, config, listed=["qwen"])
    assert (asked, ran) == ([], False)
    assert "LAB_KEY is not set" in message


def test_a_local_only_dry_run_is_offline_and_free(monkeypatch, capsys):
    monkeypatch.setattr(main, "load_config", lambda p: _config())
    monkeypatch.setattr(
        main.sys, "argv", ["evalblink", "run", "bench.yaml", "--dry-run"]
    )
    monkeypatch.setattr(
        main, "_fetch_models", lambda **k: pytest.fail("catalog fetched")
    )
    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Local models cost $0 here (your own hardware): ollama:llama3.2." in out


def test_estimate_prices_local_models_at_zero():
    config = _config(models=["ollama:llama3.2", "openai/gpt-4o-mini"])
    catalog = {
        "openai/gpt-4o-mini": {"prompt": 1e-6, "completion": 2e-6, "max_output": None}
    }
    est = estimate.estimate(config, catalog)
    local, remote = est["rows"]
    assert local["est_cost"] == 0 and remote["est_cost"] > 0
    assert est["missing_pricing"] == [] and est["unknown_models"] == []
    assert est["local_models"] == ["ollama:llama3.2"]


def test_the_run_line_names_each_server_limit():
    config = _config(
        models=["ollama:a", "vllm:b", "openai/gpt-4o-mini"],
        providers={"vllm": {"concurrency": 4}},
    )
    assert runner._local_limits(config) == (" (ollama: 1 at a time, vllm: 4 at a time)")
    assert runner._local_limits(_config(models=["openai/gpt-4o-mini"])) == ""
