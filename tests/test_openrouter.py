"""Tests for the OpenRouter choke point: success, caching bypass, and retry policy."""

from __future__ import annotations

import json
import os
import time

import httpx
import pytest

from evalblink import openrouter
from evalblink.openrouter import MAX_RETRIES, fetch_models, openrouter_request


def test_success_returns_normalized_result(client_factory, completion):
    client = client_factory(
        completion("hello", prompt_tokens=7, completion_tokens=3, cost=0.01)
    )
    result = openrouter_request(client, "prompt", "model", use_cache=False)
    assert result["response"] == "hello"
    assert result["prompt_tokens"] == 7
    assert result["completion_tokens"] == 3
    assert result["cost"] == 0.01
    assert result["from_cache"] is False
    assert len(client.calls) == 1


def test_504_retries_then_raises(client_factory, error_body):
    client = client_factory(error_body(504))
    with pytest.raises(RuntimeError, match="504"):
        openrouter_request(client, "p", "m", use_cache=False)
    assert len(client.calls) == MAX_RETRIES


def test_400_raises_immediately_without_retry(client_factory, error_body):
    client = client_factory(error_body(400, "bad request"))
    with pytest.raises(RuntimeError, match="400"):
        openrouter_request(client, "p", "m", use_cache=False)
    assert len(client.calls) == 1


def test_transient_error_then_success(client_factory, error_body, completion):
    client = client_factory([error_body(429), completion("recovered")])
    result = openrouter_request(client, "p", "m", use_cache=False)
    assert result["response"] == "recovered"
    assert len(client.calls) == 2


def test_local_timeout_is_retried(client_factory, completion):
    client = client_factory([httpx.ReadTimeout("slow"), completion("ok")])
    result = openrouter_request(client, "p", "m", use_cache=False)
    assert result["response"] == "ok"
    assert len(client.calls) == 2


def test_persistent_timeout_raises(client_factory):
    client = client_factory(httpx.ReadTimeout("slow"))
    with pytest.raises(RuntimeError, match="timed out"):
        openrouter_request(client, "p", "m", use_cache=False)
    assert len(client.calls) == MAX_RETRIES


def test_null_content_raises(client_factory, completion):
    client = client_factory(completion(None))
    with pytest.raises(RuntimeError, match="null content"):
        openrouter_request(client, "p", "m", use_cache=False)


# --- fetch_models (pricing catalog) ---------------------------------------


def _catalog():
    return {
        "data": [
            {
                "id": "a/b",
                "context_length": 8000,
                "pricing": {"prompt": "0.001", "completion": "0.002"},
            },
            {"id": "free/x", "context_length": 4096, "pricing": {"prompt": "0"}},
        ]
    }


@pytest.fixture
def _models_cache(monkeypatch, tmp_path):
    """Point the model-catalog cache at a temp dir so tests never touch the real one."""
    monkeypatch.setattr(openrouter.cache, "CACHE_DIR", str(tmp_path))
    path = os.path.join(str(tmp_path), "models.json")
    monkeypatch.setattr(openrouter, "MODELS_CACHE_PATH", path)
    return path


def test_fetch_models_parses_and_caches(client_factory, _models_cache):
    client = client_factory(_catalog())
    models = fetch_models(client)
    assert models["a/b"] == {
        "prompt": 0.001,
        "completion": 0.002,
        "context_length": 8000,
    }
    # Missing completion price defaults to 0 for free models.
    assert models["free/x"]["completion"] == 0.0
    assert len(client.calls) == 1
    # A fresh cache file was written.
    assert os.path.exists(_models_cache)


def test_fetch_models_uses_fresh_cache(client_factory, _models_cache):
    with open(_models_cache, "w") as f:
        json.dump(
            {"fetched_at": time.time(), "models": {"cached/m": {"prompt": 9.0}}}, f
        )
    client = client_factory(_catalog())
    models = fetch_models(client)
    assert models == {"cached/m": {"prompt": 9.0}}  # served from cache
    assert len(client.calls) == 0  # no network


def test_fetch_models_refetches_stale_cache(client_factory, _models_cache):
    stale = time.time() - openrouter.MODELS_TTL_SECONDS - 1
    with open(_models_cache, "w") as f:
        json.dump({"fetched_at": stale, "models": {"old/m": {}}}, f)
    client = client_factory(_catalog())
    models = fetch_models(client)
    assert "a/b" in models  # refetched, not the stale entry
    assert len(client.calls) == 1


def test_fetch_models_bypass_cache(client_factory, _models_cache):
    with open(_models_cache, "w") as f:
        json.dump({"fetched_at": time.time(), "models": {"cached/m": {}}}, f)
    client = client_factory(_catalog())
    models = fetch_models(client, use_cache=False)
    assert "a/b" in models
    assert len(client.calls) == 1
