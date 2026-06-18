"""Tests for the OpenRouter choke point: success, caching bypass, and retry policy."""

from __future__ import annotations

import httpx
import pytest

from evalblink.openrouter import MAX_RETRIES, openrouter_request


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
