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
                "architecture": {
                    "input_modalities": ["text", "image"],
                    "output_modalities": ["text"],
                },
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
        "text": True,
        "max_output": None,
    }
    # Missing completion price defaults to 0 for free models.
    assert models["free/x"]["completion"] == 0.0
    assert len(client.calls) == 1
    # A fresh cache file was written.
    assert os.path.exists(_models_cache)


def test_fetch_models_uses_fresh_cache(client_factory, _models_cache):
    with open(_models_cache, "w") as f:
        json.dump(
            {
                "format": openrouter.MODELS_CACHE_FORMAT,
                "fetched_at": time.time(),
                "models": {"cached/m": {"prompt": 9.0}},
            },
            f,
        )
    client = client_factory(_catalog())
    models = fetch_models(client)
    # Served from cache (normalized: absent price = free, unknown context).
    assert models == {
        "cached/m": {
            "prompt": 9.0,
            "completion": 0.0,
            "context_length": None,
            "text": None,
            "max_output": None,
        }
    }
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
        json.dump(
            {
                "format": openrouter.MODELS_CACHE_FORMAT,
                "fetched_at": time.time(),
                "models": {"cached/m": {}},
            },
            f,
        )
    client = client_factory(_catalog())
    models = fetch_models(client, use_cache=False)
    assert "a/b" in models
    assert len(client.calls) == 1


# --- any failure of a call is a RuntimeError (round 2 contract) -----------

HTML_502 = "<html><body><h1>502 Bad Gateway</h1></body></html>"


def test_retryable_http_status_with_html_body_is_retried(client_factory, completion):
    client = client_factory([httpx.Response(502, text=HTML_502), completion("ok")])
    assert openrouter_request(client, "p", "m", use_cache=False)["response"] == "ok"
    assert len(client.calls) == 2


def test_persistent_html_502_raises_with_status(client_factory):
    client = client_factory(httpx.Response(502, text=HTML_502))
    with pytest.raises(RuntimeError, match="HTTP 502"):
        openrouter_request(client, "p", "m", use_cache=False)
    assert len(client.calls) == MAX_RETRIES


@pytest.mark.parametrize("status", [400, 401, 404, 500])
@pytest.mark.parametrize("as_json", [True, False])
def test_non_retryable_http_status_raises_once(client_factory, status, as_json):
    response = (
        httpx.Response(status, json={"error": {"code": status, "message": "nope"}})
        if as_json
        else httpx.Response(status, text=HTML_502)
    )
    client = client_factory(response)
    with pytest.raises(RuntimeError, match=f"HTTP {status}"):
        openrouter_request(client, "p", "m", use_cache=False)
    assert len(client.calls) == 1


def test_retryable_status_line_with_json_error_body_is_retried(
    client_factory, completion
):
    body = {"error": {"code": 429, "message": "slow down"}}
    client = client_factory([httpx.Response(429, json=body), completion("ok")])
    assert openrouter_request(client, "p", "m", use_cache=False)["response"] == "ok"


def test_retryable_code_in_json_body_only_is_retried(client_factory, completion):
    # HTTP 200 on the status line, a transient code in the error body.
    body = {"error": {"code": 503, "message": "overloaded"}}
    client = client_factory([httpx.Response(200, json=body), completion("ok")])
    assert openrouter_request(client, "p", "m", use_cache=False)["response"] == "ok"


@pytest.mark.parametrize(
    "exc", [httpx.ConnectError("refused"), httpx.RemoteProtocolError("closed")]
)
def test_connection_error_is_retried_then_raises(client_factory, completion, exc):
    client = client_factory([exc, completion("ok")])
    assert openrouter_request(client, "p", "m", use_cache=False)["response"] == "ok"
    client = client_factory(exc)
    with pytest.raises(RuntimeError, match="failed to connect"):
        openrouter_request(client, "p", "m", use_cache=False)
    assert len(client.calls) == MAX_RETRIES


_USAGE = {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0}


@pytest.mark.parametrize(
    "response, message",
    [
        (httpx.Response(200, text=HTML_502), "non-JSON"),
        (httpx.Response(200, text=""), "non-JSON"),
        (httpx.Response(200, json=[1, 2]), "non-JSON"),
        (httpx.Response(200, json={"error": "broke"}), "broke"),
        (httpx.Response(200, json={"usage": _USAGE}), "malformed"),
        (httpx.Response(200, json={"choices": [], "usage": _USAGE}), "malformed"),
        (httpx.Response(200, json={"choices": [{}], "usage": _USAGE}), "malformed"),
        (
            httpx.Response(200, json={"choices": [{"message": None}]}),
            "malformed",
        ),
        (
            httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]}),
            "malformed",
        ),
        (
            httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "x"}}],
                    "usage": {"prompt_tokens": "1", "completion_tokens": 1, "cost": 0},
                },
            ),
            "malformed usage",
        ),
        (
            httpx.Response(
                200, json={"choices": [{"message": {"content": ""}}], "usage": _USAGE}
            ),
            "empty",
        ),
        (
            httpx.Response(
                200, json={"choices": [{"message": {"content": [1]}}], "usage": _USAGE}
            ),
            "non-text",
        ),
    ],
)
def test_unusable_response_raises_runtime_error(client_factory, response, message):
    client = client_factory(response)
    with pytest.raises(RuntimeError, match=message):
        openrouter_request(client, "p", "m", use_cache=False)


def test_corrupt_cache_entry_reads_as_miss(
    monkeypatch, tmp_path, client_factory, completion
):
    monkeypatch.setattr(openrouter.cache, "CACHE_DIR", str(tmp_path))
    payload = {
        "model": "m",
        "temperature": 0,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": "p"}],
    }
    key = openrouter.cache.sha256_key(payload)
    (tmp_path / f"{key}.json").write_text('{"response": "trunc')
    client = client_factory(completion("fresh"))
    assert openrouter_request(client, "p", "m")["response"] == "fresh"
    # ...and the good result overwrote the corrupt entry.
    assert openrouter_request(client, "p", "m")["from_cache"] is True


def test_fetch_models_unusable_catalog_raises_runtime_error(
    client_factory, _models_cache
):
    for response in (httpx.Response(502, text=HTML_502), httpx.Response(200, json=[1])):
        with pytest.raises(RuntimeError, match="unusable model catalog"):
            fetch_models(client_factory(response))


def test_fetch_models_corrupt_cache_refetches(client_factory, _models_cache):
    with open(_models_cache, "w") as f:
        f.write("{not json")
    client = client_factory(_catalog())
    assert "a/b" in fetch_models(client)
    assert len(client.calls) == 1


def test_error_status_is_an_error_even_with_a_completion_shaped_body(
    client_factory, completion
):
    client = client_factory(httpx.Response(500, json=completion("looks fine")))
    with pytest.raises(RuntimeError, match="HTTP 500"):
        openrouter_request(client, "p", "m", use_cache=False)


# --- round 3: surrogates, cache validation, client errors, catalog hygiene ---


def _key(prompt="p", model="m"):
    return openrouter.cache.sha256_key(
        {
            "model": model,
            "temperature": 0,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }
    )


def test_lone_surrogates_never_reach_the_wire_or_the_output(
    monkeypatch, tmp_path, client_factory
):
    monkeypatch.setattr(openrouter.cache, "CACHE_DIR", str(tmp_path))
    # Content with a "\ud800" JSON escape, as a model may emit.
    usage = json.dumps(_USAGE).encode()
    body = b'{"choices":[{"message":{"content":"a\\ud800b"}}],"usage":' + usage + b"}"
    client = client_factory(httpx.Response(200, content=body))
    result = openrouter_request(client, "q\ud800", "m\udfff")
    result["response"].encode("utf-8")  # no lone surrogate left
    # The request body httpx builds (ensure_ascii=False, UTF-8) is encodable.
    json.dumps(client.calls[0]["json"], ensure_ascii=False).encode("utf-8")
    assert openrouter_request(client, "q\ud800", "m\udfff")["from_cache"] is True


def test_error_message_surrogate_is_cleaned(client_factory):
    client = client_factory(
        httpx.Response(400, content=b'{"error":{"code":400,"message":"x\\ud800"}}')
    )
    with pytest.raises(RuntimeError) as exc:
        openrouter_request(client, "p", "m", use_cache=False)
    str(exc.value).encode("utf-8")


@pytest.mark.parametrize(
    "entry",
    [
        {"response": None, "prompt_tokens": 1, "completion_tokens": 1, "cost": 0},
        {"response": "  ", "prompt_tokens": 1, "completion_tokens": 1, "cost": 0},
        {"response": "x"},  # usage missing
        {"response": "x", "prompt_tokens": "1", "completion_tokens": 1, "cost": 0},
        {"response": "x", "prompt_tokens": 1, "completion_tokens": 1, "cost": 10**400},
        {"response": "x", "prompt_tokens": 1, "completion_tokens": True, "cost": 0},
    ],
)
def test_invalid_cache_entry_is_a_miss_and_refetched(
    monkeypatch, tmp_path, client_factory, completion, entry
):
    monkeypatch.setattr(openrouter.cache, "CACHE_DIR", str(tmp_path))
    (tmp_path / f"{_key()}.json").write_text(json.dumps(entry))
    client = client_factory(completion("fresh"))
    result = openrouter_request(client, "p", "m")
    assert result == {
        "response": "fresh",
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "cost": 0.0,
        "from_cache": False,
    }
    assert len(client.calls) == 1


def test_unreadable_cache_entry_is_a_miss(
    monkeypatch, tmp_path, client_factory, completion
):
    monkeypatch.setattr(openrouter.cache, "CACHE_DIR", str(tmp_path))
    (tmp_path / f"{_key()}.json").mkdir()  # a directory where the entry should be
    client = client_factory(completion("fresh"))
    assert openrouter_request(client, "p", "m")["response"] == "fresh"


def test_cache_write_failure_warns_and_continues(
    monkeypatch, tmp_path, client_factory, completion, capsys
):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")
    monkeypatch.setattr(openrouter.cache, "CACHE_DIR", str(blocker))
    monkeypatch.setattr(openrouter, "_cache_write_warned", False)
    client = client_factory(completion("ok"))
    assert openrouter_request(client, "p", "m")["response"] == "ok"
    assert openrouter_request(client, "p2", "m")["response"] == "ok"
    out = capsys.readouterr().out
    assert out.count("Could not write the response cache") == 1  # warned once


@pytest.mark.parametrize(
    "exc",
    [
        httpx.DecodingError("bad gzip"),
        httpx.TooManyRedirects("loop"),
        httpx.HTTPStatusError(
            "x", request=httpx.Request("POST", "http://x"), response=httpx.Response(500)
        ),
    ],
)
def test_non_transport_httpx_error_is_a_runtime_error_without_retry(
    client_factory, exc
):
    client = client_factory(exc)
    with pytest.raises(RuntimeError, match="failed"):
        openrouter_request(client, "p", "m", use_cache=False)
    assert len(client.calls) == 1


@pytest.mark.parametrize("body", [{}, {"choices": []}, {"error": None}])
def test_retryable_status_line_without_error_object_is_retried(
    client_factory, completion, body
):
    # 503 on the status line, a JSON body with no 'error' object: still a
    # transient failure, retried — not parsed as a completion, not fatal.
    client = client_factory([httpx.Response(503, json=body), completion("ok")])
    assert openrouter_request(client, "p", "m", use_cache=False)["response"] == "ok"
    assert len(client.calls) == 2


def test_catalog_unknown_prices_and_bad_context_are_normalized(
    client_factory, _models_cache
):
    catalog = {
        "data": [
            {"id": "openrouter/auto", "pricing": {"prompt": "-1", "completion": "-1"}},
            {"id": "s/x", "context_length": "lots", "pricing": {"prompt": "abc"}},
            {"id": "h/x", "context_length": 1e400, "pricing": {"prompt": 10**400}},
            {"id": "ok/x", "context_length": "8192", "pricing": {"prompt": "0"}},
            "not-an-entry",
            {"id": 5},
        ]
    }
    models = fetch_models(client_factory(catalog))
    assert models["openrouter/auto"]["prompt"] is None
    assert models["openrouter/auto"]["completion"] is None
    assert models["s/x"] == {
        "prompt": None,
        "completion": 0.0,
        "context_length": None,
        "text": None,
        "max_output": None,
    }
    assert models["h/x"]["prompt"] is None and models["h/x"]["context_length"] is None
    assert models["ok/x"]["context_length"] == 8192
    assert set(models) == {"openrouter/auto", "s/x", "h/x", "ok/x"}


def test_stale_format_models_cache_is_renormalized(client_factory, _models_cache):
    with open(_models_cache, "w") as f:
        json.dump(
            {
                "format": openrouter.MODELS_CACHE_FORMAT,
                "fetched_at": time.time(),
                "models": {
                    "r/x": {"prompt": -1, "completion": -1, "context_length": "?"}
                },
            },
            f,
        )
    assert fetch_models(client_factory(_catalog()))["r/x"] == {
        "prompt": None,
        "completion": None,
        "context_length": None,
        "text": None,
        "max_output": None,
    }


def test_models_cache_write_failure_still_returns_catalog(
    monkeypatch, tmp_path, client_factory, capsys
):
    blocker = tmp_path / "blocker"
    blocker.write_text("")
    monkeypatch.setattr(openrouter.cache, "CACHE_DIR", str(blocker))
    monkeypatch.setattr(openrouter, "MODELS_CACHE_PATH", str(blocker / "models.json"))
    assert "a/b" in fetch_models(client_factory(_catalog()))
    assert "Could not cache the model catalog" in capsys.readouterr().out


def test_cancelled_run_stops_retrying(client_factory, error_body, monkeypatch):
    # Ctrl-C sets CANCELLED; a worker mid-retry must give up, not keep calling.
    client = client_factory(error_body(429))
    monkeypatch.setattr(openrouter.time, "sleep", lambda s: openrouter.CANCELLED.set())
    with pytest.raises(RuntimeError, match="cancelled"):
        openrouter_request(client, "p", "m", use_cache=False)
    assert len(client.calls) == 1


# --- release review fixes ----------------------------------------------------


def test_no_key_sends_no_authorization_header(monkeypatch):
    # "Bearer " (empty key) is an illegal header value in httpx/h11; the public
    # /models catalog (dry-run, models, init) must work without a key.
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    headers = openrouter._auth_headers()
    assert "Authorization" not in headers
    request = httpx.Client().build_request("GET", "https://x.test/", headers=headers)
    assert "authorization" not in request.headers
    monkeypatch.setenv("OPENROUTER_API_KEY", " sk-test ")
    assert openrouter._auth_headers()["Authorization"] == "Bearer sk-test"


def test_unknown_prices_stay_unknown_when_read_back_from_cache(
    client_factory, _models_cache
):
    body = {
        "data": [
            {"id": "openrouter/auto", "pricing": {"prompt": "-1", "completion": "-1"}},
            {"id": "free/x", "pricing": {"prompt": "0"}},  # absent key = free
            {"id": "free/y", "pricing": {}},
        ]
    }
    fresh = fetch_models(client_factory(body))
    cached = fetch_models(client_factory(AssertionError("must hit the cache")))
    for models in (fresh, cached):
        assert models["openrouter/auto"]["prompt"] is None
        assert models["openrouter/auto"]["completion"] is None
        assert models["free/x"]["completion"] == 0.0
        assert models["free/y"] == {
            **models["free/y"],
            "prompt": 0.0,
            "completion": 0.0,
        }


def test_null_content_error_is_short_and_names_finish_reason(client_factory):
    body = {
        "choices": [
            {
                "message": {"content": None, "reasoning": "x" * 2000},
                "finish_reason": "length",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0},
    }
    with pytest.raises(RuntimeError) as exc:
        openrouter_request(client_factory(body), "p", "m", use_cache=False)
    message = str(exc.value)
    assert "null content" in message and "finish_reason: length" in message
    assert "xxxx" not in message and len(message) < 200


@pytest.mark.parametrize(
    "architecture, text",
    [
        ({"input_modalities": ["text"], "output_modalities": ["text"]}, True),
        ({"input_modalities": ["image", "text"], "output_modalities": ["text"]}, True),
        ({"input_modalities": ["text"], "output_modalities": ["text", "audio"]}, False),
        ({"input_modalities": ["image"], "output_modalities": ["text"]}, False),
        ({"input_modalities": ["text"], "output_modalities": ["image"]}, False),
        ({"input_modalities": "text", "output_modalities": ["text"]}, None),
        ({}, None),
        (None, None),
        ("text->text", None),
    ],
)
def test_text_modality_survives_the_cache(
    client_factory, _models_cache, architecture, text
):
    entry = {"id": "m/x", "pricing": {"prompt": "0"}, "architecture": architecture}
    assert fetch_models(client_factory({"data": [entry]}))["m/x"]["text"] is text
    assert fetch_models(client_factory({"data": []}))["m/x"]["text"] is text  # cached


@pytest.mark.parametrize(
    "top_provider, cap",
    [
        ({"max_completion_tokens": 2048}, 2048),
        ({"max_completion_tokens": "4000"}, 4000),
        ({"max_completion_tokens": None}, None),
        ({"max_completion_tokens": 0}, None),
        ({"max_completion_tokens": -5}, None),
        ({"max_completion_tokens": True}, None),
        ({"max_completion_tokens": 1e400}, None),
        ({}, None),
        (None, None),
        ([2048], None),
    ],
)
def test_output_cap_survives_the_cache(
    client_factory, _models_cache, top_provider, cap
):
    entry = {"id": "m/x", "pricing": {"prompt": "0"}, "top_provider": top_provider}
    assert fetch_models(client_factory({"data": [entry]}))["m/x"]["max_output"] == cap
    assert fetch_models(client_factory({"data": []}))["m/x"]["max_output"] == cap


def test_a_cache_from_an_older_release_is_refetched(client_factory, _models_cache):
    # 0.1.0 wrote no "format" and no text / max_output fields.
    with open(_models_cache, "w") as f:
        json.dump({"fetched_at": time.time(), "models": {"old/m": {}}}, f)
    client = client_factory(_catalog())
    assert "a/b" in fetch_models(client) and len(client.calls) == 1
    assert "a/b" in fetch_models(client) and len(client.calls) == 1  # now cached


def _sleeps(monkeypatch):
    waits = []
    monkeypatch.setattr(openrouter.time, "sleep", waits.append)
    return waits


@pytest.mark.parametrize(
    "first",
    [
        {"error": {"code": 429, "message": "slow down"}},
        httpx.Response(429, text="<html>Too Many Requests</html>"),
    ],
)
def test_rate_limits_back_off_for_longer_with_jitter(
    client_factory, completion, monkeypatch, first
):
    waits = _sleeps(monkeypatch)
    monkeypatch.setattr(openrouter.random, "uniform", lambda a, b: (a + b) / 2)
    client = client_factory([first, first, first, completion("ok")])
    assert openrouter_request(client, "p", "m/x", use_cache=False)["response"] == "ok"
    assert waits == [5.5, 10.5, 20.5]


def test_other_transient_errors_keep_the_short_backoff(
    client_factory, completion, error_body, monkeypatch
):
    waits = _sleeps(monkeypatch)
    client = client_factory([error_body(504), error_body(502), completion("ok")])
    openrouter_request(client, "p", "m/x", use_cache=False)
    assert waits == [1, 2]


@pytest.mark.parametrize("header, wait", [("7", 7.0), ("0", 0.0), ("600", 60.0)])
def test_retry_after_is_honoured_and_capped(
    client_factory, completion, monkeypatch, header, wait
):
    waits = _sleeps(monkeypatch)
    limited = httpx.Response(
        429, headers={"Retry-After": header}, json={"error": {"code": 429}}
    )
    openrouter_request(
        client_factory([limited, completion("ok")]), "p", "m/x", use_cache=False
    )
    assert waits == [wait]


@pytest.mark.parametrize("header", ["soon", "-3", "nan", "inf", ""])
def test_unusable_retry_after_falls_back_to_the_backoff(
    client_factory, completion, monkeypatch, header
):
    waits = _sleeps(monkeypatch)
    limited = httpx.Response(
        429, headers={"Retry-After": header}, json={"error": {"code": 429}}
    )
    openrouter_request(
        client_factory([limited, completion("ok")]), "p", "m/x", use_cache=False
    )
    assert 5 <= waits[0] < 6


@pytest.mark.parametrize(
    "body",
    [
        {"error": {"code": 429, "message": "Provider returned error"}},
        httpx.Response(429, text="<html>Too Many Requests</html>"),
        httpx.Response(429, json={"detail": "busy"}),
    ],
)
def test_exhausted_rate_limit_says_what_to_do(client_factory, monkeypatch, body):
    _sleeps(monkeypatch)
    with pytest.raises(RuntimeError, match="Rate-limited: free models") as exc:
        openrouter_request(client_factory(body), "p", "m/x", use_cache=False)
    assert "429" in str(exc.value)


def test_other_failures_carry_no_rate_limit_hint(
    client_factory, error_body, monkeypatch
):
    _sleeps(monkeypatch)
    with pytest.raises(RuntimeError) as exc:
        openrouter_request(client_factory(error_body(504)), "p", "m/x", use_cache=False)
    assert "Rate-limited" not in str(exc.value)
