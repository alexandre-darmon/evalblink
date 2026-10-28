"""Shared pytest fixtures: a fake OpenRouter client and cache/sleep controls.

The production code is built to be tested without the network — ``openrouter_request``
and ``evaluate_llm_judge`` take an injectable ``httpx.Client``. These fixtures supply
a stand-in that returns queued response bodies, plus knobs to disable the real file
cache and the retry/rate-limit sleeps.
"""

from __future__ import annotations

import httpx
import pytest

from evalblink import cache, main, openrouter


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Never actually sleep during retry/backoff while testing."""
    monkeypatch.setattr(openrouter.time, "sleep", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """No test may reach the network or pick up a real ``.env`` API key.

    Any real ``httpx.Client`` request fails the test loudly; ``main()``'s
    ``.env`` loading is a no-op (tests that exercise it restore the real one).
    """

    def _no_network(self, request, *a, **k):
        raise AssertionError(f"test attempted a real HTTP request: {request.url}")

    monkeypatch.setattr(httpx.Client, "send", _no_network)
    openrouter.CANCELLED.clear()  # a Ctrl-C test must not leak into the next one
    monkeypatch.setattr(main, "load_dotenv", lambda *a, **k: False)


@pytest.fixture
def disable_cache(monkeypatch):
    """Bypass the file cache so judge tests never touch ``.evalblink_cache/``."""
    monkeypatch.setattr(cache, "get", lambda key: None)
    monkeypatch.setattr(cache, "set", lambda key, value: None)


@pytest.fixture
def completion():
    """Factory for a successful OpenRouter completion body."""

    def _make(content, prompt_tokens=10, completion_tokens=5, cost=0.0):
        return {
            "choices": [{"message": {"content": content}}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost": cost,
            },
        }

    return _make


@pytest.fixture
def error_body():
    """Factory for an OpenRouter error body, e.g. ``error_body(504)``."""

    def _make(code, message="boom"):
        return {"error": {"code": code, "message": message}}

    return _make


@pytest.fixture
def client_factory():
    """Return the FakeClient class (instantiate with queued bodies)."""

    class _FakeResponse:
        status_code = 200

        def __init__(self, body):
            self._body = body
            self.text = str(body)

        def json(self):
            return self._body

    class FakeClient:
        """Minimal ``httpx.Client`` stand-in for ``openrouter_request``.

        ``bodies`` is consumed one per ``post`` call; an element that is an
        ``Exception`` instance is raised instead of returned, an ``httpx.Response``
        (real status line + raw body) is returned as is, and the last element
        repeats if there are more calls than bodies. ``calls`` records every POST.
        """

        def __init__(self, bodies):
            self.bodies = bodies if isinstance(bodies, list) else [bodies]
            self.calls = []

        def post(self, **kwargs):
            self.calls.append(kwargs)
            body = self.bodies[min(len(self.calls) - 1, len(self.bodies) - 1)]
            if isinstance(body, Exception):
                raise body
            if isinstance(body, httpx.Response):
                return body
            return _FakeResponse(body)

        # ``fetch_models`` issues a GET; reuse the same queued-body mechanism.
        get = post

    return FakeClient
