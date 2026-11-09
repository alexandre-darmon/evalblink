"""Where a model runs: OpenRouter, or a local / self-hosted OpenAI-compatible server.

A model id names its provider before the first colon — ``ollama:llama3.2``,
``lmstudio:qwen/qwen3-8b``, ``vllm:Qwen/Qwen2.5-7B-Instruct`` — and is sent to
that server's ``/chat/completions`` with the prefix removed. Every other id goes
to OpenRouter: its ids always have a ``vendor/`` before any colon
(``openai/gpt-4o:free``), so a prefix without ``/`` is never one of them.

Four servers are built in at their default local address; the config's
``providers:`` block moves one (``ollama: {base_url: http://gpu-box:11434/v1}``)
or adds any other OpenAI-compatible endpoint. Pure lookup, no I/O: the requests
themselves stay in ``openrouter``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

OPENROUTER = "openrouter"

# Defaults for the servers people run on their own machine. `start` / `pull`
# are the hints printed when the server is down or lacks the model.
BUILTIN = {
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "start": "ollama serve",
        "pull": "ollama pull {model}",
    },
    "lmstudio": {
        "base_url": "http://localhost:1234/v1",
        "start": "lms server start, or Developer → Start Server in LM Studio",
        "pull": "lms get {model}",
    },
    "llamacpp": {
        "base_url": "http://localhost:8080/v1",
        "start": "llama-server -m <model.gguf>",
        # llama-server answers with whichever model it loaded, whatever the name.
        "check_models": False,
    },
    "vllm": {
        "base_url": "http://localhost:8000/v1",
        "start": "vllm serve <model>",
    },
}

# What a `providers:` entry may set.
KEYS = {"base_url", "api_key_env", "concurrency", "timeout"}
# One request at a time: a local server queues the rest, and the queue wait
# would be measured as latency.
DEFAULT_CONCURRENCY = 1
# Seconds per request: a local model on a laptop can take minutes.
DEFAULT_TIMEOUT = 600
OPENROUTER_TIMEOUT = 120

NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class ProviderError(ValueError):
    """A model id whose provider prefix names no known provider."""


@dataclass(frozen=True)
class Endpoint:
    provider: str  # "openrouter", or the prefix of a local id
    model: str  # the name the server knows: the id without its prefix
    base_url: str  # no trailing slash
    api_key_env: str | None = None
    concurrency: int | None = None  # None: only the run's `concurrency`
    timeout: float = OPENROUTER_TIMEOUT
    check_models: bool = True
    start: str | None = None
    pull: str | None = None

    @property
    def local(self) -> bool:
        return self.provider != OPENROUTER


def split(model: str) -> tuple[str | None, str]:
    """``("ollama", "llama3.2:3b")`` for ``ollama:llama3.2:3b``; ``(None, model)``
    for an OpenRouter id."""
    prefix, sep, rest = str(model).partition(":")
    if sep and rest and NAME.match(prefix):
        return prefix, rest
    return None, model


def _entries(providers) -> dict:
    """The config's ``providers:`` block as ``{name: settings}``; a bare
    ``ollama:`` (null) keeps the built-in settings."""
    if not isinstance(providers, dict):
        return {}
    return {
        name: value if isinstance(value, dict) else {}
        for name, value in providers.items()
    }


def known(providers=None) -> list[str]:
    """Every provider name a model id may use."""
    return sorted(set(BUILTIN) | {str(n) for n in _entries(providers)})


def resolve(model: str, providers=None) -> Endpoint:
    """The endpoint that serves ``model``; raises :class:`ProviderError` for an
    unknown prefix (``olama:llama3.2``)."""
    name, rest = split(model)
    if name is None:
        return Endpoint(OPENROUTER, model, "https://openrouter.ai/api/v1")
    settings = {**BUILTIN.get(name, {}), **_entries(providers).get(name, {})}
    base_url = settings.get("base_url")
    if not isinstance(base_url, str) or not base_url:
        raise ProviderError(f"unknown provider '{name}' in model '{model}'")
    return Endpoint(
        provider=name,
        model=rest,
        base_url=base_url.rstrip("/"),
        api_key_env=settings.get("api_key_env"),
        concurrency=settings.get("concurrency") or DEFAULT_CONCURRENCY,
        timeout=settings.get("timeout") or DEFAULT_TIMEOUT,
        check_models=settings.get("check_models", True),
        start=settings.get("start"),
        pull=settings.get("pull"),
    )


def called_models(config: dict) -> list[str]:
    """The model ids a run of ``config`` calls: the candidates, plus the judge
    when an ``llm_judge`` case needs it."""
    models = [m for m in config.get("models") or [] if isinstance(m, str)]
    judge = (config.get("evaluation") or {}).get("judge_model")
    judged = any(
        isinstance(tc, dict) and tc.get("evaluation") == "llm_judge"
        for tc in config.get("test_cases") or []
    )
    if judged and isinstance(judge, str) and judge not in models:
        models.append(judge)
    return models


def is_local(model: str) -> bool:
    return split(model)[0] is not None
