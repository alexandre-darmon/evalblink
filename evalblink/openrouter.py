"""The single OpenRouter API choke point.

One job: turn a prompt into a completion (with usage accounting), reusing a
caller-supplied ``httpx.Client`` and consulting the SHA256 cache so identical
requests never hit the network twice. Both the candidate loop (``runner``) and
the LLM judge (``evaluator``) go through here, which is why it lives in its own
leaf module — keeping it out of ``runner`` avoids a runner<->evaluator import
cycle.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

import httpx

from . import cache

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
# Live model catalog (pricing + context) is cached on disk for a day so a
# dry-run estimate doesn't refetch it on every invocation (PRD §15).
MODELS_CACHE_PATH = os.path.join(cache.CACHE_DIR, "models.json")
MODELS_TTL_SECONDS = 24 * 3600

# OpenRouter / upstream errors that are worth retrying: gateway timeouts and
# transient overload from the provider it routes to, plus rate limiting.
RETRYABLE_CODES = {408, 429, 502, 503, 504}
MAX_RETRIES = 4


def _auth_headers() -> dict:
    """Standard OpenRouter request headers with the Bearer key from the env."""
    return {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "HTTP-Referer": "localhost",  # Optional. Site URL for rankings on openrouter.ai.
        "X-OpenRouter-Title": "evalblink",  # Optional. Site title for rankings on openrouter.ai.
    }


def openrouter_request(
    client: httpx.Client,
    prompt: str,
    model: str,
    temperature: float = 0,
    max_tokens: int = 4096,
    system: Optional[str] = None,
    timeout: int = 120,
    use_cache: bool = True,
) -> dict:
    """Call OpenRouter (or the cache) and return ``{response, *_tokens, cost}``.

    The returned dict carries an extra ``from_cache`` flag so callers can skip
    rate-limit sleeps on a cache hit; that flag is never persisted in the cache.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    # The request body fully determines the response — it is the cache key.
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    key = cache.sha256_key(payload)
    if use_cache:
        cached = cache.get(key)
        if cached is not None:
            return {**cached, "from_cache": True}

    headers = _auth_headers()

    # OpenRouter free/slow models 504 intermittently; retry transient failures so a
    # single flaky call doesn't abort the whole benchmark run. Backoff: 1s, 2s, 4s.
    full: dict = {}
    for attempt in range(MAX_RETRIES):
        try:
            response = client.post(
                url=OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=timeout,  # default httpx timeout is 5s — too short for free models
            )
            full = response.json()
        except httpx.TimeoutException as exc:
            # Local read/connect timeout — treat like an upstream 504.
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(
                    f"OpenRouter request timed out after {MAX_RETRIES} attempts"
                ) from exc
            time.sleep(2**attempt)
            continue

        error = full.get("error")
        if error and error.get("code") in RETRYABLE_CODES and attempt < MAX_RETRIES - 1:
            print(
                f"OpenRouter {error['code']} (attempt {attempt + 1}/{MAX_RETRIES}); retrying..."
            )
            time.sleep(2**attempt)
            continue
        break

    if "error" in full:
        raise RuntimeError(
            f"OpenRouter error {full['error']['code']}: {full['error']['message']}"
        )
    content = full["choices"][0]["message"]["content"]
    if content is None:
        raise RuntimeError(
            f"OpenRouter returned null content. Increase max_tokens. Full response: {full}"
        )
    request_result = {
        "response": content,
        "prompt_tokens": full["usage"]["prompt_tokens"],
        "completion_tokens": full["usage"]["completion_tokens"],
        "cost": full["usage"]["cost"],
    }
    if use_cache:
        cache.set(key, request_result)
    return {**request_result, "from_cache": False}


def _parse_models(data: list) -> dict:
    """Reduce the raw ``/models`` payload to ``{id: {prompt, completion, context_length}}``."""
    models = {}
    for entry in data:
        model_id = entry.get("id")
        if not model_id:
            continue
        pricing = entry.get("pricing") or {}
        models[model_id] = {
            # Prices are per-token USD strings ("0" for free models).
            "prompt": float(pricing.get("prompt", 0) or 0),
            "completion": float(pricing.get("completion", 0) or 0),
            "context_length": entry.get("context_length"),
        }
    return models


def fetch_models(client: httpx.Client, use_cache: bool = True) -> dict:
    """Return ``{model_id: {prompt, completion, context_length}}`` from OpenRouter.

    The catalog is cached on disk for ``MODELS_TTL_SECONDS`` (24h); a fresh cache
    is served without a network call. Prices are per-token USD floats.
    """
    if use_cache and os.path.exists(MODELS_CACHE_PATH):
        with open(MODELS_CACHE_PATH) as f:
            cached = json.load(f)
        if time.time() - cached.get("fetched_at", 0) < MODELS_TTL_SECONDS:
            return cached["models"]

    response = client.get(
        url=OPENROUTER_MODELS_URL, headers=_auth_headers(), timeout=30
    )
    full = response.json()
    if "error" in full:
        raise RuntimeError(
            f"OpenRouter error {full['error']['code']}: {full['error']['message']}"
        )
    models = _parse_models(full.get("data", []))

    os.makedirs(cache.CACHE_DIR, exist_ok=True)
    with open(MODELS_CACHE_PATH, "w") as f:
        json.dump({"fetched_at": time.time(), "models": models}, f, indent=4)
    return models
