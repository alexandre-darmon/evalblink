"""The single OpenRouter API choke point.

One job: turn a prompt into a completion (with usage accounting), reusing a
caller-supplied ``httpx.Client`` and consulting the SHA256 cache so identical
requests never hit the network twice. Both the candidate loop (``runner``) and
the LLM judge (``evaluator``) go through here, which is why it lives in its own
leaf module — keeping it out of ``runner`` avoids a runner<->evaluator import
cycle.
"""

from __future__ import annotations

import os
from typing import Optional

import httpx

from . import cache

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


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

    api_key = os.getenv("OPENROUTER_API_KEY")
    response = client.post(
        url=OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "localhost",  # Optional. Site URL for rankings on openrouter.ai.
            "X-OpenRouter-Title": "evalblink",  # Optional. Site title for rankings on openrouter.ai.
        },
        json=payload,
        timeout=timeout,  # default httpx timeout is 5s — too short for free models
    )
    full = response.json()
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
