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
import math
import os
import threading
import time

import httpx

from . import cache

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
# Live model catalog (pricing + context) is cached on disk for a day so a
# dry-run estimate doesn't refetch it on every invocation (PRD §15).
MODELS_CACHE_PATH = os.path.join(cache.CACHE_DIR, "models.json")
# Bumped when the cached fields change; an older cache file is refetched.
MODELS_CACHE_FORMAT = 2
MODELS_TTL_SECONDS = 24 * 3600

# OpenRouter / upstream errors that are worth retrying: gateway timeouts and
# transient overload from the provider it routes to, plus rate limiting.
RETRYABLE_CODES = {408, 429, 502, 503, 504}
MAX_RETRIES = 4


def is_number(value) -> bool:
    """A real number that converts to a finite float.

    The one numeric gate for anything read from a config, an API response, a
    cache entry or a results file: rejects bools (an int subclass), ``inf`` /
    ``nan``, and ints too large for a float (``1`` followed by 400 zeros), which
    would otherwise overflow later in arithmetic or formatting.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def clean_text(text: str) -> str:
    """``text`` with lone UTF-16 surrogates (e.g. a ``"\\ud800"`` JSON escape)
    replaced, so it can be UTF-8 encoded for the request body, files and the
    terminal."""
    return text.encode("utf-8", "replace").decode("utf-8")


# The fields of a completion result, as returned and as cached.
_RESULT_KEYS = ("response", "prompt_tokens", "completion_tokens", "cost")


def _result_problem(result) -> str | None:
    """Why a completion result (fresh or cached) is unusable, or ``None``."""
    content = result.get("response")
    if not isinstance(content, str) or not content.strip():
        what = "null" if content is None else f"empty/non-text ({content!r})"
        return f"{what} content. Increase max_tokens."
    if not all(is_number(result.get(k)) for k in _RESULT_KEYS[1:]):
        return "malformed usage: " + str({k: result.get(k) for k in _RESULT_KEYS[1:]})
    return None


_cache_write_warned = False

# Set by the runner on Ctrl-C so worker threads stop retrying and calling.
CANCELLED = threading.Event()


def _cache_set(key: str, value: dict) -> None:
    """Persist to the cache; a write failure warns once and never aborts a run."""
    global _cache_write_warned
    try:
        cache.set(key, value)
    except OSError as exc:
        if not _cache_write_warned:
            _cache_write_warned = True
            print(f"⚠️  Could not write the response cache ({exc}); continuing.")


def _auth_headers() -> dict:
    """Standard OpenRouter request headers with the Bearer key from the env."""
    key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if not key.isascii():
        # httpx would raise UnicodeEncodeError building the header; every caller
        # handles RuntimeError. Typical cause: a pasted no-break space or quote.
        raise RuntimeError("OPENROUTER_API_KEY contains a non-ASCII character")
    headers = {
        "HTTP-Referer": "localhost",  # Optional. Site URL for rankings on openrouter.ai.
        "X-OpenRouter-Title": "evalblink",  # Optional. Site title for rankings on openrouter.ai.
    }
    # No key: the public /models catalog works without one, and "Bearer " is
    # an illegal header value.
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def openrouter_request(
    client: httpx.Client,
    prompt: str,
    model: str,
    temperature: float = 0,
    max_tokens: int = 4096,
    system: str | None = None,
    timeout: int = 120,
    use_cache: bool = True,
) -> dict:
    """Call OpenRouter (or the cache) and return ``{response, *_tokens, cost}``.

    The returned dict carries an extra ``from_cache`` flag so callers can skip
    rate-limit sleeps on a cache hit; that flag is never persisted in the cache.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": clean_text(system)})
    messages.append({"role": "user", "content": clean_text(prompt)})

    # The request body fully determines the response — it is the cache key.
    payload = {
        "model": clean_text(model),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    key = cache.sha256_key(payload)
    if use_cache:
        cached = cache.get(key)
        # A cache hit is checked exactly like a fresh response; a tampered or
        # stale-format entry is a miss (refetched and overwritten), not a crash.
        if cached is not None and _result_problem(cached) is None:
            cached = {k: cached[k] for k in _RESULT_KEYS}
            return {
                **cached,
                "response": clean_text(cached["response"]),
                "from_cache": True,
            }

    headers = _auth_headers()

    # OpenRouter free/slow models 504 intermittently; retry transient failures so a
    # single flaky call doesn't abort the whole benchmark run. Backoff: 1s, 2s, 4s.
    # A transient code may arrive on the HTTP status line or in the JSON error body.
    for attempt in range(MAX_RETRIES):
        if CANCELLED.is_set():  # Ctrl-C: no new attempt, no more retries
            raise RuntimeError("cancelled")
        last = attempt == MAX_RETRIES - 1
        try:
            response = client.post(
                url=OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=timeout,  # default httpx timeout is 5s — too short for free models
            )
        except httpx.HTTPError as exc:
            # Local timeout / connection failure — treat like an upstream 504.
            # Any other client-side error (bad gzip, redirect loop) fails at once.
            if last or not isinstance(exc, httpx.TransportError):
                detail = str(exc) or type(exc).__name__
                if isinstance(exc, httpx.TimeoutException):
                    what = "timed out"
                elif isinstance(exc, httpx.TransportError):
                    what = f"failed to connect ({detail})"
                else:
                    what = f"failed ({detail})"
                raise RuntimeError(
                    f"OpenRouter request {what} after {attempt + 1} attempt(s)"
                ) from exc
            time.sleep(2**attempt)
            continue

        status = response.status_code
        try:
            full = response.json()
        except ValueError:  # HTML error page, empty body, bad encoding...
            full = None
        if not isinstance(full, dict):  # also a bare JSON list / string / null
            full = None
        error = full.get("error") if full is not None else None
        code = error.get("code") if isinstance(error, dict) else None
        transient = status in RETRYABLE_CODES or (
            isinstance(code, int) and code in RETRYABLE_CODES
        )
        failed = status >= 400 or full is None or error is not None
        if failed and transient and not last:
            print(
                f"OpenRouter {code or status} (attempt {attempt + 1}/{MAX_RETRIES});"
                " retrying..."
            )
            time.sleep(2**attempt)
            continue
        break

    if error is not None:
        message = error.get("message") if isinstance(error, dict) else error
        raise RuntimeError(
            f"OpenRouter error {code if code is not None else status}"
            f" (HTTP {status}): {clean_text(str(message))}"
        )
    if full is None:
        raise RuntimeError(
            f"OpenRouter returned a non-JSON response (HTTP {status}):"
            f" {response.text[:200]!r}"
        )
    if status >= 400:
        raise RuntimeError(f"OpenRouter HTTP {status}: {str(full)[:200]}")
    try:
        usage = full["usage"]
        request_result = {
            "response": full["choices"][0]["message"]["content"],
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "cost": usage["cost"],
        }
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"OpenRouter returned a malformed response (HTTP {status},"
            f" {type(exc).__name__}: {exc}): {str(full)[:200]}"
        ) from exc
    problem = _result_problem(request_result)
    if problem:
        finish = None
        try:
            finish = full["choices"][0].get("finish_reason")
        except (KeyError, IndexError, TypeError, AttributeError):
            pass
        reason = f" (finish_reason: {finish})" if finish else ""
        raise RuntimeError(f"OpenRouter returned {problem}{reason}")
    request_result["response"] = clean_text(request_result["response"])
    if use_cache:
        _cache_set(key, request_result)
    return {**request_result, "from_cache": False}


def _price(value) -> float | None:
    """A per-token USD price, or ``None`` (unknown) when it is not a finite
    non-negative number — e.g. the ``-1`` OpenRouter quotes for its routers."""
    if value is None:  # unknown — e.g. a router's price as stored in the models cache
        return None
    try:
        price = float(value or 0)  # "" reads as free, as OpenRouter means
    except (TypeError, ValueError, OverflowError):
        return None
    return price if math.isfinite(price) and price >= 0 else None


def _normalize(meta: dict) -> dict:
    """``{prompt, completion, context_length, text, max_output}`` with unknown
    values as ``None``."""
    context = _positive_int(meta.get("context_length"))
    text = meta.get("text")
    return {
        # An absent key reads as free (OpenRouter omits zero prices); None is unknown.
        "prompt": _price(meta.get("prompt", "")),
        "completion": _price(meta.get("completion", "")),
        "context_length": context,
        "text": text if isinstance(text, bool) else None,
        # The provider's cap on completion tokens for this model.
        "max_output": _positive_int(meta.get("max_output")),
    }


def _positive_int(value) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number > 0 else None


def _is_text(architecture) -> bool | None:
    """Text in, text only out; ``None`` when the catalog doesn't say."""
    arch = architecture if isinstance(architecture, dict) else {}
    ins, outs = arch.get("input_modalities"), arch.get("output_modalities")
    if not (isinstance(ins, list) and isinstance(outs, list)):
        return None
    return "text" in ins and outs == ["text"]


def _parse_models(data: list) -> dict:
    """Reduce the raw ``/models`` payload to ``{id: {prompt, completion, context_length, text, max_output}}``.

    Prices are per-token USD floats (``0`` for free models) or ``None`` when the
    catalog's value is unusable; a malformed entry never sinks the whole catalog.
    """
    models = {}
    for entry in data:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            continue
        pricing = entry.get("pricing")
        pricing = pricing if isinstance(pricing, dict) else {}
        top = entry.get("top_provider")
        models[entry["id"]] = _normalize(
            {
                **pricing,
                "context_length": entry.get("context_length"),
                "text": _is_text(entry.get("architecture")),
                "max_output": top.get("max_completion_tokens")
                if isinstance(top, dict)
                else None,
            }
        )
    return models


def fetch_models(client: httpx.Client, use_cache: bool = True) -> dict:
    """Return ``{model_id: {prompt, completion, context_length, text, max_output}}`` from OpenRouter.

    The catalog is cached on disk for ``MODELS_TTL_SECONDS`` (24h); a fresh cache
    is served without a network call. Prices are per-token USD floats, or
    ``None`` when unknown (negative, non-numeric); callers treat ``None`` as
    "no pricing".
    """
    if use_cache and os.path.exists(MODELS_CACHE_PATH):
        try:
            with open(MODELS_CACHE_PATH) as f:
                cached = json.load(f)
            fresh = time.time() - cached.get("fetched_at", 0) < MODELS_TTL_SECONDS
            if fresh and cached.get("format") == MODELS_CACHE_FORMAT:
                return {k: _normalize(v) for k, v in cached["models"].items()}
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            OverflowError,
        ):
            pass  # corrupt cache file — refetch and overwrite it

    response = client.get(
        url=OPENROUTER_MODELS_URL, headers=_auth_headers(), timeout=30
    )
    try:
        full = response.json()
        if "error" in full:
            raise RuntimeError(f"OpenRouter error: {full['error']}")
        models = _parse_models(full.get("data", []))
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise RuntimeError(
            "OpenRouter returned an unusable model catalog"
            f" (HTTP {response.status_code}): {type(exc).__name__}: {exc}"
        ) from exc

    try:
        os.makedirs(cache.CACHE_DIR, exist_ok=True)
        with open(MODELS_CACHE_PATH, "w") as f:
            json.dump(
                {
                    "format": MODELS_CACHE_FORMAT,
                    "fetched_at": time.time(),
                    "models": models,
                },
                f,
                indent=4,
            )
    except OSError as exc:
        print(f"⚠️  Could not cache the model catalog ({exc}); continuing.")
    return models
