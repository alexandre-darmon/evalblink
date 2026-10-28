"""SHA256 file cache for OpenRouter responses.

A request is fully determined by its payload (model, messages, temperature,
max_tokens). We hash that payload and store the resulting completion under
``.evalblink_cache/<sha256>.json`` so re-running an identical benchmark costs
nothing and skips the network entirely.

Two public functions: :func:`get` and :func:`set`. :func:`sha256_key` builds
the key from a request payload.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile

CACHE_DIR = ".evalblink_cache"


def sha256_key(payload: dict) -> str:
    """Stable SHA256 of a request payload.

    ``sort_keys`` makes the hash insensitive to dict ordering so the same
    logical request always maps to the same key.
    """
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _path(key: str) -> str:
    return os.path.join(CACHE_DIR, f"{key}.json")


def get(key: str) -> dict | None:
    """Return the cached value for ``key``, or ``None`` on a miss.

    A corrupt, truncated or unreadable entry also reads as a miss (it is
    overwritten on the next successful call) rather than aborting the run. The
    caller still validates the entry's contents.
    """
    path = _path(key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            value = json.load(f)
    except (OSError, ValueError, RecursionError):
        return None
    return value if isinstance(value, dict) and "response" in value else None


def set(key: str, value: dict) -> None:
    """Persist ``value`` under ``key`` atomically.

    Write to a unique temp file then ``os.replace`` (atomic on POSIX) so a
    concurrent reader never sees a half-written file — the runner now populates
    the cache from multiple worker threads, and two threads can race on the same
    key when distinct test cases render to an identical request.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=CACHE_DIR, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=4)
        os.replace(tmp, _path(key))
    except BaseException:
        # Never leave a stray temp file behind on failure.
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _is_file(name: str) -> bool:
    return os.path.isfile(os.path.join(CACHE_DIR, name))


def stats() -> dict:
    """Entry count and total size of the cache directory."""
    if not os.path.isdir(CACHE_DIR):
        return {"entries": 0, "size_bytes": 0}
    files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".json") and _is_file(f)]
    size = sum(os.path.getsize(os.path.join(CACHE_DIR, f)) for f in files)
    return {"entries": len(files), "size_bytes": size}


def clear() -> int:
    """Delete all cached responses. Returns number of files removed."""
    if not os.path.isdir(CACHE_DIR):
        return 0
    files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".json") and _is_file(f)]
    for f in files:
        os.remove(os.path.join(CACHE_DIR, f))
    return len(files)
