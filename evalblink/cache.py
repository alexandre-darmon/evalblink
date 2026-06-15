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
from typing import Optional

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


def get(key: str) -> Optional[dict]:
    """Return the cached value for ``key``, or ``None`` on a miss."""
    path = _path(key)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def set(key: str, value: dict) -> None:
    """Persist ``value`` under ``key``."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(_path(key), "w") as f:
        json.dump(value, f, indent=4)
