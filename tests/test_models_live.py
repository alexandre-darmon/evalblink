"""Network guard: every model ID shipped in benchmarks/ and the docs exists on OpenRouter.

OpenRouter renames and retires models (`:free` variants especially), which
silently breaks the examples. Skipped unless EVALBLINK_NETWORK_TESTS=1; runs
weekly in .github/workflows/models-check.yml. Uses the public /models endpoint,
no API key.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import httpx
import pytest
import yaml

from evalblink import starter

pytestmark = pytest.mark.skipif(
    os.environ.get("EVALBLINK_NETWORK_TESTS") != "1",
    reason="network test; set EVALBLINK_NETWORK_TESTS=1",
)

ROOT = Path(__file__).resolve().parent.parent
# conftest blocks real HTTP for every test; this one is the deliberate exception.
_REAL_SEND = httpx.Client.send


def _configs():
    for path in sorted((ROOT / "benchmarks").glob("*.yaml")):
        yield path.name, yaml.safe_load(path.read_text())
    for doc in ("README.md", "PRD.md"):
        for block in re.findall(
            r"```yaml\n(.*?)```", (ROOT / doc).read_text(), flags=re.DOTALL
        ):
            data = yaml.safe_load(block)
            if isinstance(data, dict):
                yield doc, data


def _referenced_models():
    # `evalblink init` writes these into every starter benchmark.
    refs = {
        ("starter.py", starter.DEFAULT_MODEL),
        ("starter.py", starter.JUDGE_MODEL),
        ("starter.py", starter.PAID_MODEL),
    }
    for source, cfg in _configs():
        for model in cfg.get("models") or []:
            refs.add((source, model))
        judge = (cfg.get("evaluation") or {}).get("judge_model")
        if judge:
            refs.add((source, judge))
    return sorted(refs)


def test_every_referenced_model_exists(monkeypatch):
    monkeypatch.setattr(httpx.Client, "send", _REAL_SEND)
    resp = httpx.get("https://openrouter.ai/api/v1/models", timeout=30)
    resp.raise_for_status()
    catalog = {m["id"] for m in resp.json()["data"]}
    refs = _referenced_models()
    assert refs, "no model references found — extraction is broken"
    missing = [f"{model} ({source})" for source, model in refs if model not in catalog]
    assert not missing, "Not on OpenRouter anymore:\n" + "\n".join(missing)
