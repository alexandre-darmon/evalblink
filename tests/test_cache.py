"""Tests for the SHA256 file cache."""

from __future__ import annotations

from evalblink import cache


def test_sha256_key_is_order_insensitive():
    assert cache.sha256_key({"a": 1, "b": 2}) == cache.sha256_key({"b": 2, "a": 1})


def test_sha256_key_differs_for_different_payloads():
    assert cache.sha256_key({"a": 1}) != cache.sha256_key({"a": 2})


def test_set_get_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path))
    cache.set("key1", {"response": "hi", "cost": 0.0})
    assert cache.get("key1") == {"response": "hi", "cost": 0.0}


def test_get_miss_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", str(tmp_path))
    assert cache.get("does-not-exist") is None
