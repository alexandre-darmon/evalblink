"""Derives insights from a finished run's raw results.

Pure functions over the ``results`` list that ``runner.run`` returns — no I/O, no
network. This module owns the *decisions* the reporter is forbidden to make:
which combo wins on quality, which wins on value, and how each combo's quality
breaks down per tag. The reporter only formats what ``summarize`` returns.

A "combo" is one ``model × prompt`` entry of ``results``; each carries a ``score``
(pass-rate %, ``success/total*100``) and a ``test_cases`` list whose entries hold
``tags`` and ``match_result``.
"""

from __future__ import annotations

# Any tag whose pass-rate falls below this gets flagged (README: "below 70%").
WARN_THRESHOLD = 70.0

# Floor on cost so best-value ranking never divides by zero (free models cost $0).
_COST_FLOOR = 1e-4


def tag_breakdown(combo) -> list[dict]:
    """Per-tag pass-rate for one combo.

    Returns ``[{"tag", "cases", "scored", "errors", "quality", "warn"}, ...]``. A
    test case carrying several tags counts toward each of them. Quality is the
    pass-rate over **scored** cases only (mean of ``match_result`` as 0/1);
    pipeline failures (``match_score is None`` — a judge API/parse error) are
    excluded from the denominator and tallied under ``errors`` rather than counted
    as a real failure. A tag with no scored cases has ``quality=None`` and never
    warns. Untagged cases are omitted. Sorted warnings-first, then by tag name.
    """
    passed: dict[str, int] = {}
    scored: dict[str, int] = {}
    counts: dict[str, int] = {}
    for tc in combo["test_cases"]:
        errored = tc.get("match_score") is None
        for tag in tc.get("tags") or []:
            counts[tag] = counts.get(tag, 0) + 1
            if errored:
                continue
            scored[tag] = scored.get(tag, 0) + 1
            if tc["match_result"]:
                passed[tag] = passed.get(tag, 0) + 1

    rows = []
    for tag, n in counts.items():
        n_scored = scored.get(tag, 0)
        quality = passed.get(tag, 0) / n_scored * 100 if n_scored else None
        warn = quality is not None and quality < WARN_THRESHOLD
        rows.append(
            {
                "tag": tag,
                "cases": n,
                "scored": n_scored,
                "errors": n - n_scored,
                "quality": quality,
                "warn": warn,
            }
        )
    rows.sort(key=lambda r: (not r["warn"], r["tag"]))
    return rows


def best_quality(results) -> dict:
    """Combo with the highest score; ties broken by lower total cost."""
    return max(results, key=lambda c: (c["score"], -c["total_cost"]))


def best_value(results) -> dict:
    """Combo with the best score-per-dollar.

    ``score / max(total_cost, _COST_FLOOR)`` — with free models (cost 0) the floor
    makes this fall back to the highest-scoring combo. A documented heuristic; easy
    to retune later.
    """
    return max(results, key=lambda c: c["score"] / max(c["total_cost"], _COST_FLOOR))


def summarize(results, quality_threshold) -> dict | None:
    """Single entry point: winners, the best combo's tag breakdown, and the gate.

    ``passed`` is the CI/CD verdict — the best combo's score vs ``quality_threshold``.
    When ``quality_threshold`` is ``None`` no gate is configured, so ``passed`` is
    ``True``. Returns ``None`` when ``results`` is empty.
    """
    if not results:
        return None
    top = best_quality(results)
    breakdown = tag_breakdown(top)
    return {
        "best_quality": top,
        "best_value": best_value(results),
        "tag_breakdown": breakdown,
        "warnings": [r["tag"] for r in breakdown if r["warn"]],
        "errors": top.get("errors", 0),
        "passed": quality_threshold is None or top["score"] >= quality_threshold,
        "quality_threshold": quality_threshold,
    }
