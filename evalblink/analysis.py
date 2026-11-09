"""Derives insights from a finished run's raw results.

Pure functions over the ``results`` list that ``runner.run`` returns — no I/O, no
network. This module owns the *decisions* the reporter is forbidden to make:
which combo wins on quality, which wins on value, and how each combo's quality
breaks down per tag. The reporter only formats what ``summarize`` returns.

A "combo" is one ``model × prompt`` entry of ``results``; each carries a ``score``
(pass-rate %, ``success/total*100``) and a ``test_cases`` list whose entries hold
``tags``, ``match_result`` and ``latency_s``.

It also owns the statistics: the 95 % interval on a score (:func:`score_interval`),
whether two scores differ beyond noise (:func:`within_noise`), latency
percentiles (:func:`latency_stats`) and cases whose repeats disagree
(:func:`flaky_cases`).
"""

from __future__ import annotations

import math

# Any tag whose pass-rate falls below this gets flagged (README: "below 70%").
WARN_THRESHOLD = 70.0

# Floor on cost so best-value ranking never divides by zero (free models cost $0).
_COST_FLOOR = 1e-4


# z for a two-sided 95 % interval.
_Z = 1.959963984540054


def case_key(tc) -> tuple:
    """A case result's identity: its test-case id, plus its repeat index when
    the run used ``repeats`` (records without ``sample`` are single runs)."""
    return (str(tc.get("id")), tc.get("sample") or 0)


def case_label(tc) -> str:
    """``conv_001``, or ``conv_001 #2`` for the second repeat."""
    sample = tc.get("sample")
    return str(tc.get("id")) if sample is None else f"{tc.get('id')} #{sample + 1}"


def wilson(passed: float, n: float) -> tuple[float, float] | None:
    """95 % Wilson score interval (as fractions) for ``passed`` out of ``n``.

    Unlike ``p ± 1.96·SE`` it stays inside 0–1 and is honest at 0 % and 100 %
    with few cases. ``None`` when ``n`` is 0.
    """
    if n <= 0:
        return None
    p = passed / n
    z2 = _Z * _Z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = _Z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def _scored_cases(combo) -> int:
    """Distinct test cases with at least one scored repeat — the sample size.

    Repeats steady each case's verdict, but they are not new cases: the
    interval stays as wide as the number of test cases allows.
    """
    return len(
        {
            str(tc.get("id"))
            for tc in combo.get("test_cases") or []
            if tc.get("match_score") is not None
        }
    )


def _rate_and_n(combo) -> tuple[float, int] | None:
    """``(pass rate as a fraction, number of scored test cases)``."""
    scored = combo.get("scored", combo.get("total"))
    if not scored:
        return None
    n = _scored_cases(combo) if combo.get("test_cases") else scored
    return combo.get("success", 0) / scored, n or scored


def score_interval(combo) -> tuple[float, float] | None:
    """The combo's 95 % interval on its score, in percent, or ``None``."""
    rate = _rate_and_n(combo)
    if rate is None:
        return None
    p, n = rate
    low, high = wilson(p * n, n)
    return low * 100, high * 100


def within_noise(a, b) -> bool:
    """Whether combos ``a`` and ``b``'s scores could differ by chance alone.

    Newcombe's hybrid-score interval (method 10) for the difference of two
    proportions, built from each side's Wilson interval: when it contains 0 the
    gap is within noise at 95 %. Treats the two samples as independent, which
    is conservative when they share test cases. A combo with nothing scored is
    never "within noise" (there is nothing to compare).
    """
    ra, rb = _rate_and_n(a), _rate_and_n(b)
    if ra is None or rb is None:
        return False
    (pa, na), (pb, nb) = ra, rb
    la, ua = wilson(pa * na, na)
    lb, ub = wilson(pb * nb, nb)
    d = pa - pb
    low = d - math.sqrt((pa - la) ** 2 + (ub - pb) ** 2)
    high = d + math.sqrt((ua - pa) ** 2 + (pb - lb) ** 2)
    return low <= 0 <= high


def _percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile (``q`` in 0–100) of sorted ``values``."""
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q / 100
    lo = math.floor(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def latency_stats(test_cases) -> dict:
    """``{latency_p50_s, latency_p95_s, latency_mean_s}`` over the candidate
    calls that returned a reply (``None`` each when none has a latency — an
    all-error combo, or a record from before latency was measured)."""
    values = sorted(
        float(tc["latency_s"])
        for tc in test_cases
        if isinstance(tc.get("latency_s"), (int, float))
        and not isinstance(tc.get("latency_s"), bool)
        and math.isfinite(tc["latency_s"])
    )
    if not values:
        return {"latency_p50_s": None, "latency_p95_s": None, "latency_mean_s": None}
    return {
        "latency_p50_s": round(_percentile(values, 50), 3),
        "latency_p95_s": round(_percentile(values, 95), 3),
        "latency_mean_s": round(sum(values) / len(values), 3),
    }


def latency_p50(combo) -> float | None:
    """The combo's median latency (s): stored, else computed from its cases."""
    value = combo.get("latency_p50_s")
    if value is None:
        value = latency_stats(combo.get("test_cases") or [])["latency_p50_s"]
    return value


def flaky_cases(combo) -> list[str]:
    """Test-case ids whose scored repeats disagree (some pass, some fail)."""
    verdicts: dict[str, set] = {}
    for tc in combo.get("test_cases") or []:
        if tc.get("match_score") is not None:
            verdicts.setdefault(str(tc.get("id")), set()).add(bool(tc["match_result"]))
    return sorted(cid for cid, v in verdicts.items() if len(v) > 1)


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


def fail_safe_score(combo) -> float:
    """Pass rate (%) with unscored cases counted as failures."""
    total = combo.get("total")
    if total is None:  # a hand-built combo without counts: its score is all we have
        return combo["score"]
    return combo.get("success", 0) / total * 100 if total else 0.0


def _gated(results, gate) -> list:
    """The combos the CI gate looks at: all, or those ``gate`` names."""
    gate = gate or {}
    return [
        c
        for c in results
        if gate.get("model") in (None, c["model"])
        and gate.get("prompt") in (None, c["prompt_id"])
    ]


def _brief(combo) -> dict:
    """A combo's identity, score and median latency, for the insights."""
    return {
        "model": combo["model"],
        "prompt_id": combo["prompt_id"],
        "score": combo["score"],
        "latency_p50_s": latency_p50(combo),
    }


def summarize(results, quality_threshold, gate=None) -> dict | None:
    """Single entry point: winners, the best combo's tag breakdown, and the gate.

    ``passed`` is the CI/CD verdict against ``quality_threshold``. It is
    fail-safe: ``gate_score`` is the best combo's pass rate with every unscored
    (pipeline-error) case counted as a failure, so a run where most calls failed
    can't pass on the few that were scored. ``gate`` (``{model?, prompt?}``)
    restricts the verdict to the combos it names; by default the best combo of
    the run decides. When ``quality_threshold`` is
    ``None`` no gate is configured, so ``passed`` is ``True`` — unless no case
    at all could be scored, which never passes (``run`` exits 1 then too).
    Returns ``None`` when ``results`` is empty.
    """
    if not results:
        return None
    top = best_quality(results)
    breakdown = tag_breakdown(top)
    # Combos whose score is statistically indistinguishable from the best one.
    tied = sorted(
        (c for c in results if c is not top and within_noise(top, c)),
        key=lambda c: -c["score"],
    )
    timed = [c for c in results if latency_p50(c) is not None]
    # Only a choice between combos is worth a line.
    fastest = (
        min(timed, key=lambda c: (latency_p50(c), -c["score"]))
        if len(timed) > 1
        else None
    )
    errors = sum(c.get("errors") or 0 for c in results)
    gated = _gated(results, gate)
    gate_score = max((fail_safe_score(c) for c in gated), default=0.0)
    total = sum(c.get("total") or 0 for c in results)
    nothing_scored = bool(total) and errors >= total
    return {
        "best_quality": top,
        "best_value": best_value(results),
        "best_interval": score_interval(top),
        # Just enough to name them: the full combos are already in `results`.
        "within_noise": [_brief(c) for c in tied],
        "fastest": _brief(fastest) if fastest else None,
        "flaky": sum(len(flaky_cases(c)) for c in results),
        "tag_breakdown": breakdown,
        "warnings": [r["tag"] for r in breakdown if r["warn"]],
        # Run total (every combo), matching the end-of-run pipeline-error warning.
        "errors": errors,
        "gate_score": gate_score,
        "passed": not nothing_scored
        and (quality_threshold is None or gate_score >= quality_threshold),
        "quality_threshold": quality_threshold,
        "gate": gate or None,
    }
