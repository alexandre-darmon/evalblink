"""Diffs two finished run records into a per-combo quality/cost delta.

Pure functions over the JSON records the reporter persists — no network, no
scoring. Like ``analysis``, this module owns *decisions* (which combos changed,
by how much, and the overall verdict); the reporter only formats what ``diff``
returns. A combo is keyed by ``(model, prompt_id)``; each carries a ``score``
(pass-rate %) and ``total_cost``.

Latency is intentionally absent: the runner does not capture it, so comparing it
would be inventing data. Quality and cost only.
"""

from __future__ import annotations

import json

# Score deltas smaller than this (in percentage points) read as "stable" — keeps
# floating-point noise from showing as spurious ↑/↓.
_SCORE_EPS = 0.05


def load_record(path) -> dict:
    """Load a persisted run record, defaulting a missing ``schema_version`` to 0.

    Records written before the field existed are legacy version 0; ``diff`` still
    compares them rather than hard-blocking.
    """
    with open(path) as f:
        record = json.load(f)
    record.setdefault("schema_version", 0)
    return record


def _index(record) -> dict:
    """Map ``(model, prompt_id)`` → combo for one record's ``results``."""
    return {(c["model"], c["prompt_id"]): c for c in record.get("results", [])}


def diff(record_a, record_b) -> dict:
    """Per-combo deltas from run A to run B over the union of their combos.

    Each row is ``{model, prompt_id, score_a, score_b, score_delta, cost_a,
    cost_b, cost_delta, present_in}`` where ``present_in`` is
    ``"both" | "a_only" | "b_only"``. Combos missing from one side carry ``None``
    for that side's score/cost and a ``None`` delta. Rows are sorted with
    changed-or-missing combos first, then by model/prompt name.
    """
    index_a = _index(record_a)
    index_b = _index(record_b)
    keys = sorted(set(index_a) | set(index_b))

    rows = []
    improved = 0
    comparable = 0
    for key in keys:
        a = index_a.get(key)
        b = index_b.get(key)
        if a and b:
            present_in = "both"
            score_delta = b["score"] - a["score"]
            cost_delta = b["total_cost"] - a["total_cost"]
            comparable += 1
            if score_delta > _SCORE_EPS:
                improved += 1
        elif b:
            present_in = "b_only"
            score_delta = cost_delta = None
        else:
            present_in = "a_only"
            score_delta = cost_delta = None
        rows.append(
            {
                "model": key[0],
                "prompt_id": key[1],
                "score_a": a["score"] if a else None,
                "score_b": b["score"] if b else None,
                "score_delta": score_delta,
                "cost_a": a["total_cost"] if a else None,
                "cost_b": b["total_cost"] if b else None,
                "cost_delta": cost_delta,
                "present_in": present_in,
            }
        )

    rows.sort(
        key=lambda r: (
            r["present_in"] == "both" and _is_stable(r),
            r["model"],
            r["prompt_id"],
        )
    )

    schema_a = record_a.get("schema_version", 0)
    schema_b = record_b.get("schema_version", 0)
    version_note = (
        f"Record schema versions differ (A=v{schema_a}, B=v{schema_b}); "
        "comparing on shared fields only."
        if schema_a != schema_b
        else None
    )

    return {
        "run_a": record_a.get("run_id"),
        "run_b": record_b.get("run_id"),
        "schema_a": schema_a,
        "schema_b": schema_b,
        "version_note": version_note,
        "rows": rows,
        "verdict": _verdict(improved, comparable, rows),
    }


def _is_stable(row) -> bool:
    """Whether a both-sides row's quality is unchanged (for sort ordering)."""
    delta = row["score_delta"]
    return delta is not None and abs(delta) <= _SCORE_EPS


def _verdict(improved, comparable, rows) -> str:
    """One-line summary of B versus A."""
    only = [r for r in rows if r["present_in"] != "both"]
    only_note = f" {len(only)} combo(s) present in only one run." if only else ""
    if not comparable:
        return "No combos in common between the two runs." + only_note
    if improved == comparable:
        return f"Run B improves quality on all {comparable} shared combo(s).{only_note}"
    if improved == 0:
        return f"Run B improves quality on no shared combo(s).{only_note}"
    return (
        f"Run B improves quality on {improved}/{comparable} shared combo(s).{only_note}"
    )


# ---------------------------------------------------------------------------
# Test-level diff (the --detailed view)
# ---------------------------------------------------------------------------
#
# Outcome of a single case, read the same way analysis.py distinguishes a
# pipeline error from a real 0: ``match_score is None`` means the judge
# API/parse failed — it is an "error", never a real fail.


def _state(tc) -> str:
    """One of ``"pass" | "fail" | "error"`` for a test-case result dict."""
    if tc.get("match_score") is None:
        return "error"
    return "pass" if tc.get("match_result") else "fail"


def _classify_case(a, b) -> str:
    """Transition label for a case present in both runs (A → B)."""
    sa, sb = _state(a), _state(b)
    if sa == "error" and sb == "error":
        return "still_error"
    if sa != "error" and sb == "error":
        return "new_error"
    if sa == "error" and sb != "error":
        return "recovered"
    # Both scored.
    if sa == "pass" and sb == "fail":
        return "regressed"
    if sa == "fail" and sb == "pass":
        return "improved"
    return "unchanged"


# Transitions the reporter lists as a "change"; ``unchanged``/``still_error``
# collapse into the footer count.
CHANGED_TRANSITIONS = {
    "regressed",
    "improved",
    "new_error",
    "recovered",
    "added",
    "removed",
}


def case_diff(combo_a, combo_b) -> dict:
    """Per-test-case transitions within one shared ``(model, prompt)`` combo.

    Indexes each combo's ``test_cases`` by ``id`` and classifies the union. Each
    row is ``{id, tags, transition, a_state, b_state, score_a, score_b,
    score_delta, present_in}``; ``rollup`` tallies every transition kind.
    """
    by_id_a = {tc["id"]: tc for tc in combo_a.get("test_cases", [])}
    by_id_b = {tc["id"]: tc for tc in combo_b.get("test_cases", [])}

    rollup = {
        k: 0
        for k in (
            "regressed",
            "improved",
            "unchanged",
            "new_error",
            "recovered",
            "still_error",
            "added",
            "removed",
        )
    }
    rows = []
    for case_id in sorted(set(by_id_a) | set(by_id_b)):
        a = by_id_a.get(case_id)
        b = by_id_b.get(case_id)
        if a and b:
            transition = _classify_case(a, b)
            present_in = "both"
            sa, sb = _state(a), _state(b)
            score_a, score_b = a.get("match_score"), b.get("match_score")
            score_delta = (
                score_b - score_a
                if score_a is not None and score_b is not None
                else None
            )
            tags = b.get("tags") or a.get("tags") or []
        elif b:
            transition, present_in = "added", "b_only"
            sa, sb = None, _state(b)
            score_a, score_b, score_delta = None, b.get("match_score"), None
            tags = b.get("tags") or []
        else:
            assert a is not None  # union guarantees the case is in A when not in B
            transition, present_in = "removed", "a_only"
            sa, sb = _state(a), None
            score_a, score_b, score_delta = a.get("match_score"), None, None
            tags = a.get("tags") or []
        rollup[transition] += 1
        rows.append(
            {
                "id": case_id,
                "tags": tags,
                "transition": transition,
                "a_state": sa,
                "b_state": sb,
                "score_a": score_a,
                "score_b": score_b,
                "score_delta": score_delta,
                "present_in": present_in,
            }
        )
    return {"rows": rows, "rollup": rollup}


def detailed_diff(record_a, record_b) -> dict:
    """Combo diff plus a per-case breakdown and a global summary.

    Extends :func:`diff` with ``case_combos`` (one entry per *shared* combo,
    carrying its :func:`case_diff`) and ``summary`` — change totals + verdict,
    the cases that regressed across the most combos, and net change per tag.
    """
    result = diff(record_a, record_b)
    index_a = _index(record_a)
    index_b = _index(record_b)

    case_combos = []
    totals = {"regressed": 0, "improved": 0, "new_error": 0, "recovered": 0}
    regressed_by_id: dict[str, int] = {}
    tag_net: dict[str, int] = {}

    for row in result["rows"]:
        if row["present_in"] != "both":
            continue
        key = (row["model"], row["prompt_id"])
        cd = case_diff(index_a[key], index_b[key])
        case_combos.append({"model": key[0], "prompt_id": key[1], **cd})
        for k in totals:
            totals[k] += cd["rollup"][k]
        for case_row in cd["rows"]:
            if case_row["transition"] == "regressed":
                regressed_by_id[case_row["id"]] = (
                    regressed_by_id.get(case_row["id"], 0) + 1
                )
                for tag in case_row["tags"]:
                    tag_net[tag] = tag_net.get(tag, 0) - 1
            elif case_row["transition"] == "improved":
                for tag in case_row["tags"]:
                    tag_net[tag] = tag_net.get(tag, 0) + 1

    worst_regressed = [
        {"id": case_id, "count": n}
        for case_id, n in sorted(
            regressed_by_id.items(), key=lambda kv: (-kv[1], kv[0])
        )
    ]
    tag_net_rows = [
        {"tag": tag, "net": net}
        for tag, net in sorted(tag_net.items(), key=lambda kv: (kv[1], kv[0]))
    ]

    result["case_combos"] = case_combos
    result["summary"] = {
        **totals,
        "net": totals["improved"] - totals["regressed"],
        "worst_regressed": worst_regressed,
        "tag_net": tag_net_rows,
        "verdict": _case_verdict(totals),
    }
    return result


def _case_verdict(totals) -> str:
    """One-line synthesis of the test-level change counts."""
    net = totals["improved"] - totals["regressed"]
    if not any(totals.values()):
        return "No test-case changes across shared combos."
    if net > 0:
        direction = f"net +{net} case(s) improved"
    elif net < 0:
        direction = f"net {net} case(s) regressed"
    else:
        direction = "no net quality change"
    err = ""
    if totals["new_error"]:
        err = f"; {totals['new_error']} new pipeline error(s)"
    return (
        f"Run B: {totals['improved']} improved, {totals['regressed']} regressed "
        f"({direction}){err}."
    )
