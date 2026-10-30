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

from .openrouter import is_number

# Score deltas smaller than this (in percentage points) read as "stable" — keeps
# floating-point noise from showing as spurious ↑/↓.
_SCORE_EPS = 0.05


class RecordError(ValueError):
    """A run record that can't be read or isn't the shape ``reporter.write`` persists."""


def _record_problem(record) -> str | None:
    """What makes ``record`` unusable by report / compare / history, or ``None``.

    Checks exactly the fields those commands read, with the types they need:
    strings where they are sorted or used as keys/filenames, finite numbers where
    they are formatted or subtracted.
    """
    if not isinstance(record, dict):
        return f"expected a JSON object, got {type(record).__name__}"
    for key in ("benchmark", "timestamp"):
        if not isinstance(record.get(key), str):
            return f"'{key}' is missing or not a string"
    if record.get("run_id") is not None and not isinstance(record["run_id"], str):
        return "'run_id' is not a string"
    results = record.get("results")
    if not isinstance(results, list):
        return "'results' is missing or not a list"
    for i, combo in enumerate(results):
        where = f"results[{i}]"
        if not isinstance(combo, dict):
            return f"{where} is not an object"
        for key in ("model", "prompt_id"):
            if not isinstance(combo.get(key), str):
                return f"{where}.{key} is missing or not a string"
        for key in (
            "score",
            "total_cost",
            "success",
            "total",
            "total_prompt_tokens",
            "total_completion_tokens",
        ):
            if not is_number(combo.get(key)):
                return f"{where}.{key} is missing or not a finite number"
        for key in ("scored", "errors"):
            if combo.get(key) is not None and not is_number(combo[key]):
                return f"{where}.{key} is not a finite number"
        cases = combo.get("test_cases")
        if not isinstance(cases, list):
            return f"{where}.test_cases is missing or not a list"
        for j, tc in enumerate(cases):
            here = f"{where}.test_cases[{j}]"
            if not isinstance(tc, dict) or not isinstance(tc.get("id"), str):
                return f"{here} is not an object with a string 'id'"
            if "match_score" not in tc:
                return f"{here}.match_score is missing"
            if tc["match_score"] is not None and not is_number(tc["match_score"]):
                return f"{here}.match_score is not a finite number"
            if tc["match_score"] is not None and not isinstance(
                tc.get("match_result"), bool
            ):
                return f"{here}.match_result is missing or not true/false"
            if tc.get("status") is not None and not isinstance(tc["status"], str):
                return f"{here}.status is not a string"
            if not is_number(tc.get("total_cost")):
                return f"{here}.total_cost is missing or not a finite number"
            tags = tc.get("tags")
            if tags is not None and not (
                isinstance(tags, list) and all(isinstance(t, str) for t in tags)
            ):
                return f"{here}.tags is not a list of strings"
    return None


def load_record(path) -> dict:
    """Load and shape-check a persisted run record.

    Raises :class:`RecordError` (with a one-line reason) for a missing,
    unreadable, non-JSON or wrongly-shaped file, so every caller can fail
    cleanly. A missing ``schema_version`` defaults to 0: records written before
    the field existed are legacy version 0; ``diff`` still compares them rather
    than hard-blocking.
    """
    try:
        with open(path, encoding="utf-8") as f:
            record = json.load(f)
    except FileNotFoundError:
        raise RecordError(f"Run record not found: {path}") from None
    except OSError as exc:  # a directory, no permission, ...
        raise RecordError(
            f"Cannot read run record {path}: {exc.strerror or exc}"
        ) from None
    except (ValueError, RecursionError) as exc:  # not JSON / not UTF-8 / too deep
        raise RecordError(f"{path} is not a valid JSON run record: {exc}") from None
    problem = _record_problem(record)
    if problem:
        raise RecordError(f"{path} is not an evalblink run record: {problem}")
    record.setdefault("schema_version", 0)
    return record


def _index(record) -> dict:
    """Map ``(model, prompt_id)`` → combo for one record's ``results``."""
    return {(c["model"], c["prompt_id"]): c for c in record.get("results", [])}


def diff(record_a, record_b) -> dict:
    """Per-combo deltas from run A to run B over the union of their combos.

    Each row is ``{model, prompt_id, score_a, score_b, score_delta, cost_a,
    cost_b, cost_delta, errors_a, errors_b, present_in}`` where ``present_in``
    is ``"both" | "a_only" | "b_only"``. Combos missing from one side carry
    ``None`` for that side's score/cost/errors and a ``None`` delta. Scores
    exclude pipeline-error cases, so a shared combo whose error count differs
    is not like-for-like; the verdict says how many there are. Rows are sorted with
    changed-or-missing combos first, then by model/prompt name.
    """
    index_a = _index(record_a)
    index_b = _index(record_b)
    keys = sorted(set(index_a) | set(index_b))

    rows = []
    improved = 0
    comparable = 0
    uneven = 0
    for key in keys:
        a = index_a.get(key)
        b = index_b.get(key)
        if a and b:
            present_in = "both"
            score_delta = b["score"] - a["score"]
            cost_delta = b["total_cost"] - a["total_cost"]
            # Scores exclude pipeline errors: a combo whose error count changed
            # was scored over different cases, so it isn't counted either way.
            if (a.get("errors") or 0) != (b.get("errors") or 0):
                uneven += 1
            else:
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
                "errors_a": (a.get("errors") or 0) if a else None,
                "errors_b": (b.get("errors") or 0) if b else None,
                "present_in": present_in,
            }
        )

    rows.sort(
        key=lambda r: (
            r["present_in"] == "both" and _is_stable(r) and not _uneven(r),
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
        "verdict": _verdict(improved, comparable, uneven, rows),
    }


def _is_stable(row) -> bool:
    """Whether a both-sides row's quality is unchanged (for sort ordering)."""
    delta = row["score_delta"]
    return delta is not None and abs(delta) <= _SCORE_EPS


def _uneven(row) -> bool:
    """A shared combo whose pipeline-error count changed between the runs."""
    return row["present_in"] == "both" and row["errors_a"] != row["errors_b"]


def _verdict(improved, comparable, uneven, rows) -> str:
    """One-line summary of B versus A over like-for-like shared combos."""
    only = [r for r in rows if r["present_in"] != "both"]
    only_note = f" {len(only)} combo(s) present in only one run." if only else ""
    if uneven:
        only_note += (
            f" {uneven} shared combo(s) left out: their number of pipeline errors"
            " (excluded from scores) changed, so their quality Δ is not"
            " like-for-like."
        )
    if not comparable:
        if uneven:
            return "No like-for-like shared combos." + only_note
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
