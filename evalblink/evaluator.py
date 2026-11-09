"""The evaluators: the text checks (exact_match, contains, not_contains,
regex), json_match, json_schema, user-written Python scorers, weighted_match
and llm_judge.

Pure scoring logic (a Python scorer runs the user's own code). ``evaluate_llm_judge`` makes an API call, but it receives
the ``httpx.Client`` as a parameter rather than creating one — that keeps this
module testable in isolation and free of any runner dependency.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import threading
import unicodedata

import httpx
import jsonschema

from .openrouter import is_number, openrouter_request
from .schemas import DEFAULT_JUDGE_MAX_TOKENS


def _extract_json(raw):
    """Parse JSON from a model response, tolerating a markdown code fence.

    Models often wrap JSON in ```` ```json … ``` ````. Strip a leading/trailing
    fence before parsing so both the judge and weighted_match accept fenced output.
    Raises ``json.JSONDecodeError`` if the stripped text still isn't valid JSON
    — including JSON Python refuses to build (nested thousands deep, or an
    integer literal past the 4300-digit int-conversion limit).
    """
    text = raw.strip()
    if text.startswith("```"):
        # Strip the opening fence (with optional language tag) and the closing fence,
        # whether or not they sit on their own lines.
        text = re.sub(r"^```[a-zA-Z0-9]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise
    except (RecursionError, ValueError) as exc:  # "[[[[…" / a 5000-digit int
        raise json.JSONDecodeError(f"unparseable JSON ({exc})", text, 0) from None


def exact_match(response, expected):
    return response.strip().lower() == expected.strip().lower()


_PUNCTUATION = {
    **dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"), "-"),
    ord("\u2044"): "/",  # NFKC writes ½ as 1⁄2
}


def plain_text(text):
    """Text as a reader sees it, for ``contains`` / ``not_contains``.

    Models write "30\u202fdays", "30\u2011day" and "STAFF\u00ad50". NFKC turns
    Unicode spaces into " " (and fullwidth letters into ASCII), every dash
    becomes "-", invisible format characters (soft hyphen, zero-width space)
    are dropped, runs of whitespace become one space, and case is folded.
    """
    text = unicodedata.normalize("NFKC", text).translate(_PUNCTUATION)
    text = "".join(c for c in text if unicodedata.category(c) != "Cf")
    return " ".join(text.split()).casefold()


def _snippets(expected):
    """``expected_output`` as a list of plain texts."""
    texts = [expected] if isinstance(expected, str) else expected
    return [plain_text(s) for s in texts]


def contains(response, expected):
    """Every expected snippet appears in the reply (see :func:`plain_text`)."""
    text = plain_text(response)
    return all(s in text for s in _snippets(expected))


def not_contains(response, expected):
    """No expected snippet appears in the reply (see :func:`plain_text`)."""
    text = plain_text(response)
    return not any(s in text for s in _snippets(expected))


def regex(response, pattern):
    """The pattern matches somewhere in the reply (``re.search``; ``(?i)`` for
    case-insensitive, ``^…$`` for the whole reply). Like ``exact_match``, the
    reply's surrounding whitespace is ignored, so ``^…$`` survives a stray
    leading or trailing newline. Otherwise the reply is matched as written, so
    a pattern can look for (or ban) a "—" or count characters; ``\\s`` already
    matches no-break spaces and ``\\W`` any space or dash."""
    return re.search(pattern, response.strip()) is not None


# Pass/fail checks of the reply against expected_output: scored 1.0 or 0.0.
TEXT_CHECKS = {
    "exact_match": exact_match,
    "contains": contains,
    "not_contains": not_contains,
    "regex": regex,
}


# ── Structured output ─────────────────────────────────────────────────────────


def _leaves(expected) -> int:
    """How many scalar values ``expected`` holds (an empty object/list counts as one)."""
    if isinstance(expected, dict):
        return sum(_leaves(v) for v in expected.values()) or 1
    if isinstance(expected, list):
        return sum(_leaves(v) for v in expected) or 1
    return 1


def _same_scalar(expected, actual) -> bool:
    """Strings ignore case and surrounding whitespace (like ``exact_match``);
    numbers compare by value (``1 == 1.0``) but never equal a bool."""
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected is actual
    if isinstance(expected, str):
        return isinstance(actual, str) and (
            expected.strip().casefold() == actual.strip().casefold()
        )
    if is_number(expected):
        return is_number(actual) and math.isclose(expected, actual, abs_tol=1e-9)
    return expected == actual  # None


def _matched(expected, actual) -> int:
    """How many of ``expected``'s leaves ``actual`` reproduces."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return 0
        if not expected:
            return 1
        return sum(
            _matched(v, actual[k]) if k in actual else 0 for k, v in expected.items()
        )
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return 0
        if not expected:
            return 1
        return sum(_matched(e, a) for e, a in zip(expected, actual))
    return 1 if _same_scalar(expected, actual) else 0


def json_match(response, expected) -> float:
    """Share (0–1) of ``expected``'s values that the reply's JSON reproduces.

    Objects match key by key, recursively; keys the reply adds are ignored, so
    ``expected_output`` lists only what matters. Arrays must have the same
    length and match element by element, in order. Strings ignore case and
    surrounding whitespace, numbers compare by value. A reply that isn't JSON
    (a markdown fence is tolerated) scores 0 — a task failure, not a pipeline
    error. The case passes only at 1.0.
    """
    try:
        parsed = _extract_json(response)
    except json.JSONDecodeError:
        return 0.0
    return _matched(expected, parsed) / _leaves(expected)


def json_schema(response, schema) -> tuple[bool, str | None]:
    """``(valid, reason)``: does the reply's JSON satisfy the JSON Schema?

    The schema's ``$schema`` picks the draft (2020-12 by default). ``reason``
    names the first violation, or why the reply isn't JSON.
    """
    try:
        parsed = _extract_json(response)
    except json.JSONDecodeError as exc:
        return False, f"reply is not JSON ({exc.msg})"
    validator = jsonschema.validators.validator_for(schema)(schema)
    error = jsonschema.exceptions.best_match(validator.iter_errors(parsed))
    if error is None:
        return True, None
    where = "/".join(map(str, error.absolute_path)) or "(root)"
    return False, f"{where}: {error.message}"[:500]


def schema_problem(schema) -> str | None:
    """Why ``schema`` isn't a usable JSON Schema, or ``None``."""
    if not isinstance(schema, (dict, bool)):
        return f"must be a JSON Schema object (got {type(schema).__name__})"
    try:
        jsonschema.validators.validator_for(schema).check_schema(schema)
    except jsonschema.exceptions.SchemaError as exc:
        return f"is not a valid JSON Schema: {exc.message}"
    return None


# ── Python scorers ────────────────────────────────────────────────────────────

_SCORERS: dict = {}
_SCORERS_LOCK = threading.Lock()


class ScorerError(Exception):
    """A Python scorer that can't be loaded, raised, or returned something unusable."""


def load_scorer(spec: str, base_dir: str = ""):
    """The function named by ``"file.py:function"`` (path relative to ``base_dir``).

    Each file is imported once per process, so module-level setup (a model, a
    lookup table) runs once. Raises :class:`ScorerError` with a one-line reason.
    """
    if not isinstance(spec, str) or ":" not in spec:
        raise ScorerError(
            f"scorer must look like 'scorers.py:function_name' (got {spec!r})"
        )
    file, _, name = spec.rpartition(":")
    path = os.path.abspath(os.path.join(base_dir or "", file.strip()))
    name = name.strip()
    with _SCORERS_LOCK:
        module = _SCORERS.get(path)
        if module is None:
            if not os.path.isfile(path):
                raise ScorerError(f"scorer file not found: {file}")
            module_name = (
                "evalblink_scorer_" + hashlib.sha256(path.encode()).hexdigest()[:12]
            )
            try:
                loader = importlib.util.spec_from_file_location(module_name, path)
                module = importlib.util.module_from_spec(loader)
                loader.loader.exec_module(module)
            except Exception as exc:  # noqa: BLE001 — the user's module, any failure
                raise ScorerError(
                    f"could not import {file}: {type(exc).__name__}: {exc}"
                ) from None
            _SCORERS[path] = module
    fn = getattr(module, name, None)
    if not callable(fn):
        raise ScorerError(f"{file} has no function {name!r}")
    return fn


def python_score(fn, response, case) -> tuple[bool, float, str | None]:
    """``(passed, score, reason)`` from a user scorer ``fn(response, case)``.

    ``fn`` returns ``True``/``False``, or a dict ``{"passed": bool, "score"?:
    0–1, "reason"?: str}`` (``score`` defaults to 1.0 / 0.0). Anything else,
    or an exception, raises :class:`ScorerError` — a pipeline error, never a
    silent 0.
    """
    try:
        out = fn(response, case)
    except Exception as exc:  # noqa: BLE001 — the user's code, any failure
        raise ScorerError(f"{type(exc).__name__}: {exc}") from None
    if isinstance(out, bool):
        return out, 1.0 if out else 0.0, None
    if isinstance(out, dict) and isinstance(out.get("passed"), bool):
        passed = out["passed"]
        score = out.get("score", 1.0 if passed else 0.0)
        if not (is_number(score) and 0 <= score <= 1):
            raise ScorerError(f"returned score {score!r} — it must be a number 0–1")
        reason = out.get("reason")
        return passed, float(score), None if reason is None else str(reason)
    raise ScorerError(
        f"returned {out!r} — return True/False or"
        " {'passed': bool, 'score': 0–1, 'reason': str}"
    )


def _vendor(model):
    """The provider prefix of an OpenRouter model id (text before the first '/')."""
    return model.split("/", 1)[0].strip().lower() if model else ""


def judge_vendor_warning(judge_model, models):
    """Warn when the judge shares a vendor with any candidate it grades.

    Self-preference bias: a judge tends to over-score responses from its own
    vendor (README: 10-25% magnitude). Returns a one-line warning naming the
    overlapping candidates, or ``None`` when there is no judge or no overlap.
    """
    judge_vendor = _vendor(judge_model)
    if not judge_vendor:
        return None
    overlap = [m for m in (models or []) if _vendor(m) == judge_vendor]
    if not overlap:
        return None
    return (
        f"⚠️  Self-preference risk: judge '{judge_model}' shares vendor "
        f"'{judge_vendor}' with candidate(s): {', '.join(overlap)}. "
        "Judge scores for these may be inflated."
    )


def evaluate_llm_judge(
    client,
    evaluation_params,
    candidate_response,
    task,
    criteria,
    reference=None,
    use_cache=True,
):
    judge_model = evaluation_params.get("judge_model")
    if not judge_model:
        raise ValueError("Judge model not specified in evaluation parameters.")

    threshold = evaluation_params.get("judge_threshold")
    if threshold is None:  # absent or a bare `judge_threshold:`
        threshold = 0.70

    reference_line = f"Reference answer: {reference}\n" if reference else ""
    judge_prompt = f"""You are an expert evaluator. Your goal is to assess quality, not style.
        Original task: {task}
        Evaluation criteria: {criteria}
        Model response: {candidate_response}
        {reference_line}
        Think step by step about whether the response meets the criteria.

        Important:
        - Judge on criteria satisfaction only — do not reward length.
        - Do not reward confident tone over accuracy.
        - The criteria were not shown to the model being judged.

        Then return ONLY valid JSON, no markdown:
        {{"reasoning": "<your analysis>", "score": <integer 1-5>}}"""

    # The judge must be deterministic regardless of candidate inference settings,
    # with its own budget: reasoning + JSON needs far more room than a candidate's
    # max_tokens (e.g. 50).
    judge_max_tokens = evaluation_params.get("judge_max_tokens")
    if judge_max_tokens is None:  # absent or a bare `judge_max_tokens:`
        judge_max_tokens = DEFAULT_JUDGE_MAX_TOKENS
    try:
        judge_response = openrouter_request(
            client,
            prompt=judge_prompt,
            model=judge_model,
            temperature=0,
            max_tokens=judge_max_tokens,
            use_cache=use_cache,
        )
    except (RuntimeError, httpx.HTTPError) as exc:
        # Judge API failed (error response, timeout, rate limit) — pipeline
        # failure, not a real evaluation. score is None, not 0.
        return {
            "status": "judge_api_error",
            "match_result": False,
            "score_raw": None,
            "score_normalized": None,
            "reasoning": None,
            "judge_model": judge_model,
            "judge_prompt_tokens": 0,
            "judge_completion_tokens": 0,
            "judge_cost": 0.0,
            "from_cache": False,
            # The judge's budget is its own setting, not inference.max_tokens.
            "error": (str(exc) or type(exc).__name__).replace(
                "Increase max_tokens.", "Increase evaluation.judge_max_tokens."
            ),
        }

    from_cache = judge_response.get("from_cache", False)
    raw = judge_response["response"]
    judge_prompt_tokens = judge_response["prompt_tokens"]
    judge_completion_tokens = judge_response["completion_tokens"]
    judge_cost = judge_response["cost"]
    try:
        parsed = _extract_json(raw)
        score = int(parsed["score"])
    except (KeyError, TypeError, ValueError, OverflowError):
        # Judge returned malformed JSON / no usable score — pipeline failure.
        # ValueError covers JSONDecodeError and int("high"); OverflowError is
        # int(Infinity) from a judge replying {"score": Infinity}.
        return {
            "status": "judge_error",
            "match_result": False,
            "score_raw": None,
            "score_normalized": None,
            "reasoning": None,
            "judge_model": judge_model,
            "judge_prompt_tokens": judge_prompt_tokens,
            "judge_completion_tokens": judge_completion_tokens,
            "judge_cost": judge_cost,
            "raw_response": raw,
            "from_cache": from_cache,
            "error": (
                f"judge reply has no usable score ({judge_completion_tokens}"
                " completion tokens; if it was cut off, raise"
                " evaluation.judge_max_tokens)"
            ),
        }

    score = max(1, min(5, score))
    normalized = (score - 1) / 4  # 1-5 → 0.0-1.0
    passed = normalized >= threshold
    return {
        "status": "success" if passed else "fail",
        "match_result": passed,
        "score_raw": score,
        "score_normalized": normalized,
        "reasoning": parsed.get("reasoning"),
        "judge_model": judge_model,
        "judge_prompt_tokens": judge_prompt_tokens,
        "judge_completion_tokens": judge_completion_tokens,
        "judge_cost": judge_cost,
        "from_cache": from_cache,
    }


def weighted_match(evaluation_params, response, expected, tolerance=0.20):
    param_map = {}
    for eval_param in evaluation_params["variables"]:
        name = eval_param["name"]
        param_map[name] = eval_param

    try:
        parsed = _extract_json(response)
    except json.JSONDecodeError:
        # Candidate failed to produce parseable JSON — that's a task failure
        # (score 0.0), not a pipeline error. Don't abort the whole run.
        return 0.0

    # Valid JSON of the wrong shape (not a list of objects with the scored keys,
    # a hashable use_case and a finite numeric percent) is also a task failure —
    # guard so the scoring below can't raise and abort.
    if not isinstance(parsed, list) or not all(
        isinstance(item, dict)
        and {"use_case", "percent", "order"} <= item.keys()
        and not isinstance(item["use_case"], (list, dict))
        and is_number(item["percent"])
        for item in parsed
    ):
        return 0.0

    # use_case score — F1 over expected vs found labels
    expected_labels = {item["use_case"] for item in expected}
    found_labels = {item["use_case"] for item in parsed}
    correct = len(expected_labels & found_labels)
    precision = correct / len(found_labels) if found_labels else 0
    recall = correct / len(expected_labels) if expected_labels else 0
    label_score = (
        2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    )

    response_lookup = {item["use_case"]: item for item in parsed}

    # percent score — within tolerance for matched labels
    matched_percent = 0
    for exp_item in expected:
        label = exp_item["use_case"]
        if label in response_lookup:
            tol = param_map["percent"].get("tolerance", tolerance)
            if abs(response_lookup[label]["percent"] - exp_item["percent"]) <= tol:
                matched_percent += 1
    percent_score = matched_percent / len(expected)

    # order score — exact match for matched labels
    matched_order = 0
    for exp_item in expected:
        label = exp_item["use_case"]
        if (
            label in response_lookup
            and response_lookup[label]["order"] == exp_item["order"]
        ):
            matched_order += 1
    order_score = matched_order / len(expected)

    return (
        label_score * param_map["use_case"]["weight"]
        + percent_score * param_map["percent"]["weight"]
        + order_score * param_map["order"]["weight"]
    )
