"""The three evaluators: exact_match, weighted_match, llm_judge.

Pure scoring logic. ``evaluate_llm_judge`` makes an API call, but it receives
the ``httpx.Client`` as a parameter rather than creating one — that keeps this
module testable in isolation and free of any runner dependency.
"""

from __future__ import annotations

import json
import re

import httpx

from .openrouter import openrouter_request


def _extract_json(raw):
    """Parse JSON from a model response, tolerating a markdown code fence.

    Models often wrap JSON in ```` ```json … ``` ````. Strip a leading/trailing
    fence before parsing so both the judge and weighted_match accept fenced output.
    Raises ``json.JSONDecodeError`` if the stripped text still isn't valid JSON.
    """
    text = raw.strip()
    if text.startswith("```"):
        # Strip the opening fence (with optional language tag) and the closing fence,
        # whether or not they sit on their own lines.
        text = re.sub(r"^```[a-zA-Z0-9]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    return json.loads(text)


def exact_match(response, expected):
    return response.strip().lower() == expected.strip().lower()


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
    client, evaluation_params, candidate_response, task, criteria, reference=None
):
    judge_model = evaluation_params.get("judge_model")
    if not judge_model:
        raise ValueError("Judge model not specified in evaluation parameters.")

    threshold = evaluation_params.get("judge_threshold", 0.70)

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

    # The judge must be deterministic regardless of candidate inference settings.
    # Use a dedicated max_tokens — reasoning + JSON needs more room than the
    # candidate's configured max_tokens (e.g. 50).
    try:
        judge_response = openrouter_request(
            client,
            prompt=judge_prompt,
            model=judge_model,
            temperature=0,
            max_tokens=512,
        )
    except (RuntimeError, httpx.HTTPError):
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
        }

    from_cache = judge_response.get("from_cache", False)
    raw = judge_response["response"]
    judge_prompt_tokens = judge_response["prompt_tokens"]
    judge_completion_tokens = judge_response["completion_tokens"]
    judge_cost = judge_response["cost"]
    try:
        parsed = _extract_json(raw)
        score = int(parsed["score"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        # Judge returned malformed JSON / no usable score — pipeline failure.
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

    # Valid JSON of the wrong shape (not a list of objects with the scored keys)
    # is also a task failure — guard so the scoring below can't raise and abort.
    if not isinstance(parsed, list) or not all(
        isinstance(item, dict) and {"use_case", "percent", "order"} <= item.keys()
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
        if label in response_lookup:
            if response_lookup[label]["order"] == exp_item["order"]:
                matched_order += 1
    order_score = matched_order / len(expected)

    return (
        label_score * param_map["use_case"]["weight"]
        + percent_score * param_map["percent"]["weight"]
        + order_score * param_map["order"]["weight"]
    )
