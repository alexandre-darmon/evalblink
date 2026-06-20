"""Executes a benchmark config and returns raw results.

One job: own the ``httpx.Client`` and the model × prompt × test-case loop, call
OpenRouter for each candidate, route the response to the right evaluator, and
accumulate per-prompt results and token/cost totals. It makes no output
decisions — that's the reporter's job.
"""

from __future__ import annotations

import datetime
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import httpx
from jinja2 import Template

from .evaluator import (
    evaluate_llm_judge,
    exact_match,
    judge_vendor_warning,
    weighted_match,
)
from .openrouter import openrouter_request


def render_template(prompt, variables, test_case):
    if variables:
        render_kwargs = dict(variables)
    else:
        render_kwargs = {}
    if test_case.get("variables"):
        render_kwargs.update(test_case["variables"])
    rendered_prompt = Template(prompt["template"]).render(**render_kwargs)
    rendered_system = (
        Template(prompt["system"]).render(**render_kwargs)
        if prompt.get("system")
        else None
    )
    return rendered_prompt, rendered_system


def _run_case(client, model, prompt, config, test_case, verbose, use_cache):
    """Run one (model × prompt × test_case) triple and return a tagged result.

    Returned dict has ``"model"``, ``"prompt_id"``, and ``"test_case"`` keys.
    The caller collects these and re-groups them after all futures complete.
    """
    print(f"Test case: {test_case['id']}")
    rendered, rendered_system = render_template(
        prompt, config.get("variables"), test_case
    )
    if verbose:
        print(f"Rendered prompt: {rendered}")
    response = openrouter_request(
        client,
        rendered,
        model,
        config["inference"]["temperature"],
        config["inference"]["max_tokens"],
        rendered_system,
        use_cache=use_cache,
    )
    # Rate-limit only real API calls — cache hits need no backoff.
    if not response.get("from_cache"):
        time.sleep(5)
    if verbose:
        print(f"Raw response: {response['response']}")

    match_result = False
    match_score = None
    status = "scored"
    reasoning = None
    judge_prompt_tokens = 0
    judge_completion_tokens = 0
    judge_cost = 0.0
    evaluation = test_case.get("evaluation")

    if evaluation == "exact_match":
        match_result = exact_match(response["response"], test_case["expected_output"])
        match_score = 1.0 if match_result else 0.0
    elif evaluation == "weighted_match":
        match_score = weighted_match(
            config["evaluation"],
            response["response"],
            test_case["expected_output"],
        )
        threshold = (
            config["evaluation"]["quality_threshold"]
            if "quality_threshold" in config["evaluation"]
            else 0.70
        )
        match_result = match_score >= threshold
        if verbose:
            print(
                f"Match score: {match_score:.4f} (threshold: {threshold:.4f}) match result: {match_result}"
            )
    elif evaluation == "llm_judge":
        judge_result = evaluate_llm_judge(
            client,
            config["evaluation"],
            response["response"],
            rendered,  # full rendered prompt — no variable-name coupling
            test_case["criteria"],
            test_case.get("reference"),
        )
        if not judge_result.get("from_cache"):
            time.sleep(5)
        match_result = judge_result["match_result"]
        match_score = judge_result["score_normalized"]
        # Carry the judge's status so a pipeline failure (score=None)
        # is distinguished downstream from a real score of 0.
        status = judge_result["status"]
        reasoning = judge_result["reasoning"]
        judge_prompt_tokens = judge_result["judge_prompt_tokens"]
        judge_completion_tokens = judge_result["judge_completion_tokens"]
        judge_cost = judge_result["judge_cost"]
        if verbose:
            print(
                f"Judge status: {judge_result['status']} "
                f"raw: {judge_result['score_raw']} "
                f"score: {match_score} match result: {match_result}"
            )
    else:
        raise ValueError(f"Unknown evaluation type: {evaluation!r}")

    if verbose:
        print(
            f"Response: {response['response']} Prompt tokens: {response['prompt_tokens']} "
            f"Completion tokens: {response['completion_tokens']} Cost: ${response['cost']:.6f}"
        )
        print(f"Expected: {test_case.get('expected_output')}")
        print(f"Match result: {match_result}\n")

    cost = response["cost"]
    return {
        "model": model,
        "prompt_id": prompt["id"],
        "test_case": {
            "id": test_case["id"],
            "tags": test_case.get("tags") or [],
            "evaluation": evaluation,
            "status": status,
            "match_result": match_result,
            "match_score": match_score,
            "response": response["response"],
            "reasoning": reasoning,
            "expected": test_case.get("expected_output"),
            "prompt_tokens": response["prompt_tokens"],
            "completion_tokens": response["completion_tokens"],
            "cost": cost,
            "judge_prompt_tokens": judge_prompt_tokens,
            "judge_completion_tokens": judge_completion_tokens,
            "judge_cost": judge_cost,
            "total_cost": cost + judge_cost,
        },
    }


def run(config, verbose=False, use_cache=True):
    """Run every model × prompt × test case concurrently and return ``(results, timestamp)``.

    ``concurrency`` (config key, default 5) controls the ``ThreadPoolExecutor``
    worker count. ``verbose`` enables per-test-case detail; by default only
    per-combo summaries are printed after all cases complete.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    concurrency = config.get("concurrency", 5)

    warning = judge_vendor_warning(
        config.get("evaluation", {}).get("judge_model"), config["models"]
    )
    if warning:
        print(f"{warning}\n")

    # Build the full task list in config order so futures preserve that order.
    tasks = [
        (model, prompt, test_case)
        for model in config["models"]
        for prompt in config["prompts"]
        for test_case in config["test_cases"]
    ]

    with httpx.Client() as client:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(
                    _run_case,
                    client,
                    model,
                    prompt,
                    config,
                    test_case,
                    verbose,
                    use_cache,
                )
                for model, prompt, test_case in tasks
            ]
            # .result() blocks until each future completes; submission order is preserved.
            case_results = [f.result() for f in futures]

    # Re-group by (model, prompt_id), maintaining config order within each group.
    groups: dict = defaultdict(list)
    for cr in case_results:
        groups[(cr["model"], cr["prompt_id"])].append(cr["test_case"])

    results = []
    for model in config["models"]:
        print(f"Model: {model}\n")
        for prompt in config["prompts"]:
            test_case_results = groups[(model, prompt["id"])]
            success = sum(1 for tc in test_case_results if tc["match_result"])
            errors = sum(1 for tc in test_case_results if tc["match_score"] is None)
            scored = len(test_case_results) - errors
            total = len(test_case_results)
            total_prompt_tokens = sum(tc["prompt_tokens"] for tc in test_case_results)
            total_completion_tokens = sum(
                tc["completion_tokens"] for tc in test_case_results
            )
            total_cost = sum(tc["total_cost"] for tc in test_case_results)
            # Quality is over genuinely-scored cases only.
            score = success / scored * 100 if scored else 0.0
            results.append(
                {
                    "model": model,
                    "prompt_id": prompt["id"],
                    "success": success,
                    "scored": scored,
                    "errors": errors,
                    "total": total,
                    "score": score,
                    "total_prompt_tokens": total_prompt_tokens,
                    "total_completion_tokens": total_completion_tokens,
                    "total_cost": total_cost,
                    "test_cases": test_case_results,
                }
            )
            print(f"Prompt: {prompt['id']}")
            err_note = f", {errors} error(s)" if errors else ""
            print(f"Quality score: {success}/{scored} scored ({score:.1f}%){err_note}")
            print(f"Total prompt tokens: {total_prompt_tokens}")
            print(f"Total completion tokens: {total_completion_tokens}")
            print(f"Total cost: ${total_cost:.6f}")

    return results, timestamp
