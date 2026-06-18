"""Executes a benchmark config and returns raw results.

One job: own the ``httpx.Client`` and the model × prompt × test-case loop, call
OpenRouter for each candidate, route the response to the right evaluator, and
accumulate per-prompt results and token/cost totals. It makes no output
decisions — that's the reporter's job.
"""

from __future__ import annotations

import datetime
import time

import httpx
from jinja2 import Template

from .evaluator import evaluate_llm_judge, exact_match, weighted_match
from .openrouter import openrouter_request


def render_template(prompt, variables, test_case):
    if variables:
        render_kwargs = dict(variables)
    else:
        render_kwargs = {}
    if test_case["variables"]:
        render_kwargs.update(test_case["variables"])
    rendered_prompt = Template(prompt["template"]).render(**render_kwargs)
    rendered_system = (
        Template(prompt["system"]).render(**render_kwargs)
        if prompt.get("system")
        else None
    )
    return rendered_prompt, rendered_system


def run(config, verbose=False):
    """Run every model × prompt × test case and return ``(results, timestamp)``.

    ``verbose`` enables per-test-case detail (rendered prompts, raw responses);
    by default only high-level progress and the final report are printed.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    results = []

    with httpx.Client() as client:
        for model in config["models"]:
            print(f"Model: {model}\n")
            for prompt in config["prompts"]:
                success = 0
                total = 0
                test_case_results = []
                total_prompt_tokens = 0
                total_completion_tokens = 0
                total_cost = 0
                print(f"Prompt: {prompt['id']}")
                for test_case in config["test_cases"]:
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
                    )
                    # Rate-limit only real API calls — cache hits need no backoff.
                    if not response.get("from_cache"):
                        time.sleep(5)
                    if verbose:
                        print(f"Raw response: {response['response']}")
                    match_result = False
                    match_score = None
                    reasoning = None
                    judge_prompt_tokens = 0
                    judge_completion_tokens = 0
                    judge_cost = 0.0
                    if test_case["evaluation"] == "exact_match":
                        match_result = exact_match(
                            response["response"], test_case["expected_output"]
                        )
                        match_score = 1.0 if match_result else 0.0
                    elif test_case["evaluation"] == "weighted_match":
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
                    elif test_case["evaluation"] == "llm_judge":
                        judge_result = evaluate_llm_judge(
                            client,
                            config["evaluation"],
                            response["response"],
                            rendered,  # full rendered prompt — no variable-name coupling
                            test_case["criteria"],
                            test_case.get("expected_output"),  # reference (None here)
                        )
                        if not judge_result.get("from_cache"):
                            time.sleep(5)
                        match_result = judge_result["match_result"]
                        match_score = judge_result["score_normalized"]
                        reasoning = judge_result["reasoning"]
                        judge_prompt_tokens = judge_result["judge_prompt_tokens"]
                        judge_completion_tokens = judge_result[
                            "judge_completion_tokens"
                        ]
                        judge_cost = judge_result["judge_cost"]
                        if verbose:
                            print(
                                f"Judge status: {judge_result['status']} "
                                f"raw: {judge_result['score_raw']} "
                                f"score: {match_score} match result: {match_result}"
                            )

                    if verbose:
                        print(
                            f"Response: {response['response']} Prompt tokens: {response['prompt_tokens']} Completion tokens: {response['completion_tokens']} Cost: ${response['cost']:.6f}"
                        )
                    prompt_tokens = response["prompt_tokens"]
                    completion_tokens = response["completion_tokens"]
                    cost = response["cost"]
                    test_case_results.append(
                        {
                            "id": test_case["id"],
                            "tags": test_case["tags"],
                            "evaluation": test_case["evaluation"],
                            "match_result": match_result,
                            "match_score": match_score,
                            "response": response["response"],
                            "reasoning": reasoning,
                            "expected": test_case.get("expected_output"),
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "cost": cost,
                            "judge_prompt_tokens": judge_prompt_tokens,
                            "judge_completion_tokens": judge_completion_tokens,
                            "judge_cost": judge_cost,
                            "total_cost": cost + judge_cost,
                        }
                    )

                    total_prompt_tokens += prompt_tokens
                    total_completion_tokens += completion_tokens
                    total_cost += cost + judge_cost

                    if verbose:
                        print(f"Expected: {test_case.get('expected_output')}")
                        print(f"Match result: {match_result}\n")
                    if match_result:
                        success += 1
                    total += 1

                score = success / total * 100
                results.append(
                    {
                        "model": model,
                        "prompt_id": prompt["id"],
                        "success": success,
                        "total": total,
                        "score": score,
                        "total_prompt_tokens": total_prompt_tokens,
                        "total_completion_tokens": total_completion_tokens,
                        "total_cost": total_cost,
                        "test_cases": test_case_results,
                    }
                )
                print(f"Quality score: {success}/{total} ({score:.1f}%)")
                print(f"Total prompt tokens: {total_prompt_tokens}")
                print(f"Total completion tokens: {total_completion_tokens}")
                print(f"Total cost: ${total_cost:.6f}")

    return results, timestamp
