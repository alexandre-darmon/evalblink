import datetime
import os
import time
import yaml
from dotenv import load_dotenv
import httpx
import json
from jinja2 import Template

from evalblink.openrouter import openrouter_request


def load_config(filepath):
    with open(filepath) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        return config


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


def exact_match(response, expected):
    return response.strip().lower() == expected.strip().lower()


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
        }

    raw = judge_response["response"]
    judge_prompt_tokens = judge_response["prompt_tokens"]
    judge_completion_tokens = judge_response["completion_tokens"]
    judge_cost = judge_response["cost"]
    try:
        parsed = json.loads(raw)
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
    }


def weighted_match(evaluation_params, response, expected, tolerance=0.20):
    param_map = {}
    for eval_param in evaluation_params["variables"]:
        name = eval_param["name"]
        param_map[name] = eval_param

    parsed = json.loads(response)

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


def save_results(config, results, timestamp):
    benchmark_name = config["name"]
    run_id = f"{timestamp}_{benchmark_name}"
    data = {
        "run_id": run_id,
        "benchmark": config["name"],
        "judge_model": config.get("evaluation", {}).get("judge_model"),
        "temperature": config["inference"]["temperature"]
        if "temperature" in config["inference"]
        else 0,
        "max_tokens": config["inference"]["max_tokens"]
        if "max_tokens" in config["inference"]
        else 4096,
        "quality_threshold": config["evaluation"].get(
            "quality_threshold", config["evaluation"].get("judge_threshold")
        ),
        "timestamp": timestamp,
        "results": results,
    }
    os.makedirs("results", exist_ok=True)
    file_path = f"results/{run_id}.json"
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    return file_path


if __name__ == "__main__":
    load_dotenv()
    client = httpx.Client()
    filepath = "benchmarks/llm_as_judge.yaml"
    config = load_config(filepath)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    results = []
    test_case_results = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0

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
                print(f"Rendered prompt: {rendered}")
                response = openrouter_request(
                    client,
                    rendered,
                    model,
                    config["inference"]["temperature"],
                    config["inference"]["max_tokens"],
                    rendered_system,
                )
                if not response.get("from_cache"):
                    time.sleep(5)
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
                    time.sleep(5)
                    match_result = judge_result["match_result"]
                    match_score = judge_result["score_normalized"]
                    reasoning = judge_result["reasoning"]
                    judge_prompt_tokens = judge_result["judge_prompt_tokens"]
                    judge_completion_tokens = judge_result["judge_completion_tokens"]
                    judge_cost = judge_result["judge_cost"]
                    print(
                        f"Judge status: {judge_result['status']} "
                        f"raw: {judge_result['score_raw']} "
                        f"score: {match_score} match result: {match_result}"
                    )

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
    file = save_results(config, results, timestamp)
    print(f"Results saved to {file}")
