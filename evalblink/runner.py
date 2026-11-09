"""Executes a benchmark config and returns raw results.

One job: own the ``httpx.Client`` and the model × prompt × test-case (× repeat)
loop, call OpenRouter for each candidate, route the response to the right
evaluator, and accumulate per-prompt results: token/cost totals, latency
percentiles and the score's interval (computed by ``analysis``). Its only output is
diagnostics on stderr (one progress line per finished case, warnings); the
results table is the reporter's job.
"""

from __future__ import annotations

import datetime
import functools
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import httpx
from jinja2.sandbox import SandboxedEnvironment

from . import analysis
from .evaluator import (
    TEXT_CHECKS,
    ScorerError,
    evaluate_llm_judge,
    json_match,
    json_schema,
    judge_vendor_warning,
    load_scorer,
    python_score,
    weighted_match,
)
from .openrouter import CANCELLED, openrouter_request
from .schemas import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, ERROR_LABELS

# Templates come from user-supplied YAML: parse (validator) and render them in a
# sandbox so a template can't reach Python internals via attribute access.
JINJA_ENV = SandboxedEnvironment()


@functools.lru_cache(maxsize=512)
def _compile(source: str):
    """One compile per distinct template (validate renders every prompt × case)."""
    return JINJA_ENV.from_string(source)


def render_template(prompt, variables, test_case):
    if variables:
        render_kwargs = dict(variables)
    else:
        render_kwargs = {}
    if test_case.get("variables"):
        render_kwargs.update(test_case["variables"])
    rendered_prompt = _compile(prompt["template"]).render(**render_kwargs)
    rendered_system = (
        _compile(prompt["system"]).render(**render_kwargs)
        if prompt.get("system")
        else None
    )
    return rendered_prompt, rendered_system


# Modes whose score is graded (0–1), not just 1/0: the progress line shows it.
GRADED = {"llm_judge", "weighted_match", "json_match", "python"}


def _run_case(client, model, prompt, config, test_case, verbose, use_cache, sample):
    """``_evaluate_case``, but no single case can crash the run.

    Anything unexpected (a template that only fails on some renders, a value the
    scorer can't handle) is recorded as an ``internal_error`` pipeline-error case
    instead of discarding every other, already-paid-for result.
    """
    try:
        return _evaluate_case(
            client, model, prompt, config, test_case, verbose, use_cache, sample
        )
    except Exception as exc:  # noqa: BLE001 — the point: one case never kills a run
        error = f"{type(exc).__name__}: {exc}"
        return _sampled(
            sample,
            {
                "model": model,
                "prompt_id": prompt["id"],
                "test_case": {
                    "id": test_case["id"],
                    "tags": test_case.get("tags") or [],
                    "evaluation": test_case.get("evaluation"),
                    "status": "internal_error",
                    "match_result": False,
                    "match_score": None,
                    "response": "",
                    "reasoning": None,
                    "expected": test_case.get("expected_output"),
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cost": 0.0,
                    "judge_prompt_tokens": 0,
                    "judge_completion_tokens": 0,
                    "judge_cost": 0.0,
                    "total_cost": 0.0,
                    "latency_s": None,
                    "error": error,
                },
            },
        )


def _sampled(sample, result):
    """Tag a case result with its repeat index (``None``: a run without repeats)."""
    if sample is not None:
        result["test_case"]["sample"] = sample
    return result


def _evaluate_case(
    client, model, prompt, config, test_case, verbose, use_cache, sample=None
):
    """Run one (model × prompt × test_case) triple and return a tagged result.

    Returned dict has ``"model"``, ``"prompt_id"``, and ``"test_case"`` keys.
    The caller collects these and re-groups them after all futures complete.
    """
    if verbose:
        print(f"Test case: {model} / {prompt['id']} / {test_case['id']}")
    rendered, rendered_system = render_template(
        prompt, config.get("variables"), test_case
    )
    if verbose:
        print(f"Rendered prompt: {rendered}")
    inference = config.get("inference") or {}
    eval_config = config.get("evaluation") or {}
    error = None
    try:
        response = openrouter_request(
            client,
            rendered,
            model,
            inference.get("temperature", DEFAULT_TEMPERATURE),
            inference.get("max_tokens", DEFAULT_MAX_TOKENS),
            rendered_system,
            use_cache=use_cache,
            sample=sample or 0,
        )
    except (RuntimeError, httpx.HTTPError) as exc:
        # Candidate API failed after retries — a pipeline failure, not a real
        # score of 0. Record it and keep the run going (mirrors the judge).
        error = str(exc) or type(exc).__name__
        response = {
            "response": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cost": 0.0,
            "latency_s": None,
        }
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

    if error is not None:
        status = "api_error"
    elif evaluation in TEXT_CHECKS:
        check = TEXT_CHECKS[evaluation]
        match_result = check(response["response"], test_case["expected_output"])
        match_score = 1.0 if match_result else 0.0
    elif evaluation == "json_match":
        match_score = json_match(response["response"], test_case["expected_output"])
        match_result = match_score >= 1.0
    elif evaluation == "json_schema":
        match_result, reasoning = json_schema(
            response["response"], test_case["expected_output"]
        )
        match_score = 1.0 if match_result else 0.0
    elif evaluation == "python":
        try:
            scorer = load_scorer(test_case["scorer"], config.get("_base_dir", ""))
            match_result, match_score, reasoning = python_score(
                scorer, response["response"], test_case
            )
        except ScorerError as exc:
            # The user's scorer failed — a pipeline error, not a 0 for the model.
            status = "scorer_error"
            error = f"{test_case['scorer']}: {exc}"
    elif evaluation == "weighted_match":
        match_score = weighted_match(
            eval_config,
            response["response"],
            test_case["expected_output"],
        )
        threshold = eval_config.get("quality_threshold", 0.7)
        match_result = match_score >= threshold
        if verbose:
            print(
                f"Match score: {match_score:.4f} (threshold: {threshold:.4f}) match result: {match_result}"
            )
    elif evaluation == "llm_judge":
        judge_result = evaluate_llm_judge(
            client,
            eval_config,
            response["response"],
            rendered,  # full rendered prompt — no variable-name coupling
            test_case["criteria"],
            test_case.get("reference"),
            use_cache=use_cache,
        )
        match_result = judge_result["match_result"]
        match_score = judge_result["score_normalized"]
        # Carry the judge's status so a pipeline failure (score=None)
        # is distinguished downstream from a real score of 0.
        status = judge_result["status"]
        reasoning = judge_result["reasoning"]
        judge_prompt_tokens = judge_result["judge_prompt_tokens"]
        judge_completion_tokens = judge_result["judge_completion_tokens"]
        judge_cost = judge_result["judge_cost"]
        if judge_result.get("error"):
            error = judge_result["error"]
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
    result = {
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
            "latency_s": response.get("latency_s"),
        },
    }
    if error is not None:
        result["test_case"]["error"] = error
    return _sampled(sample, result)


def run(config, verbose=False, use_cache=True):
    """Run every model × prompt × test case concurrently and return ``(results, timestamp)``.

    ``concurrency`` (config key, default 5, clamped to >= 1) controls the
    ``ThreadPoolExecutor`` worker count. Transient API failures (429/5xx) are
    handled by the retry/backoff in ``openrouter_request`` rather than a fixed
    per-call sleep; a candidate call that still fails is recorded as an
    ``api_error`` case (``match_score=None``) and the run continues. Each
    finished case prints one progress line to stderr (:func:`_progress_line`);
    ``verbose`` adds per-case detail on stdout.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    # A bare `concurrency:` (None) means the default. Clamp to a sane minimum —
    # ThreadPoolExecutor raises on max_workers < 1.
    concurrency = config.get("concurrency")
    concurrency = max(1, 5 if concurrency is None else concurrency)
    repeats = config.get("repeats") or 1
    # A run without repeats records no `sample`, like records before repeats.
    samples = list(range(repeats)) if repeats > 1 else [None]

    warning = judge_vendor_warning(
        (config.get("evaluation") or {}).get("judge_model"), config["models"]
    )
    if warning:
        _diag(warning)

    # Build the full task list in config order so futures preserve that order.
    tasks = [
        (model, prompt, test_case, sample)
        for model in config["models"]
        for prompt in config["prompts"]
        for test_case in config["test_cases"]
        for sample in samples
    ]
    shape = [
        _plural(len(config[key]), noun)
        for key, noun in (
            ("models", "model"),
            ("prompts", "prompt"),
            ("test_cases", "test case"),
        )
    ]
    if repeats > 1:
        shape.append(f"{repeats} repeats")
    _diag(
        f"Running {_plural(len(tasks), 'case')} ({' × '.join(shape)}), {concurrency} at a time"
    )

    lock = threading.Lock()
    finished = 0

    def report(future):
        # Runs in the worker thread that finished the case.
        nonlocal finished
        if future.cancelled() or future.exception() is not None or CANCELLED.is_set():
            return
        with lock:
            finished += 1
            _diag(_progress_line(finished, len(tasks), future.result()))

    CANCELLED.clear()
    # Not a `with`: its exit would wait for in-flight calls, even after Ctrl-C.
    executor = ThreadPoolExecutor(max_workers=concurrency)
    with httpx.Client() as client:
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
                sample,
            )
            for model, prompt, test_case, sample in tasks
        ]
        for future in futures:
            future.add_done_callback(report)
        try:
            # .result() blocks until each future completes; submission order is preserved.
            case_results = [f.result() for f in futures]
        except KeyboardInterrupt:
            # Ctrl-C: stop retries, drop every queued call and don't wait for
            # in-flight ones (a socket read can block until the timeout);
            # main() exits without joining the workers.
            CANCELLED.set()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        executor.shutdown()

    # Re-group by (model, prompt_id), maintaining config order within each group.
    groups: dict = defaultdict(list)
    for cr in case_results:
        groups[(cr["model"], cr["prompt_id"])].append(cr["test_case"])

    results = []
    for model in config["models"]:
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
            combo = {
                "success": success,
                "scored": scored,
                "total": total,
                "test_cases": test_case_results,
            }
            interval = analysis.score_interval(combo)
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
                    # 95 % interval on `score` (Wilson, over the test cases).
                    "score_ci": [round(v, 1) for v in interval] if interval else None,
                    **analysis.latency_stats(test_case_results),
                    "repeats": repeats,
                    "flaky": analysis.flaky_cases(combo),
                    "test_cases": test_case_results,
                }
            )

    return results, timestamp


def _diag(line):
    """Progress and warnings go to stderr, so stdout carries only the results."""
    print(line, file=sys.stderr, flush=True)


def _plural(n, noun):
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _progress_line(done, total, case_result):
    """``[ 3/12] model / prompt / case  pass (0.85)`` for one finished case."""
    tc = case_result["test_case"]
    if tc["match_score"] is None:
        label = ERROR_LABELS.get(tc["status"], "error")
        outcome = f"⚠ {label}: {tc.get('error', '')}"
    else:
        outcome = "pass" if tc["match_result"] else "fail"
        if tc["evaluation"] in GRADED:  # graded, not 1/0
            outcome += f" ({tc['match_score']:.2f})"
        if tc.get("latency_s") is not None:
            outcome += f"  {tc['latency_s']:.2f}s"
    where = (
        f"{case_result['model']} / {case_result['prompt_id']} /"
        f" {analysis.case_label(tc)}"
    )
    return f"[{done:>{len(str(total))}}/{total}] {where}  {outcome}"
