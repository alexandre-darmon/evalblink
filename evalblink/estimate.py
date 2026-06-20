"""Offline cost estimation for ``evalblink run --dry-run``.

Pure logic: given a benchmark config and a pricing catalog (from
``openrouter.fetch_models``), estimate what a full run would cost *without*
calling the completions API. Like ``analysis``/``compare``, this module only
decides — ``reporter.render_estimate`` formats the result.

Estimates are deliberately rough and honest:
- prompt tokens ≈ ``len(text) / 4`` (the ~4-chars-per-token rule of thumb),
- completion tokens = the configured ``max_tokens`` (a ceiling — real outputs
  are usually shorter).
"""

from __future__ import annotations

import math

from .runner import render_template

# Mirrors the fixed judge budget in ``evaluator.evaluate_llm_judge``.
JUDGE_MAX_TOKENS = 512


def _est_tokens(text) -> int:
    """Rough token count for ``text`` using the ~4-chars-per-token heuristic."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def _price(models_meta, model, missing):
    """Pricing for ``model``; records a miss (and returns zeros) when unknown."""
    meta = models_meta.get(model)
    if meta is None:
        if model not in missing:
            missing.append(model)
        return 0.0, 0.0
    return meta["prompt"], meta["completion"]


def estimate(config, models_meta) -> dict:
    """Estimate the cost of running ``config`` against ``models_meta`` pricing.

    Returns a dict of per-(model, prompt_id) rows plus grand totals, the set of
    models with no pricing, and the ``max_cost_usd`` budget verdict.
    """
    inference = config.get("inference", {})
    max_tokens = inference.get("max_tokens", 100)
    judge_model = config.get("evaluation", {}).get("judge_model")
    missing: list = []

    rows = []
    total_cost = 0.0
    total_judge_cost = 0.0

    for model in config["models"]:
        cand_prompt_price, cand_completion_price = _price(models_meta, model, missing)
        for prompt in config["prompts"]:
            est_prompt_tokens = 0
            est_completion_tokens = 0
            est_cost = 0.0
            est_judge_cost = 0.0
            for test_case in config["test_cases"]:
                rendered, rendered_system = render_template(
                    prompt, config.get("variables"), test_case
                )
                prompt_tok = _est_tokens((rendered_system or "") + rendered)
                est_prompt_tokens += prompt_tok
                est_completion_tokens += max_tokens
                est_cost += (
                    prompt_tok * cand_prompt_price + max_tokens * cand_completion_price
                )
                if test_case.get("evaluation") == "llm_judge":
                    judge_prompt_price, judge_completion_price = _price(
                        models_meta, judge_model, missing
                    )
                    # The judge sees the rendered prompt, the criteria, and the
                    # candidate's response (unknown here — assume max_tokens worth).
                    judge_prompt_tok = (
                        prompt_tok
                        + _est_tokens(test_case.get("criteria", ""))
                        + max_tokens
                    )
                    est_judge_cost += (
                        judge_prompt_tok * judge_prompt_price
                        + JUDGE_MAX_TOKENS * judge_completion_price
                    )
            rows.append(
                {
                    "model": model,
                    "prompt_id": prompt["id"],
                    "est_prompt_tokens": est_prompt_tokens,
                    "est_completion_tokens": est_completion_tokens,
                    "est_cost": est_cost,
                    "est_judge_cost": est_judge_cost,
                }
            )
            total_cost += est_cost + est_judge_cost
            total_judge_cost += est_judge_cost

    max_cost_usd = config.get("max_cost_usd")
    over_budget = max_cost_usd is not None and total_cost > max_cost_usd
    return {
        "rows": rows,
        "total_cost": total_cost,
        "judge_cost": total_judge_cost,
        "missing_pricing": missing,
        "max_cost_usd": max_cost_usd,
        "over_budget": over_budget,
    }
