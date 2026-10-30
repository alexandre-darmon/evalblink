"""Offline cost estimation for ``evalblink run --dry-run``.

Pure logic: given a benchmark config and a pricing catalog (from
``openrouter.fetch_models``), estimate what a full run would cost *without*
calling the completions API. Like ``analysis``/``compare``, this module only
decides — ``reporter.render_estimate`` formats the result.

Estimates are deliberately rough and honest:
- prompt tokens ≈ ``len(text) / 4`` (the ~4-chars-per-token rule of thumb),
- completion tokens = the configured ``max_tokens`` (a ceiling — real outputs
  are usually shorter), capped at the model's own output limit from the catalog;
  a budget above that limit is reported in ``over_output_limit``.
"""

from __future__ import annotations

import difflib
import math

from .runner import render_template
from .schemas import DEFAULT_JUDGE_MAX_TOKENS, DEFAULT_MAX_TOKENS


def _est_tokens(text) -> int:
    """Rough token count for ``text`` using the ~4-chars-per-token heuristic."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def _price(models_meta, model, missing):
    """Pricing for ``model``; records a miss (and returns zeros) when unknown —
    absent from the catalog, or priced ``None`` there (``openrouter`` maps a
    negative / non-numeric catalog price to ``None``)."""
    meta = models_meta.get(model)
    if meta is None or meta.get("prompt") is None or meta.get("completion") is None:
        if model not in missing:
            missing.append(model)
        return 0.0, 0.0
    return meta["prompt"], meta["completion"]


def _capped(models_meta, model, budget, setting, over):
    """``budget`` capped at ``model``'s output limit; records each excess once."""
    cap = (models_meta.get(model) or {}).get("max_output")
    if cap is None or budget <= cap:
        return budget
    entry = {"model": model, "setting": setting, "budget": budget, "limit": cap}
    if entry not in over:
        over.append(entry)
    return cap


def estimate(config, models_meta) -> dict:
    """Estimate the cost of running ``config`` against ``models_meta`` pricing.

    Returns a dict of per-(model, prompt_id) rows plus grand totals, the set of
    models with no pricing, and the ``max_cost_usd`` budget verdict.
    """
    inference = config.get("inference") or {}
    max_tokens = inference.get("max_tokens", DEFAULT_MAX_TOKENS)
    evaluation = config.get("evaluation") or {}
    judge_model = evaluation.get("judge_model")
    judge_max_tokens = evaluation.get("judge_max_tokens")
    if judge_max_tokens is None:
        judge_max_tokens = DEFAULT_JUDGE_MAX_TOKENS
    missing: list = []
    over: list = []

    rows = []
    total_cost = 0.0
    total_judge_cost = 0.0

    for model in config["models"]:
        cand_prompt_price, cand_completion_price = _price(models_meta, model, missing)
        cand_tokens = _capped(
            models_meta, model, max_tokens, "inference.max_tokens", over
        )
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
                est_completion_tokens += cand_tokens
                est_cost += (
                    prompt_tok * cand_prompt_price + cand_tokens * cand_completion_price
                )
                if test_case.get("evaluation") == "llm_judge":
                    judge_prompt_price, judge_completion_price = _price(
                        models_meta, judge_model, missing
                    )
                    # The judge sees the rendered prompt, the criteria, and the
                    # candidate's response (unknown here — assume its full budget).
                    judge_prompt_tok = (
                        prompt_tok
                        + _est_tokens(test_case.get("criteria", ""))
                        + cand_tokens
                    )
                    judge_tokens = _capped(
                        models_meta,
                        judge_model,
                        judge_max_tokens,
                        "evaluation.judge_max_tokens",
                        over,
                    )
                    est_judge_cost += (
                        judge_prompt_tok * judge_prompt_price
                        + judge_tokens * judge_completion_price
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

    # Absent from a non-empty catalog: almost always a typo, and every call to
    # it would fail — unlike a listed model whose price is just unknown.
    unknown = [m for m in missing if models_meta and m not in models_meta]
    unknown_models = [
        {
            "model": m,
            "suggestion": next(
                iter(difflib.get_close_matches(str(m), models_meta, n=1)), None
            ),
        }
        for m in unknown
    ]

    max_cost_usd = config.get("max_cost_usd")
    over_budget = max_cost_usd is not None and total_cost > max_cost_usd
    return {
        "rows": rows,
        "total_cost": total_cost,
        "judge_cost": total_judge_cost,
        "missing_pricing": [m for m in missing if m not in unknown],
        "unknown_models": unknown_models,
        "over_output_limit": over,
        "max_cost_usd": max_cost_usd,
        "over_budget": over_budget,
    }
