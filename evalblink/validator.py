"""Static validation for benchmark YAML configs — no API calls."""

from __future__ import annotations

import json
import re

import jinja2
import jinja2.meta

from .openrouter import is_number as _is_number
from .runner import JINJA_ENV, render_template
from .schemas import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE

KNOWN_EVAL_TYPES = {"exact_match", "weighted_match", "llm_judge"}
# Hardcoded in ``evaluator.weighted_match`` — every one must be weighted.
WEIGHTED_DIMENSIONS = ("use_case", "percent", "order")

# httpx's default pool holds 100 connections; more workers would only queue
# (and time out) on it.
MAX_CONCURRENCY = 100

_SINGLE_BRACE_RE = re.compile(r"(?<!\{)\{(\w+)\}(?!\})")


def _template_vars(text: str) -> set[str]:
    """Return variable names used in a Jinja2 template, including filtered ones.

    Raises ``jinja2.TemplateSyntaxError`` on a malformed template.
    """
    return jinja2.meta.find_undeclared_variables(JINJA_ENV.parse(text))


def _hashable(value) -> bool:
    """weighted_match keys labels by ``use_case``; a YAML list/map/set can't be one."""
    try:
        hash(value)
    except TypeError:
        return False
    return True


def _mapping(value, label: str, errors: list[str]) -> dict:
    """``value`` as a dict: ``None`` reads as absent, any other non-mapping errors."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        errors.append(
            f"'{label}' must be a mapping of key: value pairs"
            f" (got {type(value).__name__})"
        )
        return {}
    return value


def validate(config: dict) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Errors block the run; warnings are advisory."""
    errors: list[str] = []
    warnings: list[str] = []

    # An empty YAML file loads as None; a bare list or scalar is not a config.
    if config is None:
        return ["Config file is empty"], []
    if not isinstance(config, dict):
        kind = type(config).__name__
        return [
            f"Config must be a YAML mapping of name/models/prompts/… (got {kind})"
        ], []
    # The config is echoed into the results JSON (expected outputs, ids, ...);
    # a self-referencing YAML alias (`&a [*a]`) would only fail there, after
    # every API call has been paid for.
    try:
        json.dumps(config, default=str)
    except (ValueError, RecursionError):
        return ["Config contains a self-referencing YAML alias (e.g. `&a [*a]`)"], []

    # Required top-level fields — check presence and type
    for field in ("name", "models", "prompts", "test_cases"):
        val = config.get(field)
        if not val:
            errors.append(f"Missing required field: '{field}'")
        elif field == "name" and not isinstance(val, str):
            errors.append(f"'name' must be a string (got {val!r}) — quote it")
        elif field in ("models", "prompts", "test_cases") and not isinstance(val, list):
            errors.append(f"'{field}' must be a list (got {type(val).__name__})")

    models = config.get("models")
    if isinstance(models, list):
        seen_models: set[str] = set()
        for i, model in enumerate(models):
            if not isinstance(model, str) or not model:
                errors.append(f"models[{i}] must be a model ID string (got {model!r})")
            elif model in seen_models:
                errors.append(f"Duplicate model: '{model}'")
            else:
                seen_models.add(model)

    # Top-level scalars
    threshold = config.get("quality_threshold")
    if threshold is not None and not (_is_number(threshold) and 0 <= threshold <= 100):
        errors.append(
            "quality_threshold (the CI gate) must be a number between 0 and 100"
            f" (got {threshold!r})"
        )
    concurrency = config.get("concurrency")
    if concurrency is not None and not (
        _is_number(concurrency)
        and isinstance(concurrency, int)
        and not isinstance(concurrency, bool)
        and 1 <= concurrency <= MAX_CONCURRENCY
    ):
        errors.append(
            f"concurrency must be a whole number from 1 to {MAX_CONCURRENCY}"
            f" (got {concurrency!r})"
        )
    max_cost_usd = config.get("max_cost_usd")
    if max_cost_usd is not None and not (
        _is_number(max_cost_usd) and max_cost_usd >= 0
    ):
        errors.append(
            f"max_cost_usd must be a non-negative number (got {max_cost_usd!r})"
        )

    # Inference
    inference = _mapping(config.get("inference"), "inference", errors)
    max_tokens = inference.get("max_tokens", DEFAULT_MAX_TOKENS)
    # An int: 100.0, 50.5, true, .inf and float-overflowing ints are rejected.
    if not (_is_number(max_tokens) and isinstance(max_tokens, int) and max_tokens > 0):
        errors.append(
            "inference.max_tokens must be a positive whole number like 100"
            f" (got {max_tokens!r})"
        )
        max_tokens = DEFAULT_MAX_TOKENS
    temperature = inference.get("temperature", DEFAULT_TEMPERATURE)
    if not _is_number(temperature):
        errors.append(f"inference.temperature must be a number (got {temperature!r})")

    # Prompts
    prompts = config.get("prompts") or []
    prompt_ids: set[str] = set()
    required_template_vars: set[str] = set()

    for i, prompt in enumerate(prompts if isinstance(prompts, list) else []):
        if not isinstance(prompt, dict):
            errors.append(f"prompts[{i}] must be a mapping with 'id' and 'template'")
            continue
        pid = prompt.get("id")
        template = prompt.get("template") or ""
        system = prompt.get("system") or ""

        if pid is None or pid == "":
            errors.append(f"prompts[{i}]: missing required field 'id'")
        elif not isinstance(pid, str):
            errors.append(
                f"prompts[{i}]: 'id' must be a string (got {pid!r}) — quote it"
            )
        elif pid in prompt_ids:
            errors.append(f"Duplicate prompt id: '{pid}'")
        else:
            prompt_ids.add(pid)

        label = f"'{pid}'" if isinstance(pid, str) and pid else str(i)

        if not template:
            errors.append(f"prompts[{label}]: missing required field 'template'")
        elif not isinstance(template, str):
            errors.append(f"prompts[{label}]: 'template' must be a string")
            template = ""
        if not isinstance(system, str):
            errors.append(f"prompts[{label}]: 'system' must be a string")
            system = ""

        for m in _SINGLE_BRACE_RE.finditer(template + " " + system):
            warnings.append(
                f"prompts[{label}]: template contains single-brace '{{{m.group(1)}}}'"
                f" — did you mean '{{{{ {m.group(1)} }}}}'?"
            )

        for key, text in (("template", template), ("system", system)):
            if not text:
                continue
            try:
                required_template_vars.update(_template_vars(text))
            except jinja2.TemplateSyntaxError as exc:
                errors.append(
                    f"prompts[{label}].{key}: Jinja2 syntax error on line"
                    f" {exc.lineno}: {exc.message}"
                )

    # Evaluation config
    evaluation = _mapping(config.get("evaluation"), "evaluation", errors)

    # Test cases
    test_cases = config.get("test_cases") or []
    global_vars = set(_mapping(config.get("variables"), "variables", errors))
    tc_ids: set[str] = set()
    has_llm_judge = False
    has_weighted = False

    for i, tc in enumerate(test_cases if isinstance(test_cases, list) else []):
        if not isinstance(tc, dict):
            errors.append(
                f"test_cases[{i}] must be a mapping with 'id' and 'evaluation'"
            )
            continue
        tc_id = tc.get("id")
        label = f"'{tc_id}'" if isinstance(tc_id, str) and tc_id else str(i)

        if tc_id is None or tc_id == "":
            errors.append(f"test_cases[{i}]: missing required field 'id'")
        elif not isinstance(tc_id, str):
            errors.append(
                f"test_cases[{i}]: 'id' must be a string (got {tc_id!r}) — quote it"
            )
        elif tc_id in tc_ids:
            errors.append(f"Duplicate test_case id: '{tc_id}'")
        else:
            tc_ids.add(tc_id)

        eval_type = tc.get("evaluation")
        if eval_type is None:
            errors.append(f"test_cases[{label}]: missing required field 'evaluation'")
        elif not isinstance(eval_type, str) or eval_type not in KNOWN_EVAL_TYPES:
            errors.append(
                f"test_cases[{label}]: unknown evaluation type '{eval_type}'"
                f" (must be one of: {', '.join(sorted(KNOWN_EVAL_TYPES))})"
            )

        tags = tc.get("tags")
        if tags is not None and not (
            isinstance(tags, list) and all(isinstance(t, str) for t in tags)
        ):
            errors.append(f"test_cases[{label}]: 'tags' must be a list of strings")

        if eval_type == "exact_match":
            expected = tc.get("expected_output")
            if expected is None:
                errors.append(
                    f"test_cases[{label}]: exact_match requires 'expected_output'"
                )
            elif not isinstance(expected, str):
                errors.append(
                    f"test_cases[{label}]: exact_match expected_output must be a"
                    f" string (got {expected!r}) — quote it in the YAML; unquoted"
                    " yes/no/true/false and numbers are not parsed as text"
                )
            elif len(expected) > max_tokens * 4:
                warnings.append(
                    f"test_cases[{label}]: expected_output is long"
                    f" ({len(expected)} chars) relative to"
                    f" inference.max_tokens ({max_tokens})"
                )

        if eval_type == "llm_judge":
            has_llm_judge = True
            if not tc.get("criteria"):
                errors.append(f"test_cases[{label}]: llm_judge requires 'criteria'")
            elif not isinstance(tc["criteria"], str):
                errors.append(f"test_cases[{label}]: 'criteria' must be a string")

        if eval_type == "weighted_match":
            has_weighted = True
            expected = tc.get("expected_output")
            if expected is None:
                errors.append(
                    f"test_cases[{label}]: weighted_match requires 'expected_output'"
                )
            elif not isinstance(expected, list) or not expected:
                errors.append(
                    f"test_cases[{label}]: weighted_match expected_output must be a"
                    " non-empty list of {use_case, percent, order} items"
                )
            elif not all(
                isinstance(item, dict)
                and set(WEIGHTED_DIMENSIONS) <= item.keys()
                and _hashable(item["use_case"])
                and _is_number(item["percent"])
                for item in expected
            ):
                errors.append(
                    f"test_cases[{label}]: every weighted_match expected_output item"
                    " needs use_case, percent (a number) and order"
                )

        # Template variable coverage
        tc_vars = set(
            _mapping(tc.get("variables"), f"test_cases[{label}].variables", errors)
        )
        available = global_vars | tc_vars
        missing = required_template_vars - available
        if missing:
            errors.append(
                f"test_cases[{label}]: template variable(s) not provided:"
                f" {', '.join(sorted(missing))}"
            )

    # Cross-cutting evaluation config checks. judge_model is type-checked even
    # without llm_judge cases: the runner reads it for the vendor warning.
    judge_model = evaluation.get("judge_model")
    if judge_model is not None and not isinstance(judge_model, str):
        errors.append(
            f"evaluation.judge_model must be a model ID string (got {judge_model!r})"
        )
    elif has_llm_judge and not judge_model:
        errors.append(
            "One or more test_cases use 'llm_judge' but 'evaluation.judge_model' is not set"
        )
    if has_llm_judge:
        judge_threshold = evaluation.get("judge_threshold")
        if judge_threshold is not None and not (
            _is_number(judge_threshold) and 0 <= judge_threshold <= 1
        ):
            errors.append(
                "evaluation.judge_threshold must be a number between 0 and 1"
                f" (got {judge_threshold!r})"
            )

    if has_weighted:
        case_threshold = evaluation.get("quality_threshold")
        if case_threshold is None:
            errors.append(
                "One or more test_cases use 'weighted_match' but"
                " 'evaluation.quality_threshold' is not set"
            )
        elif not (_is_number(case_threshold) and 0 <= case_threshold <= 1):
            errors.append(
                "evaluation.quality_threshold must be a number between 0 and 1"
                f" (got {case_threshold!r})"
            )
        variables = evaluation.get("variables")
        if variables is None:
            errors.append(
                "One or more test_cases use 'weighted_match' but"
                " 'evaluation.variables' is not set"
            )
        elif not isinstance(variables, list) or not all(
            isinstance(v, dict) and isinstance(v.get("name"), str) for v in variables
        ):
            errors.append(
                "evaluation.variables must be a list of {name, weight, tolerance?} items"
            )
        else:
            by_name = {v["name"]: v for v in variables}
            absent = [d for d in WEIGHTED_DIMENSIONS if d not in by_name]
            if absent:
                errors.append(
                    "evaluation.variables is missing weighted_match dimension(s):"
                    f" {', '.join(absent)} — use_case, percent and order all need"
                    " an entry with a weight"
                )
            for dim in WEIGHTED_DIMENSIONS:
                v = by_name.get(dim)
                if v is not None and not (
                    _is_number(v.get("weight")) and _is_number(v.get("tolerance", 0))
                ):
                    errors.append(
                        f"evaluation.variables '{dim}': weight (and tolerance, if"
                        " set) must be numbers"
                    )

    # Heuristic warnings
    if has_llm_judge and max_tokens < 200:
        warnings.append(
            f"inference.max_tokens: {max_tokens} may be too low for llm_judge"
            " — the candidate's open-ended answer may be cut off before it is"
            " judged (recommend ≥ 200)"
        )

    # Trial-render every prompt × test case exactly as the runner will (same
    # function, same sandbox), so a template that would raise mid-run — nested
    # attribute on an undefined variable, a sandbox violation, a bad filter
    # argument — fails here, before any API call. Only on an otherwise-valid
    # config: the render relies on the structure checked above.
    if not errors:
        for prompt in prompts:
            for tc in test_cases:
                try:
                    render_template(prompt, config.get("variables"), tc)
                except Exception as exc:  # noqa: BLE001 — any render failure
                    errors.append(
                        f"prompts['{prompt['id']}'] × test_cases['{tc['id']}']:"
                        f" template failed to render — {type(exc).__name__}: {exc}"
                    )

    return errors, warnings
