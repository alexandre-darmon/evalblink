"""Static validation for benchmark YAML configs — no API calls."""

from __future__ import annotations

import re

import jinja2
import jinja2.meta

KNOWN_EVAL_TYPES = {"exact_match", "weighted_match", "llm_judge"}

_SINGLE_BRACE_RE = re.compile(r"(?<!\{)\{(\w+)\}(?!\})")
_JINJA_ENV = jinja2.Environment()


def _template_vars(text: str) -> set[str]:
    """Return variable names used in a Jinja2 template, including filtered ones."""
    try:
        return jinja2.meta.find_undeclared_variables(_JINJA_ENV.parse(text))
    except jinja2.TemplateSyntaxError:
        return set()


def validate(config: dict) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Errors block the run; warnings are advisory."""
    errors: list[str] = []
    warnings: list[str] = []

    # Required top-level fields — check presence and type
    for field in ("name", "models", "prompts", "test_cases"):
        val = config.get(field)
        if not val:
            errors.append(f"Missing required field: '{field}'")
        elif field in ("models", "prompts", "test_cases") and not isinstance(val, list):
            errors.append(f"'{field}' must be a list (got {type(val).__name__})")

    # Inference
    inference = config.get("inference") or {}
    max_tokens = inference.get("max_tokens", 100)
    if (
        not isinstance(max_tokens, (int, float))
        or isinstance(max_tokens, bool)
        or max_tokens <= 0
    ):
        errors.append(f"inference.max_tokens must be a positive number (got {max_tokens!r})")
        max_tokens = 100

    # Prompts
    prompts = config.get("prompts") or []
    prompt_ids: set[str] = set()
    required_template_vars: set[str] = set()

    for i, prompt in enumerate(prompts if isinstance(prompts, list) else []):
        pid = prompt.get("id")
        template = prompt.get("template") or ""
        system = prompt.get("system") or ""

        if not pid:
            errors.append(f"prompts[{i}]: missing required field 'id'")
        elif pid in prompt_ids:
            errors.append(f"Duplicate prompt id: '{pid}'")
        else:
            prompt_ids.add(pid)

        label = f"'{pid}'" if pid else str(i)

        if not template:
            errors.append(f"prompts[{label}]: missing required field 'template'")

        for m in _SINGLE_BRACE_RE.finditer(template + " " + system):
            warnings.append(
                f"prompts[{label}]: template contains single-brace '{{{m.group(1)}}}'"
                f" — did you mean '{{{{ {m.group(1)} }}}}'?"
            )

        required_template_vars.update(_template_vars(template))
        if system:
            required_template_vars.update(_template_vars(system))

    # Evaluation config
    evaluation = config.get("evaluation") or {}

    # Test cases
    test_cases = config.get("test_cases") or []
    global_vars = set((config.get("variables") or {}).keys())
    tc_ids: set[str] = set()
    has_llm_judge = False
    has_weighted = False

    for i, tc in enumerate(test_cases if isinstance(test_cases, list) else []):
        tc_id = tc.get("id")
        label = f"'{tc_id}'" if tc_id else str(i)

        if not tc_id:
            errors.append(f"test_cases[{i}]: missing required field 'id'")
        elif tc_id in tc_ids:
            errors.append(f"Duplicate test_case id: '{tc_id}'")
        else:
            tc_ids.add(tc_id)

        eval_type = tc.get("evaluation")
        if eval_type is None:
            errors.append(f"test_cases[{label}]: missing required field 'evaluation'")
        elif eval_type not in KNOWN_EVAL_TYPES:
            errors.append(
                f"test_cases[{label}]: unknown evaluation type '{eval_type}'"
                f" (must be one of: {', '.join(sorted(KNOWN_EVAL_TYPES))})"
            )

        if eval_type == "exact_match":
            if tc.get("expected_output") is None:
                errors.append(f"test_cases[{label}]: exact_match requires 'expected_output'")
            elif isinstance(tc["expected_output"], str):
                if len(tc["expected_output"]) > max_tokens * 4:
                    warnings.append(
                        f"test_cases[{label}]: expected_output is long"
                        f" ({len(tc['expected_output'])} chars) relative to"
                        f" inference.max_tokens ({max_tokens})"
                    )

        if eval_type == "llm_judge":
            has_llm_judge = True
            if not tc.get("criteria"):
                errors.append(f"test_cases[{label}]: llm_judge requires 'criteria'")

        if eval_type == "weighted_match":
            has_weighted = True
            if tc.get("expected_output") is None:
                errors.append(f"test_cases[{label}]: weighted_match requires 'expected_output'")

        # Template variable coverage
        tc_vars = set((tc.get("variables") or {}).keys())
        available = global_vars | tc_vars
        missing = required_template_vars - available
        if missing:
            errors.append(
                f"test_cases[{label}]: template variable(s) not provided:"
                f" {', '.join(sorted(missing))}"
            )

    # Cross-cutting evaluation config checks
    if has_llm_judge and not evaluation.get("judge_model"):
        errors.append(
            "One or more test_cases use 'llm_judge' but 'evaluation.judge_model' is not set"
        )

    if has_weighted:
        if evaluation.get("quality_threshold") is None:
            errors.append(
                "One or more test_cases use 'weighted_match' but"
                " 'evaluation.quality_threshold' is not set"
            )
        if evaluation.get("variables") is None:
            errors.append(
                "One or more test_cases use 'weighted_match' but"
                " 'evaluation.variables' is not set"
            )

    # Heuristic warnings
    if has_llm_judge and isinstance(max_tokens, (int, float)) and max_tokens < 200:
        warnings.append(
            f"inference.max_tokens: {max_tokens} may be too low for llm_judge"
            " — the judge response needs JSON output (recommend ≥ 200)"
        )

    return errors, warnings
