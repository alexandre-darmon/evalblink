"""Tests for evalblink.validator — all offline, no API calls."""

from __future__ import annotations

import pytest

from evalblink.validator import validate

INF = float("inf")
NAN = float("nan")
HUGE = 10**400  # an int no float can hold


def _exact_match_config(**overrides) -> dict:
    base = {
        "name": "Test",
        "models": ["openai/gpt-4o"],
        "inference": {"temperature": 0, "max_tokens": 100},
        "prompts": [{"id": "v1", "template": "Classify: {{ text }}"}],
        "variables": {"text": "hello"},
        "test_cases": [
            {"id": "tc_001", "evaluation": "exact_match", "expected_output": "greeting"}
        ],
    }
    base.update(overrides)
    return base


def _llm_judge_config(**overrides) -> dict:
    base = {
        "name": "Test",
        "models": ["openai/gpt-4o"],
        "inference": {"temperature": 0, "max_tokens": 512},
        "evaluation": {"judge_model": "openai/gpt-4o", "judge_threshold": 0.70},
        "prompts": [{"id": "v1", "template": "Summarize: {{ text }}"}],
        "variables": {"text": "hello"},
        "test_cases": [
            {
                "id": "tc_001",
                "evaluation": "llm_judge",
                "criteria": "Is it a good summary?",
            }
        ],
    }
    base.update(overrides)
    return base


# --- valid configs ---


def test_valid_exact_match_returns_no_issues():
    errors, warnings = validate(_exact_match_config())
    assert errors == []
    assert warnings == []


def test_valid_llm_judge_returns_no_issues():
    errors, warnings = validate(_llm_judge_config())
    assert errors == []
    assert warnings == []


# --- required field errors ---


def test_missing_name_is_an_error():
    cfg = _exact_match_config()
    del cfg["name"]
    errors, _ = validate(cfg)
    assert any("'name'" in e for e in errors)


def test_missing_models_is_an_error():
    cfg = _exact_match_config()
    del cfg["models"]
    errors, _ = validate(cfg)
    assert any("'models'" in e for e in errors)


def test_missing_prompts_is_an_error():
    cfg = _exact_match_config()
    del cfg["prompts"]
    errors, _ = validate(cfg)
    assert any("'prompts'" in e for e in errors)


def test_missing_test_cases_is_an_error():
    cfg = _exact_match_config()
    del cfg["test_cases"]
    errors, _ = validate(cfg)
    assert any("'test_cases'" in e for e in errors)


# --- evaluation-mode-specific errors ---


def test_exact_match_without_expected_output_is_an_error():
    cfg = _exact_match_config()
    del cfg["test_cases"][0]["expected_output"]
    errors, _ = validate(cfg)
    assert any("expected_output" in e for e in errors)


def test_llm_judge_without_criteria_is_an_error():
    cfg = _llm_judge_config()
    del cfg["test_cases"][0]["criteria"]
    errors, _ = validate(cfg)
    assert any("criteria" in e for e in errors)


def test_llm_judge_without_judge_model_is_an_error():
    cfg = _llm_judge_config()
    cfg["evaluation"] = {}
    errors, _ = validate(cfg)
    assert any("judge_model" in e for e in errors)


def test_unknown_evaluation_type_is_an_error():
    cfg = _exact_match_config()
    cfg["test_cases"][0]["evaluation"] = "magic_scorer"
    errors, _ = validate(cfg)
    assert any("magic_scorer" in e for e in errors)


def test_weighted_match_without_evaluation_variables_is_an_error():
    cfg = {
        "name": "Test",
        "models": ["openai/gpt-4o"],
        "inference": {"temperature": 0, "max_tokens": 100},
        "evaluation": {"quality_threshold": 0.5},
        "prompts": [{"id": "v1", "template": "Hi"}],
        "test_cases": [
            {"id": "tc_001", "evaluation": "weighted_match", "expected_output": []}
        ],
    }
    errors, _ = validate(cfg)
    assert any("evaluation.variables" in e for e in errors)


# --- structural errors ---


def test_duplicate_test_case_ids_is_an_error():
    cfg = _exact_match_config()
    cfg["test_cases"].append(
        {"id": "tc_001", "evaluation": "exact_match", "expected_output": "other"}
    )
    errors, _ = validate(cfg)
    assert any("Duplicate test_case id" in e for e in errors)


def test_duplicate_prompt_ids_is_an_error():
    cfg = _exact_match_config()
    cfg["prompts"].append({"id": "v1", "template": "Other: {{ text }}"})
    errors, _ = validate(cfg)
    assert any("Duplicate prompt id" in e for e in errors)


def test_prompt_missing_template_is_an_error():
    cfg = _exact_match_config()
    cfg["prompts"] = [{"id": "v1"}]
    errors, _ = validate(cfg)
    assert any("template" in e for e in errors)


# --- template variable coverage errors ---


def test_template_var_missing_from_test_case_is_an_error():
    cfg = {
        "name": "Test",
        "models": ["openai/gpt-4o"],
        "inference": {"temperature": 0, "max_tokens": 100},
        "prompts": [{"id": "v1", "template": "Classify: {{ text }}"}],
        "test_cases": [
            {"id": "tc_001", "evaluation": "exact_match", "expected_output": "y"}
        ],
    }
    errors, _ = validate(cfg)
    assert any("text" in e for e in errors)


def test_template_var_provided_globally_is_ok():
    cfg = {
        "name": "Test",
        "models": ["openai/gpt-4o"],
        "inference": {"temperature": 0, "max_tokens": 100},
        "prompts": [{"id": "v1", "template": "Classify: {{ text }}"}],
        "variables": {"text": "hello"},
        "test_cases": [
            {"id": "tc_001", "evaluation": "exact_match", "expected_output": "y"}
        ],
    }
    errors, _ = validate(cfg)
    assert not any("text" in e for e in errors)


def test_template_var_provided_per_case_is_ok():
    cfg = {
        "name": "Test",
        "models": ["openai/gpt-4o"],
        "inference": {"temperature": 0, "max_tokens": 100},
        "prompts": [{"id": "v1", "template": "Classify: {{ text }}"}],
        "test_cases": [
            {
                "id": "tc_001",
                "evaluation": "exact_match",
                "expected_output": "y",
                "variables": {"text": "hello"},
            }
        ],
    }
    errors, _ = validate(cfg)
    assert not any("text" in e for e in errors)


# --- warnings ---


def test_single_brace_in_template_is_a_warning():
    cfg = _exact_match_config()
    cfg["prompts"] = [{"id": "v1", "template": "Choose from: {labels}"}]
    cfg["variables"]["labels"] = "a,b"
    _, warnings = validate(cfg)
    assert any("labels" in w and "single-brace" in w for w in warnings)


def test_double_brace_does_not_trigger_single_brace_warning():
    cfg = _exact_match_config()
    _, warnings = validate(cfg)
    assert not any("single-brace" in w for w in warnings)


def test_low_max_tokens_with_llm_judge_is_a_warning():
    cfg = _llm_judge_config()
    cfg["inference"]["max_tokens"] = 50
    _, warnings = validate(cfg)
    assert any("max_tokens" in w and "50" in w for w in warnings)


def test_adequate_max_tokens_with_llm_judge_is_not_a_warning():
    cfg = _llm_judge_config()
    cfg["inference"]["max_tokens"] = 200
    _, warnings = validate(cfg)
    assert not any("max_tokens" in w for w in warnings)


def test_long_expected_output_relative_to_max_tokens_is_a_warning():
    cfg = _exact_match_config()
    cfg["inference"]["max_tokens"] = 10
    cfg["test_cases"][0]["expected_output"] = "x" * 500
    _, warnings = validate(cfg)
    assert any("expected_output is long" in w for w in warnings)


# --- type checks for list fields ---


def test_non_list_prompts_is_an_error():
    cfg = _exact_match_config()
    cfg["prompts"] = "not a list"
    errors, _ = validate(cfg)
    assert any("'prompts' must be a list" in e for e in errors)


def test_non_list_test_cases_is_an_error():
    cfg = _exact_match_config()
    cfg["test_cases"] = "not a list"
    errors, _ = validate(cfg)
    assert any("'test_cases' must be a list" in e for e in errors)


def test_non_list_models_is_an_error():
    cfg = _exact_match_config()
    cfg["models"] = "openai/gpt-4o"
    errors, _ = validate(cfg)
    assert any("'models' must be a list" in e for e in errors)


# --- falsy-value edge cases ---


def test_quality_threshold_zero_is_not_an_error():
    cfg = {
        "name": "Test",
        "models": ["openai/gpt-4o"],
        "inference": {"temperature": 0, "max_tokens": 100},
        "evaluation": {
            "quality_threshold": 0.0,
            "variables": [{"name": "use_case", "weight": 1.0}],
        },
        "prompts": [{"id": "v1", "template": "Hi"}],
        "test_cases": [
            {"id": "tc_001", "evaluation": "weighted_match", "expected_output": []}
        ],
    }
    errors, _ = validate(cfg)
    assert not any("quality_threshold" in e for e in errors)


def test_float_max_tokens_is_an_error():
    # The API takes an int; 100.0 is rejected rather than silently truncated.
    cfg = _exact_match_config()
    cfg["inference"]["max_tokens"] = 100.0
    errors, _ = validate(cfg)
    assert any(
        "max_tokens must be a positive whole number like 100" in e for e in errors
    )


# --- jinja2 filter variable detection ---


def test_filtered_template_variable_is_detected():
    cfg = {
        "name": "Test",
        "models": ["openai/gpt-4o"],
        "inference": {"temperature": 0, "max_tokens": 100},
        "prompts": [{"id": "v1", "template": "Classify: {{ text | upper }}"}],
        "test_cases": [
            {"id": "tc_001", "evaluation": "exact_match", "expected_output": "y"}
        ],
    }
    errors, _ = validate(cfg)
    assert any("text" in e for e in errors)


def test_filtered_variable_provided_globally_is_ok():
    cfg = {
        "name": "Test",
        "models": ["openai/gpt-4o"],
        "inference": {"temperature": 0, "max_tokens": 100},
        "prompts": [{"id": "v1", "template": "Classify: {{ text | upper }}"}],
        "variables": {"text": "hello"},
        "test_cases": [
            {"id": "tc_001", "evaluation": "exact_match", "expected_output": "y"}
        ],
    }
    errors, _ = validate(cfg)
    assert not any("text" in e for e in errors)


# --- malformed values: clear errors instead of crashes later in the run ---


def _weighted_config(**overrides) -> dict:
    base = {
        "name": "Test",
        "models": ["openai/gpt-4o"],
        "evaluation": {
            "quality_threshold": 0.5,
            "variables": [
                {"name": "use_case", "weight": 0.5},
                {"name": "percent", "weight": 0.25, "tolerance": 0.2},
                {"name": "order", "weight": 0.25},
            ],
        },
        "prompts": [{"id": "v1", "template": "Hi"}],
        "test_cases": [
            {
                "id": "tc_001",
                "evaluation": "weighted_match",
                "expected_output": [{"use_case": "A", "percent": 1.0, "order": 1}],
            }
        ],
    }
    base.update(overrides)
    return base


def _set(path, value, base=_exact_match_config):
    """A builder: ``base()`` with the dotted ``path`` (list indexes as ints) set."""

    def build():
        cfg = base()
        *parents, last = path.split(".")
        node = cfg
        for key in parents:
            node = node[int(key)] if isinstance(node, list) else node[key]
        node[int(last) if isinstance(node, list) else last] = value
        return cfg

    return build


def test_valid_weighted_config_returns_no_issues():
    assert validate(_weighted_config()) == ([], [])


@pytest.mark.parametrize(
    "build, message",
    [
        (lambda: None, "Config file is empty"),
        (lambda: ["a", "b"], "must be a YAML mapping"),
        (_set("name", 123), "'name' must be a string"),
        (_set("quality_threshold", "high"), "between 0 and 100"),
        (_set("quality_threshold", 150), "between 0 and 100"),
        (_set("quality_threshold", True), "between 0 and 100"),
        (_set("concurrency", 0), "concurrency must be a whole number"),
        (_set("concurrency", "4"), "concurrency must be a whole number"),
        (_set("concurrency", 2.5), "concurrency must be a whole number"),
        (_set("inference", "fast"), "'inference' must be a mapping"),
        (_set("evaluation", ["x"]), "'evaluation' must be a mapping"),
        (_set("inference.temperature", "hot"), "inference.temperature must be"),
        (_set("prompts.0.id", 1), "'id' must be a string"),
        (_set("test_cases.0.id", 1), "'id' must be a string"),
        (_set("prompts.0.template", "{{ text "), "template: Jinja2 syntax error"),
        (_set("prompts.0.system", "{% if %}"), "system: Jinja2 syntax error"),
        (_set("test_cases.0.expected_output", True), "quote it"),
        (_set("test_cases.0.expected_output", 3), "quote it"),
        (
            _set(
                "evaluation.variables",
                [{"name": "use_case", "weight": 1}],
                _weighted_config,
            ),
            "missing weighted_match dimension(s): percent, order",
        ),
        (
            _set("test_cases.0.expected_output", [], _weighted_config),
            "non-empty list",
        ),
        (
            _set("test_cases.0.expected_output", "A", _weighted_config),
            "non-empty list",
        ),
        # Beyond the core list: other values that would crash mid-run.
        (_set("models.0", 7), "models[0] must be a model ID string"),
        (_set("max_cost_usd", "1$"), "max_cost_usd must be"),
        (_set("variables", ["text"]), "'variables' must be a mapping"),
        (_set("test_cases.0.variables", "x"), "variables' must be a mapping"),
        (_set("test_cases.0.tags", "easy"), "'tags' must be a list of strings"),
        (_set("prompts.0", "v1"), "prompts[0] must be a mapping"),
        (_set("test_cases.0", "tc"), "test_cases[0] must be a mapping"),
        (_set("test_cases.0.evaluation", ["exact_match"]), "unknown evaluation"),
        (
            _set("evaluation.judge_threshold", "high", _llm_judge_config),
            "judge_threshold must be",
        ),
        (_set("test_cases.0.criteria", 5, _llm_judge_config), "'criteria' must be"),
        (
            _set("evaluation.judge_model", ["a/b"], _llm_judge_config),
            "judge_model must be a model ID string",
        ),
        (
            _set("evaluation.quality_threshold", 80, _weighted_config),
            "evaluation.quality_threshold must be",
        ),
        (
            _set("evaluation.variables.1.weight", "x", _weighted_config),
            "'percent': weight",
        ),
        (
            _set("evaluation.variables.0", "use_case", _weighted_config),
            "evaluation.variables must be a list of",
        ),
        (
            _set("test_cases.0.expected_output", [{"use_case": "A"}], _weighted_config),
            "every weighted_match expected_output item",
        ),
        # Round 2: bounds, non-finite numbers, integer max_tokens, duplicates.
        (_set("quality_threshold", -1), "between 0 and 100"),
        (_set("concurrency", True), "concurrency must be a whole number"),
        (_set("quality_threshold", INF), "between 0 and 100"),
        (_set("quality_threshold", NAN), "between 0 and 100"),
        (_set("max_cost_usd", INF), "max_cost_usd must be"),
        (_set("inference.temperature", NAN), "inference.temperature must be"),
        (_set("inference.max_tokens", INF), "whole number like 100"),
        (_set("inference.max_tokens", 50.5), "whole number like 100"),
        (_set("inference.max_tokens", True), "whole number like 100"),
        (_set("inference.max_tokens", 100.0), "whole number like 100"),
        # Round 3: ints too large for a float are rejected, never an OverflowError.
        (_set("quality_threshold", HUGE), "between 0 and 100"),
        (_set("max_cost_usd", HUGE), "max_cost_usd must be"),
        (_set("concurrency", HUGE), "concurrency must be a whole number"),
        (_set("inference.temperature", HUGE), "inference.temperature must be"),
        (_set("inference.max_tokens", HUGE), "whole number like 100"),
        (
            _set("evaluation.judge_threshold", HUGE, _llm_judge_config),
            "judge_threshold must be",
        ),
        (
            _set("evaluation.quality_threshold", HUGE, _weighted_config),
            "evaluation.quality_threshold must be",
        ),
        (
            _set("evaluation.variables.0.weight", HUGE, _weighted_config),
            "'use_case': weight",
        ),
        (
            _set("test_cases.0.expected_output.0.percent", HUGE, _weighted_config),
            "every weighted_match expected_output item",
        ),
        # judge_model is type-checked even when no case uses llm_judge.
        (_set("evaluation", {"judge_model": 5}), "judge_model must be a model ID"),
        (_set("evaluation", {"judge_model": ["a"]}), "judge_model must be a model ID"),
        (_set("models", ["a/b", "a/b"]), "Duplicate model: 'a/b'"),
        (
            _set("evaluation.judge_threshold", NAN, _llm_judge_config),
            "judge_threshold must be",
        ),
        (
            _set("evaluation.quality_threshold", INF, _weighted_config),
            "evaluation.quality_threshold must be",
        ),
        (
            _set("evaluation.variables.0.weight", INF, _weighted_config),
            "'use_case': weight",
        ),
        (
            _set(
                "test_cases.0.expected_output",
                [{"use_case": "A", "percent": NAN, "order": 1}],
                _weighted_config,
            ),
            "every weighted_match expected_output item",
        ),
    ],
)
def test_malformed_value_is_an_error(build, message):
    errors, _ = validate(build())
    assert any(message in e for e in errors), errors


@pytest.mark.parametrize("key", ["inference", "evaluation", "variables"])
def test_null_block_is_treated_as_absent(key):
    cfg = _exact_match_config()
    cfg[key] = None
    cfg["prompts"] = [{"id": "v1", "template": "Hi"}]  # no variables needed
    assert validate(cfg) == ([], [])


# --- trial render: anything that would raise at run time fails validate ---


@pytest.mark.parametrize(
    "template, system, variables, exc_name",
    [
        ("{{ text.a.b }}", None, {"text": "x"}, "UndefinedError"),
        ("{{ 1 / 0 }}", None, {"text": "x"}, "ZeroDivisionError"),
        ("{{ ''.__class__.__mro__ }}", None, {"text": "x"}, ""),
        ("{{ text + 1 }}", None, {"text": "x"}, "TypeError"),
        ("{% include 'x.j2' %}", None, {"text": "x"}, "TypeError"),
        ("Hi {{ text }}", "{{ text.nope.deeper }}", {"text": "x"}, "UndefinedError"),
        # A non-string variable name can't become a render keyword.
        ("Hi {{ text }}", None, {"text": "x", 1: "y"}, "TypeError"),
    ],
)
def test_template_that_fails_to_render_is_a_validate_error(
    template, system, variables, exc_name
):
    prompt = {"id": "v1", "template": template}
    if system:
        prompt["system"] = system
    cfg = _exact_match_config(prompts=[prompt], variables=variables)
    errors, _ = validate(cfg)
    [error] = [e for e in errors if "failed to render" in e]
    assert "prompts['v1']" in error and "test_cases['tc_001']" in error
    assert exc_name in error


def test_trial_render_uses_per_case_variables():
    # Only the second case lacks the nested attribute — only it is reported.
    cfg = _exact_match_config(
        prompts=[{"id": "v1", "template": "{{ text.upper() }} {{ obj.k.z }}"}],
        variables={"text": "hello"},
        test_cases=[
            {
                "id": "ok",
                "evaluation": "exact_match",
                "expected_output": "y",
                "variables": {"obj": {"k": {"z": 1}}},
            },
            {
                "id": "bad",
                "evaluation": "exact_match",
                "expected_output": "y",
                "variables": {"obj": "flat"},
            },
        ],
    )
    [error] = validate(cfg)[0]
    assert "test_cases['bad']" in error and "UndefinedError" in error


# ── round 4 ──────────────────────────────────────────────────────────────────


def test_self_referencing_alias_is_an_error():
    import yaml

    config = _exact_match_config()
    config["test_cases"][0]["expected_output"] = yaml.safe_load("&a [*a]")
    errors, _ = validate(config)
    assert errors == ["Config contains a self-referencing YAML alias (e.g. `&a [*a]`)"]


@pytest.mark.parametrize("value, ok", [(100, True), (101, False)])
def test_concurrency_is_capped(value, ok):
    errors, _ = validate(_exact_match_config(concurrency=value))
    assert (errors == []) is ok


def test_unhashable_weighted_use_case_is_an_error():
    config = _exact_match_config(
        evaluation={
            "quality_threshold": 0.8,
            "variables": [
                {"name": "use_case", "weight": 0.5},
                {"name": "percent", "weight": 0.25},
                {"name": "order", "weight": 0.25},
            ],
        },
        test_cases=[
            {
                "id": "w1",
                "evaluation": "weighted_match",
                "expected_output": [{"use_case": {"a"}, "percent": 1, "order": 1}],
            }
        ],
    )
    errors, _ = validate(config)
    assert any("needs use_case" in e for e in errors), errors


@pytest.mark.parametrize(
    "value, ok", [(4096, True), (None, True), (0, False), (1.5, False), ("4k", False)]
)
def test_judge_max_tokens_must_be_a_positive_int(value, ok):
    config = _llm_judge_config()
    config["evaluation"]["judge_max_tokens"] = value
    errors, _ = validate(config)
    assert (errors == []) is ok, errors


def test_validate_compiles_each_template_once():
    from evalblink import runner

    runner._compile.cache_clear()
    cases = [
        {"id": f"c{i}", "evaluation": "exact_match", "expected_output": "x"}
        for i in range(50)
    ]
    template = "Classify {{ text }} #" + "unique-template-for-this-test"
    config = _exact_match_config(
        prompts=[{"id": "v1", "template": template}], test_cases=cases
    )
    assert validate(config)[0] == []
    info = runner._compile.cache_info()
    assert info.misses <= 2 and info.hits >= 49  # compiled once, reused per case
