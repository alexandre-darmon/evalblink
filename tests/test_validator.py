"""Tests for evalblink.validator — all offline, no API calls."""

from __future__ import annotations

from evalblink.validator import validate


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


def test_float_max_tokens_is_not_an_error():
    cfg = _exact_match_config()
    cfg["inference"]["max_tokens"] = 100.0
    errors, _ = validate(cfg)
    assert not any("max_tokens" in e for e in errors)


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
