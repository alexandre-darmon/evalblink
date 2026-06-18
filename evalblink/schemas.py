"""Data shapes for evalblink — config and result dicts.

These are ``TypedDict``s, not Pydantic models: the pipeline keeps passing the
plain dicts that come straight out of the YAML / get built during a run, so
these definitions add typing and documentation with zero runtime change. No
logic lives here.
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict


# ----------------------------------------------------------------------------
# Config side — the shape of a parsed benchmark YAML file.
# ----------------------------------------------------------------------------


class InferenceConfig(TypedDict, total=False):
    temperature: float
    max_tokens: int


class EvaluationParam(TypedDict, total=False):
    """One weighted dimension for ``weighted_match`` (use_case / percent / order)."""

    name: str
    weight: float
    tolerance: float


class EvaluationConfig(TypedDict, total=False):
    # weighted_match
    quality_threshold: float
    variables: list[EvaluationParam]
    # llm_judge
    judge_model: str
    judge_threshold: float


class Prompt(TypedDict, total=False):
    id: str
    system: str
    template: str


class TestCase(TypedDict, total=False):
    id: str
    variables: dict[str, Any]
    evaluation: str  # "exact_match" | "weighted_match" | "llm_judge"
    expected_output: Any
    criteria: str
    tags: list[str]


class BenchmarkConfig(TypedDict, total=False):
    name: str
    inference: InferenceConfig
    evaluation: EvaluationConfig
    prompts: list[Prompt]
    models: list[str]
    variables: dict[str, Any]
    test_cases: list[TestCase]


# ----------------------------------------------------------------------------
# Result side — the shape produced by a run and written to disk.
# ----------------------------------------------------------------------------


class RequestResult(TypedDict):
    """A single OpenRouter completion plus its usage accounting."""

    response: str
    prompt_tokens: int
    completion_tokens: int
    cost: float


class TestCaseResult(TypedDict):
    id: str
    tags: list[str]
    evaluation: str
    match_result: bool
    match_score: Optional[float]
    response: str
    reasoning: Optional[str]
    expected: Any
    prompt_tokens: int
    completion_tokens: int
    cost: float
    judge_prompt_tokens: int
    judge_completion_tokens: int
    judge_cost: float
    total_cost: float


class RunResult(TypedDict):
    model: str
    prompt_id: str
    success: int
    total: int
    score: float
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cost: float
    test_cases: list[TestCaseResult]
