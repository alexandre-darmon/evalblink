"""json_match, json_schema and user-written Python scorers."""

from __future__ import annotations

import textwrap

import pytest

from evalblink import dataset, evaluator, main, runner
from evalblink.evaluator import ScorerError, json_match, json_schema, python_score
from evalblink.validator import validate

# ── json_match ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "reply, score",
    [
        ('{"label": "Billing", "priority": 2, "extra": true}', 1.0),
        ('```json\n{"label": "billing", "priority": 2.0}\n```', 1.0),
        ('{"label": "billing", "priority": 3}', 0.5),
        ('{"label": "other"}', 0.0),
        ("billing", 0.0),  # not JSON: a task failure
        ('["billing", 2]', 0.0),  # not an object
    ],
)
def test_json_match_scores_the_share_of_expected_values(reply, score):
    assert json_match(reply, {"label": "billing", "priority": 2}) == score


def test_json_match_nested_lists_and_types():
    expected = {"items": [{"sku": "A1", "qty": 2}, {"sku": "B2", "qty": 1}]}
    assert (
        json_match(
            '{"items": [{"sku": "a1", "qty": 2}, {"sku": "B2", "qty": 1}]}', expected
        )
        == 1.0
    )
    # Same items, other order: arrays compare in order.
    assert (
        json_match(
            '{"items": [{"sku": "B2", "qty": 1}, {"sku": "A1", "qty": 2}]}', expected
        )
        == 0.0
    )
    # A different length matches nothing in the array.
    assert json_match('{"items": [{"sku": "A1", "qty": 2}]}', expected) == 0.0
    # Booleans are not numbers.
    assert json_match('{"ok": 1}', {"ok": True}) == 0.0
    assert json_match('{"ok": true}', {"ok": True}) == 1.0
    assert json_match('{"v": null}', {"v": None}) == 1.0


# ── json_schema ───────────────────────────────────────────────────────────────

SCHEMA = {
    "type": "object",
    "required": ["label", "confidence"],
    "properties": {
        "label": {"enum": ["billing", "order_issue"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}


def test_json_schema_valid_and_reasons():
    assert json_schema('{"label": "billing", "confidence": 0.9}', SCHEMA) == (
        True,
        None,
    )
    ok, reason = json_schema('{"label": "billing", "confidence": 3}', SCHEMA)
    assert not ok and reason.startswith("confidence: 3 is greater than")
    ok, reason = json_schema("Sure! Here it is", SCHEMA)
    assert not ok and reason.startswith("reply is not JSON")


@pytest.mark.parametrize(
    "schema, message",
    [
        ("an object", "must be a JSON Schema object"),
        ({"type": "nope"}, "is not a valid JSON Schema"),
    ],
)
def test_validate_rejects_a_bad_schema(schema, message):
    config = _config(
        [{"id": "s", "evaluation": "json_schema", "expected_output": schema}]
    )
    errors, _ = validate(config)
    assert any(message in e for e in errors), errors


@pytest.mark.parametrize("expected", [None, "billing", {}, [], 3])
def test_validate_rejects_a_bad_json_match_expected(expected):
    case = {"id": "j", "evaluation": "json_match"}
    if expected is not None:
        case["expected_output"] = expected
    errors, _ = validate(_config([case]))
    assert any("json_match expected_output must be" in e for e in errors), errors


# ── python scorers ────────────────────────────────────────────────────────────


@pytest.fixture
def scorer_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluator, "_SCORERS", {})
    (tmp_path / "scorers.py").write_text(
        textwrap.dedent(
            """
            def short(response, case):
                return len(response) <= case["variables"]["limit"]

            def graded(response, case):
                return {"passed": "refund" in response, "score": 0.4, "reason": "why"}

            def broken(response, case):
                raise KeyError("oops")

            def weird(response, case):
                return 0.7

            NOT_A_FUNCTION = 3
            """
        )
    )
    return tmp_path


def test_python_score_return_shapes(scorer_dir):
    short = evaluator.load_scorer("scorers.py:short", str(scorer_dir))
    assert python_score(short, "hi", {"variables": {"limit": 5}}) == (True, 1.0, None)
    graded = evaluator.load_scorer("scorers.py:graded", str(scorer_dir))
    assert python_score(graded, "a refund", {}) == (True, 0.4, "why")
    with pytest.raises(ScorerError, match="KeyError"):
        python_score(
            evaluator.load_scorer("scorers.py:broken", str(scorer_dir)), "", {}
        )
    with pytest.raises(ScorerError, match="return True/False"):
        python_score(evaluator.load_scorer("scorers.py:weird", str(scorer_dir)), "", {})


@pytest.mark.parametrize(
    "spec, message",
    [
        ("scorers.py", "must look like 'scorers.py:function_name'"),
        ("missing.py:f", "scorer file not found: missing.py"),
        ("scorers.py:nope", "has no function 'nope'"),
        ("scorers.py:NOT_A_FUNCTION", "has no function 'NOT_A_FUNCTION'"),
    ],
)
def test_validate_checks_the_scorer(scorer_dir, spec, message):
    config = _config([{"id": "p", "evaluation": "python", "scorer": spec}])
    config["_base_dir"] = str(scorer_dir)
    errors, _ = validate(config)
    assert any(message in e for e in errors), errors


def test_validate_requires_a_scorer():
    errors, _ = validate(_config([{"id": "p", "evaluation": "python"}]))
    assert any("python requires 'scorer'" in e for e in errors)


def test_a_scorer_that_fails_to_import_is_a_validation_error(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluator, "_SCORERS", {})
    (tmp_path / "bad.py").write_text("import not_a_module_xyz\n")
    config = _config([{"id": "p", "evaluation": "python", "scorer": "bad.py:f"}])
    config["_base_dir"] = str(tmp_path)
    errors, _ = validate(config)
    assert any("could not import bad.py: ModuleNotFoundError" in e for e in errors)


def test_pipeline_runs_every_new_mode(scorer_dir, monkeypatch):
    replies = iter(
        [
            '{"label": "billing", "priority": 3}',
            '{"label": "billing", "confidence": 0.5}',
            "no refunds here",
            "anything",
        ]
    )

    def fake(client, prompt, model, *a, **k):
        return {
            "response": next(replies),
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "cost": 0.0,
            "latency_s": 0.1,
            "from_cache": False,
        }

    monkeypatch.setattr(runner, "openrouter_request", fake)
    config = _config(
        [
            {
                "id": "j",
                "evaluation": "json_match",
                "expected_output": {"label": "billing", "priority": 2},
            },
            {"id": "s", "evaluation": "json_schema", "expected_output": SCHEMA},
            {"id": "p", "evaluation": "python", "scorer": "scorers.py:graded"},
            {"id": "b", "evaluation": "python", "scorer": "scorers.py:broken"},
        ]
    )
    config["_base_dir"] = str(scorer_dir)
    assert validate(config) == ([], [])
    results, _ = runner.run(config)
    j, s, p, b = results[0]["test_cases"]
    assert (j["match_score"], j["match_result"]) == (0.5, False)
    assert (s["match_score"], s["match_result"]) == (1.0, True)
    assert (p["match_score"], p["match_result"], p["reasoning"]) == (0.4, True, "why")
    # A scorer that raises is a pipeline error, not a 0 for the model.
    assert b["status"] == "scorer_error" and b["match_score"] is None
    assert "scorers.py:broken: KeyError" in b["error"]
    assert results[0]["errors"] == 1
    line = runner._progress_line(
        4, 4, {"model": "m", "prompt_id": "v1", "test_case": b}
    )
    assert "⚠ scorer error: scorers.py:broken: KeyError" in line


def test_load_config_resolves_scorers_next_to_the_yaml(
    tmp_path, monkeypatch, scorer_dir
):
    bench = scorer_dir / "bench.yaml"
    bench.write_text(
        textwrap.dedent(
            """
            name: t
            models: [m/x]
            prompts: [{id: v1, template: "{{ q }}"}]
            test_cases:
              - {id: p, evaluation: python, scorer: "scorers.py:short", variables: {q: x, limit: 3}}
            """
        )
    )
    monkeypatch.chdir(tmp_path.parent)  # elsewhere: the path is relative to the YAML
    config = main.load_config(str(bench))
    assert validate(config) == ([], [])


def test_csv_parses_json_expected_outputs(tmp_path):
    (tmp_path / "c.csv").write_text(
        "id,evaluation,expected_output,scorer\n"
        'j,json_match,"{""label"": ""billing""}",\n'
        's,json_schema,"{""type"": ""object""}",\n'
        "p,python,,scorers.py:short\n"
    )
    config, _ = dataset.expand({"test_cases": {"file": "c.csv"}}, str(tmp_path))
    j, s, p = config["test_cases"]
    assert j["expected_output"] == {"label": "billing"}
    assert s["expected_output"] == {"type": "object"}
    assert p["scorer"] == "scorers.py:short"


def test_csv_bad_json_names_the_mode(tmp_path):
    (tmp_path / "c.csv").write_text(
        "id,evaluation,expected_output\nj,json_match,{oops\n"
    )
    with pytest.raises(
        dataset.DatasetError, match="c.csv:2: a json_match expected_output must be JSON"
    ):
        dataset.expand({"test_cases": {"file": "c.csv"}}, str(tmp_path))


def _config(test_cases):
    return {
        "name": "Structured",
        "models": ["m/x"],
        "prompts": [{"id": "v1", "template": "Classify"}],
        "test_cases": test_cases,
        "concurrency": 1,
    }
