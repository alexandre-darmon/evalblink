"""Tests for runner.render_template (Jinja substitution + variable merging)."""

from __future__ import annotations

from evalblink.runner import render_template


def test_substitutes_global_and_testcase_variables():
    prompt = {"template": "Hi {{ name }} from {{ place }}"}
    rendered, system = render_template(
        prompt, {"name": "Alex"}, {"variables": {"place": "Paris"}}
    )
    assert rendered == "Hi Alex from Paris"
    assert system is None


def test_testcase_variables_override_globals():
    prompt = {"template": "{{ label }}"}
    rendered, _ = render_template(
        prompt, {"label": "global"}, {"variables": {"label": "local"}}
    )
    assert rendered == "local"


def test_renders_system_prompt_when_present():
    prompt = {"template": "{{ x }}", "system": "System for {{ x }}"}
    rendered, system = render_template(prompt, {"x": "ctx"}, {"variables": {}})
    assert rendered == "ctx"
    assert system == "System for ctx"


def test_single_brace_is_left_literal():
    # Guards against regressing to Python .format-style templates (a real bug class).
    prompt = {"template": "Choose from: {labels}"}
    rendered, _ = render_template(prompt, {"labels": "a,b"}, {"variables": {}})
    assert rendered == "Choose from: {labels}"
