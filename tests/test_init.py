"""`evalblink init`: three questions, then a commented starter that validates."""

from __future__ import annotations

from collections import deque

import pytest
import yaml
from rich.text import Text

from evalblink import main, starter
from evalblink.validator import validate

FREE = {"prompt": 0.0, "completion": 0.0, "context_length": 32768}
CATALOG = {
    starter.DEFAULT_MODEL: FREE,
    "a/b": {"prompt": 1e-6, "completion": 1e-6, "context_length": 8000},
}


def _scripted(answers):
    """A ``Prompt``/``Confirm`` stand-in that, like Rich, returns the default
    for an empty answer."""

    class Scripted:
        @classmethod
        def ask(cls, question, **kwargs):
            Text.from_markup(question)  # Rich renders the question as markup
            answer = answers.popleft()
            if answer == "" and "default" in kwargs:
                return kwargs["default"]
            return answer

    return Scripted


def _init(monkeypatch, argv, prompts=(), confirms=(), catalog=CATALOG):
    def fetch(client, use_cache=True):
        if isinstance(catalog, Exception):
            raise catalog
        return catalog

    monkeypatch.setattr(main.openrouter, "fetch_models", fetch)
    prompts, confirms = deque(prompts), deque(confirms)
    monkeypatch.setattr(main, "Prompt", _scripted(prompts))
    monkeypatch.setattr(main, "Confirm", _scripted(confirms))
    with pytest.raises(SystemExit) as exc:
        main.main(["init", *argv])
    assert not prompts and not confirms, "unused scripted answers"
    return exc.value.code


def _load(path):
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert validate(config) == ([], [])
    return config


def test_enter_at_every_question_writes_a_valid_exact_match_starter(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    assert _init(monkeypatch, [], prompts=["", "", ""]) == 0
    config = _load(tmp_path / "benchmarks" / "my-benchmark.yaml")
    assert config["name"] == "My benchmark"
    assert config["models"] == [starter.DEFAULT_MODEL]
    assert config["inference"] == {"temperature": 0, "max_tokens": 1024}
    assert {tc["evaluation"] for tc in config["test_cases"]} == {"exact_match"}
    assert len(config["test_cases"]) == 3


def test_llm_judge_with_several_models(monkeypatch, tmp_path, capsys):
    out = tmp_path / "judge.yaml"
    code = _init(
        monkeypatch,
        [str(out)],
        prompts=["Support summaries", "llm_judge", " a/b, typo/model ,a/b,"],
    )
    assert code == 0
    config = _load(out)
    assert config["models"] == ["a/b", "typo/model"]  # trimmed, deduplicated
    assert config["evaluation"]["judge_model"] == starter.JUDGE_MODEL
    assert {tc["evaluation"] for tc in config["test_cases"]} == {"llm_judge"}
    stdout = capsys.readouterr().out
    assert "typo/model is not in the OpenRouter catalog" in stdout
    assert "a/b is not" not in stdout


def test_invalid_answers_are_asked_again(monkeypatch, tmp_path, capsys):
    out = tmp_path / "x.yaml"
    prompts = ["   ", "X", "exact_match", ",, ,", "a b", "a/b"]
    assert _init(monkeypatch, [str(out)], prompts=prompts) == 0
    assert _load(out)["models"] == ["a/b"]
    stdout = capsys.readouterr().out
    assert stdout.count("A name is required.") == 1
    assert stdout.count("Enter one or more model IDs") == 2


def test_yes_asks_nothing_and_never_overwrites(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert _init(monkeypatch, ["--yes"]) == 0  # no scripted answers: none asked
    path = tmp_path / "benchmarks" / "my-benchmark.yaml"
    assert _load(path)["models"] == [starter.DEFAULT_MODEL]
    path.write_text("mine")
    code = _init(monkeypatch, ["-y"])
    assert isinstance(code, str) and "already exists" in code
    assert path.read_text() == "mine"


def test_declined_overwrite_leaves_the_file(monkeypatch, tmp_path):
    out = tmp_path / "x.yaml"
    out.write_text("mine")
    assert _init(monkeypatch, [str(out)], ["X", "", ""], confirms=[False]) == 0
    assert out.read_text() == "mine"


def test_overwrite_prompt_escapes_markup_in_the_path(monkeypatch, tmp_path):
    (tmp_path / "[").mkdir()
    out = tmp_path / "[" / "x].yaml"  # the path contains "[/x]"
    out.write_text("old")
    assert _init(monkeypatch, [str(out)], ["X", "", ""], confirms=[True]) == 0
    assert _load(out)["name"] == "X"


def test_catalog_down_falls_back_to_the_default_model(monkeypatch, tmp_path, capsys):
    out = tmp_path / "x.yaml"
    code = _init(monkeypatch, [str(out)], ["X", "", ""], catalog=RuntimeError("down"))
    assert code == 0
    assert _load(out)["models"] == [starter.DEFAULT_MODEL]
    stdout = capsys.readouterr().out
    assert "Could not fetch the OpenRouter model catalog" in stdout
    assert "not in the OpenRouter catalog" not in stdout


def test_retired_default_is_replaced_by_a_free_suggestion(
    monkeypatch, tmp_path, capsys
):
    catalog = {
        "a/audio-model": FREE,  # zero-priced but not a `:free` id: listed last
        "z/text:free": FREE,
    }
    out = tmp_path / "x.yaml"
    assert _init(monkeypatch, [str(out)], ["X", "", ""], catalog=catalog) == 0
    assert _load(out)["models"] == ["z/text:free"]
    stdout = capsys.readouterr().out
    assert stdout.index("z/text:free") < stdout.index("a/audio-model")


@pytest.mark.parametrize("out", ["{dir}", "\ud800.yaml"])
def test_unwritable_output_path_is_a_clean_error(monkeypatch, tmp_path, out):
    monkeypatch.chdir(tmp_path)
    out = out.format(dir=tmp_path)
    confirms = [True] if out == str(tmp_path) else []  # it "exists"
    code = _init(monkeypatch, [out], ["X", "", ""], confirms=confirms)
    assert isinstance(code, str) and "Cannot write" in code


def test_eof_is_a_clean_exit(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main.openrouter, "fetch_models", lambda c, use_cache: {})

    def eof(*a, **k):
        raise EOFError

    monkeypatch.setattr(main.Prompt, "ask", eof)
    with pytest.raises(SystemExit) as exc:
        main.main(["init"])
    assert "No input" in str(exc.value.code)


@pytest.mark.parametrize("mode", starter.MODES)
@pytest.mark.parametrize(
    "name",
    [
        "x",
        "yes",
        "123",
        "null",
        "~",
        "a: b # c",
        "- dash",
        "\"q\" 's'",
        "{x}",
        "{{ x }}",
        "[x]",
        "!tag",
        "@at",
        "%p",
        "é 日本 🙂",
        "x" * 300,
    ],
)
def test_starter_round_trips_any_name_and_validates_clean(name, mode):
    models = ["a/b", "c/d:free"]
    config = yaml.safe_load(starter.render(name, models, mode))
    assert config["name"] == name
    assert config["models"] == models
    assert validate(config) == ([], [])
