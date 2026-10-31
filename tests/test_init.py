"""`evalblink init`: three questions, then a commented starter that validates."""

from __future__ import annotations

from collections import deque

import httpx
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
    assert config["concurrency"] == 2  # free models rate-limit parallel bursts
    assert {tc["evaluation"] for tc in config["test_cases"]} == {"exact_match"}
    assert len(config["test_cases"]) == 3


def test_the_absolute_path_of_the_written_file_is_printed(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.chdir(tmp_path)
    assert _init(monkeypatch, ["--yes"]) == 0
    path = tmp_path / "benchmarks" / "my-benchmark.yaml"
    assert f"Created {path}\n" in capsys.readouterr().out


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


@pytest.mark.parametrize("answer", [False, ""])  # "" = Enter: the default is no
def test_declined_overwrite_leaves_the_file(monkeypatch, tmp_path, answer):
    out = tmp_path / "x.yaml"
    out.write_text("mine")
    assert _init(monkeypatch, [str(out)], ["X", "", ""], confirms=[answer]) == 0
    assert out.read_text() == "mine"


def test_a_directory_is_refused_before_any_overwrite_question(monkeypatch, tmp_path):
    code = _init(monkeypatch, [str(tmp_path)], ["X", "", ""])  # no confirm asked
    assert isinstance(code, str) and "is a directory" in code


def test_overwrite_prompt_escapes_markup_in_the_path(monkeypatch, tmp_path):
    (tmp_path / "[").mkdir()
    out = tmp_path / "[" / "x].yaml"  # the path contains "[/x]"
    out.write_text("old")
    assert _init(monkeypatch, [str(out)], ["X", "", ""], confirms=[True]) == 0
    assert _load(out)["name"] == "X"


@pytest.mark.parametrize(
    "failure", [RuntimeError("down"), httpx.ConnectError("offline")]
)
def test_catalog_down_falls_back_to_the_default_model(
    monkeypatch, tmp_path, capsys, failure
):
    out = tmp_path / "x.yaml"
    code = _init(monkeypatch, [str(out)], ["X", "", ""], catalog=failure)
    assert code == 0
    assert _load(out)["models"] == [starter.DEFAULT_MODEL]
    stdout = capsys.readouterr().out
    assert "Could not fetch the OpenRouter model catalog" in stdout
    assert "not in the OpenRouter catalog" not in stdout


def test_retired_default_is_replaced_by_a_free_suggestion(
    monkeypatch, tmp_path, capsys
):
    catalog = {
        "a/audio-model": {**FREE, "text": False},  # outputs audio: never suggested
        "a/paid": CATALOG["a/b"],  # never suggested
        **{f"z/text-{i}:free": FREE for i in range(6)},
    }
    out = tmp_path / "x.yaml"
    assert _init(monkeypatch, [str(out)], ["X", "", ""], catalog=catalog) == 0
    assert _load(out)["models"] == ["z/text-0:free"]
    stdout = capsys.readouterr().out
    assert "a/paid" not in stdout
    assert "z/text-4:free" in stdout and "z/text-5:free" not in stdout  # top five
    assert "a/audio-model" not in stdout


def test_the_small_paid_model_is_suggested_not_preselected(
    monkeypatch, tmp_path, capsys
):
    catalog = {**CATALOG, starter.PAID_MODEL: {"prompt": 2e-7, "completion": 4e-7}}
    out = tmp_path / "x.yaml"
    assert _init(monkeypatch, [str(out)], ["X", "", ""], catalog=catalog) == 0
    assert _load(out)["models"] == [starter.DEFAULT_MODEL]
    stdout = capsys.readouterr().out
    assert f"rate limits:\n  {starter.PAID_MODEL} ($0.200 in / $0.400 out" in stdout


def test_a_retired_paid_suggestion_is_not_shown(monkeypatch, tmp_path, capsys):
    assert _init(monkeypatch, [str(tmp_path / "x.yaml")], ["X", "", ""]) == 0
    assert starter.PAID_MODEL not in capsys.readouterr().out


def test_no_free_model_in_the_catalog_keeps_the_shipped_default(monkeypatch, tmp_path):
    out = tmp_path / "x.yaml"
    catalog = {"a/b": CATALOG["a/b"]}
    assert _init(monkeypatch, [str(out), "--yes"], catalog=catalog) == 0
    assert _load(out)["models"] == [starter.DEFAULT_MODEL]


def test_retired_judge_is_flagged(monkeypatch, tmp_path, capsys):
    out = tmp_path / "x.yaml"
    assert _init(monkeypatch, [str(out)], ["X", "llm_judge", ""]) == 0
    stdout = capsys.readouterr().out
    assert f"{starter.JUDGE_MODEL} is not in the OpenRouter catalog" in stdout
    assert f"{starter.DEFAULT_MODEL} is not" not in stdout


def test_yes_flags_a_default_absent_from_the_catalog(monkeypatch, tmp_path, capsys):
    out = tmp_path / "x.yaml"
    catalog = {"a/b": CATALOG["a/b"]}
    assert _init(monkeypatch, [str(out), "-y"], catalog=catalog) == 0
    assert "is not in the OpenRouter catalog" in capsys.readouterr().out


@pytest.mark.parametrize("out", ["file/x.yaml", "\ud800.yaml"])
def test_unwritable_output_path_is_a_clean_error(monkeypatch, tmp_path, out):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "file").write_text("")  # a parent that is a file
    code = _init(monkeypatch, [out], ["X", "", ""])
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


def test_the_default_model_is_listed_first(monkeypatch, tmp_path, capsys):
    catalog = {"a/first-by-id:free": FREE, **CATALOG}
    assert (
        _init(monkeypatch, [str(tmp_path / "x.yaml")], ["X", "", ""], catalog=catalog)
        == 0
    )
    stdout = capsys.readouterr().out
    assert stdout.index(starter.DEFAULT_MODEL) < stdout.index("a/first-by-id:free")


@pytest.mark.parametrize("mode", starter.MODES)
def test_the_csv_in_the_comment_is_the_starter_cases(mode, tmp_path):
    """Doing what the starter comment says gives the same test cases."""
    text = starter.render("Demo", [starter.DEFAULT_MODEL], mode)
    lines = text.splitlines()
    head = next(i for i, line in enumerate(lines) if line.startswith("#   id,"))
    csv_rows = []
    for line in lines[head:]:
        if line == "#":
            break
        csv_rows.append(line.removeprefix("#   "))
    (tmp_path / "cases.csv").write_text("\n".join(csv_rows) + "\n", encoding="utf-8")

    swap = f"test_cases: {{file: cases.csv, evaluation: {mode}}}"
    assert f"#   {swap}" in text
    start = text.index("\ntest_cases:\n")
    yaml_path = tmp_path / "bench.yaml"
    yaml_path.write_text(text[:start] + "\n" + swap + "\n", encoding="utf-8")

    from_csv = main.load_config(str(yaml_path))
    assert validate(from_csv) == ([], [])
    inline = yaml.safe_load(text)["test_cases"]
    assert from_csv["test_cases"] == inline
