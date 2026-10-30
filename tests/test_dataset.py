"""Test cases loaded from CSV / TSV / JSONL files (``test_cases: {file: …}``)."""

from __future__ import annotations

import json

import pytest
import yaml

from evalblink import dataset, main
from evalblink.validator import validate

CONFIG = {
    "name": "Data",
    "models": ["m/a"],
    "prompts": [{"id": "v1", "template": "Classify: {{ text }} ({{ labels }})"}],
    "variables": {"labels": "pos, neg"},
}


def _load(tmp_path, cases_file, text, spec=None, **write):
    (tmp_path / cases_file).write_text(text, **write)
    config = {**CONFIG, "test_cases": spec or {"file": cases_file}}
    path = tmp_path / "bench.yaml"
    path.write_text(yaml.safe_dump(config))
    return main.load_config(str(path))


def test_csv_rows_become_test_cases(tmp_path):
    config = _load(
        tmp_path,
        "cases.csv",
        "id,text,expected_output,tags,human_score\n"
        'c1,"great, really",pos,easy; short ,4\n'
        "\n"  # blank line skipped
        "c2,awful,neg,,\n",
        {"file": "cases.csv", "evaluation": "exact_match"},
    )
    assert config["test_cases"] == [
        {
            "id": "c1",
            "expected_output": "pos",
            "tags": ["easy", "short"],
            "human_score": 4,
            "variables": {"text": "great, really"},
            "evaluation": "exact_match",
        },
        {
            "id": "c2",
            "expected_output": "neg",
            "variables": {"text": "awful"},
            "evaluation": "exact_match",
        },
    ]
    assert validate(config) == ([], [])


def test_tsv_jsonl_and_excel_bom(tmp_path):
    tsv = _load(
        tmp_path,
        "c.tsv",
        "﻿id\ttext\texpected_output\tevaluation\nc1\ta, b\tpos\texact_match\n",
    )
    assert tsv["test_cases"][0]["variables"] == {"text": "a, b"}
    jsonl = _load(
        tmp_path,
        "c.jsonl",
        json.dumps(
            {
                "id": "j1",
                "evaluation": "llm_judge",
                "criteria": "short",
                "variables": {"text": "x"},
                "labels": "override",
            }
        )
        + "\n\n",
    )
    assert jsonl["test_cases"] == [
        {
            "id": "j1",
            "evaluation": "llm_judge",
            "criteria": "short",
            "variables": {"text": "x", "labels": "override"},
        }
    ]


def test_rows_without_an_id_are_named_by_their_line(tmp_path):
    config = _load(
        tmp_path,
        "c.csv",
        "text,expected_output\na,pos\nb,neg\n",
        {"file": "c.csv", "evaluation": "exact_match"},
    )
    assert [tc["id"] for tc in config["test_cases"]] == ["line_2", "line_3"]


def test_a_row_evaluation_wins_over_the_default(tmp_path):
    config = _load(
        tmp_path,
        "c.csv",
        "id,text,evaluation,expected_output,criteria\n"
        "a,x,,pos,\n"
        "b,y,llm_judge,,be short\n",
        {"file": "c.csv", "evaluation": "exact_match"},
    )
    assert [tc["evaluation"] for tc in config["test_cases"]] == [
        "exact_match",
        "llm_judge",
    ]


def test_weighted_expected_output_is_json_in_a_csv(tmp_path):
    import csv
    import io

    expected = [{"use_case": "a", "percent": 60, "order": 1}]
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["id", "text", "evaluation", "expected_output"])
    writer.writerow(["w1", "x", "weighted_match", json.dumps(expected)])
    config = _load(tmp_path, "c.csv", buffer.getvalue())
    assert config["test_cases"][0]["expected_output"] == expected


def test_the_path_is_relative_to_the_yaml(tmp_path, monkeypatch):
    (tmp_path / "evals" / "data").mkdir(parents=True)
    (tmp_path / "evals" / "data" / "c.csv").write_text("id,text\nc1,x\n")
    (tmp_path / "evals" / "b.yaml").write_text(
        yaml.safe_dump({**CONFIG, "test_cases": {"file": "data/c.csv"}})
    )
    monkeypatch.chdir(tmp_path)
    config = main.load_config("evals/b.yaml")
    assert config["test_cases"][0]["id"] == "c1"


def test_a_misspelled_column_is_flagged(tmp_path, capsys):
    _load(tmp_path, "c.csv", "id,text,expeced_output\nc1,x,pos\n")
    err = capsys.readouterr().err
    assert "column 'expeced_output' is passed as a template variable" in err
    assert "did you mean 'expected_output'?" in err


@pytest.mark.parametrize(
    "name, text, spec, message",
    [
        ("c.csv", "", None, "c.csv is empty"),
        ("c.csv", "id,,text\n", None, "c.csv:1: column 2 has no name"),
        ("c.csv", "id,text,id\n", None, "duplicate column 'id'"),
        ("c.csv", "id,text\nc1,a,b\n", None, "c.csv:2: 3 cells but 2 columns"),
        ("c.csv", "id,human_score\nc1,lots\n", None, "human_score must be a whole"),
        ("c.csv", "id,text\n", None, "c.csv has no test cases"),
        ("c.xlsx", "x", None, "unsupported file type '.xlsx'"),
        (
            "c.jsonl",
            '{"id": 1}\n[1]\n',
            None,
            "c.jsonl:2: each line must be a JSON object",
        ),
        ("c.jsonl", "{oops\n", None, "c.jsonl:1: invalid JSON"),
        ("c.jsonl", '{"variables": 3}\n', None, "'variables' must be an object"),
        (
            "c.csv",
            "id,evaluation,expected_output\nw,weighted_match,[oops\n",
            None,
            "c.csv:2: a weighted_match expected_output must be JSON",
        ),
        ("c.csv", "id\nc1\n", {"file": "c.csv", "fiel": 1}, "unknown key 'fiel'"),
        ("c.csv", "id\nc1\n", {"file": ""}, "test_cases.file must be"),
        ("c.csv", "id\nc1\n", {"file": "missing.csv"}, "Cannot read missing.csv"),
    ],
)
def test_bad_files_exit_naming_the_file_and_line(tmp_path, name, text, spec, message):
    with pytest.raises(SystemExit) as exc:
        _load(tmp_path, name, text, spec)
    assert isinstance(exc.value.code, str)
    assert exc.value.code.startswith("Invalid test cases in ")
    assert message in exc.value.code


def test_a_non_utf8_file_is_a_clean_error(tmp_path):
    with pytest.raises(SystemExit) as exc:
        _load(tmp_path, "c.csv", "id,text\nc1,caf\xe9\n", encoding="latin-1")
    assert "is not UTF-8 text" in exc.value.code


def test_configs_without_a_file_are_untouched():
    for config in (None, [], {"test_cases": [{"id": "x"}]}, {"name": "x"}):
        assert dataset.expand(config, ".") == (config, [])


def test_run_scores_cases_from_a_file(tmp_path, monkeypatch):
    import httpx

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    def post(self, **kwargs):
        text = kwargs["json"]["messages"][-1]["content"]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "pos" if "good" in text else "neg"}}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0},
            },
        )

    monkeypatch.setattr(httpx.Client, "post", post)
    _load(
        tmp_path,
        "c.csv",
        "id,text,expected_output\nc1,good,pos\nc2,bad,neg\nc3,good,neg\n",
        {"file": "c.csv", "evaluation": "exact_match"},
    )
    with pytest.raises(SystemExit) as exc:
        main.main(["run", str(tmp_path / "bench.yaml"), "--no-cache"])
    assert exc.value.code == 0
    [record] = (tmp_path / "results").glob("*.json")
    combo = json.loads(record.read_text())["results"][0]
    assert (combo["success"], combo["total"]) == (2, 3)
