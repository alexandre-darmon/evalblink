"""No-crash regression guard: hostile configs x hostile API responses x hostile files.

Drives the real CLI (``main.main``) offline — ``httpx.Client.post/get`` are faked
at class level — through ``validate``, ``run``, ``run --dry-run``, ``report``,
``compare``, ``history`` and ``models``. The only acceptable outcome is a clean
exit (``SystemExit`` with an int code or a message); any other exception is a
crash. Every ``run`` of a config that validates must also leave a results JSON
that ``compare.load_record`` reads back.

The corpus covers these input classes (environment, filesystem and signal
paths are in ``test_main``): malformed/odd config values,
non-finite and float-overflowing numbers, 4300+-digit ints, lone UTF-16
surrogates, Rich markup in user/model text, every response-shape / status /
transport failure, tampered cache entries, unwritable caches and malformed
results files. Add a case here whenever a new class turns up.
"""

from __future__ import annotations

import json
import os
import shutil

import httpx
import pytest

from evalblink import compare, main, openrouter

BIG = "1" + "0" * 400  # an int no float can hold
HUGE = "1" * 5000  # past Python's 4300-digit int-conversion limit

BASE = """\
name: NoCrash
models: [vendor/a, other/b]
inference: {temperature: 0, max_tokens: 300}
evaluation:
  judge_model: judge/j
  judge_threshold: 0.5
  quality_threshold: 0.5
  variables:
    - {name: use_case, weight: 0.5}
    - {name: percent, weight: 0.3, tolerance: 0.1}
    - {name: order, weight: 0.2}
prompts:
  - {id: p1, template: "Q: {{ q }}", system: "Be {{ tone }}"}
variables: {tone: brief}
quality_threshold: 50
test_cases:
  - {id: e1, evaluation: exact_match, expected_output: positive, variables: {q: a}, tags: [t1]}
  - {id: w1, evaluation: weighted_match, variables: {q: b}, expected_output: [{use_case: a, percent: 0.5, order: 1}]}
  - {id: j1, evaluation: llm_judge, criteria: good, variables: {q: c}, reference: ref}
"""


def _sub(*pairs):
    text = BASE
    for old, new in pairs:
        assert old in text, old
        text = text.replace(old, new)
    return text


CONFIGS = {
    "base": BASE,
    # Rich markup in every user string that reaches the terminal.
    "markup": _sub(
        ("name: NoCrash", 'name: "[bold]v2 [/x]"'),
        ("other/b", '"x/[/y]"'),
        ("id: p1", 'id: "[/p]"'),
        ("tags: [t1]", 'tags: ["[/t]", ":smile:"]'),
        ("id: e1", 'id: "[red]e1"'),
    ),
    # Lone surrogates (double-quoted YAML escapes) in names, ids, text.
    "surrogates": _sub(
        ("name: NoCrash", 'name: "bench \\ud800"'),
        ("other/b", '"o/\\udfff"'),
        ("id: e1", 'id: "e\\ud800"'),
        ("expected_output: positive", 'expected_output: "pos\\ud800"'),
        ('template: "Q: {{ q }}"', 'template: "Q\\ud800: {{ q }}"'),
        ("tags: [t1]", 'tags: ["t\\udc80"]'),
    ),
    "name_punct": _sub(("name: NoCrash", 'name: "!!!"')),
    "name_long": _sub(("name: NoCrash", "name: " + "x" * 3000)),
    "judge_ref_big": _sub(("reference: ref", f"reference: {BIG}")),
    "judge_expected_big": _sub(
        ("criteria: good", f"criteria: good, expected_output: {BIG}")
    ),
    # Invalid: numbers that overflow a float, in every numeric field.
    "qt_big": _sub(("quality_threshold: 50", f"quality_threshold: {BIG}")),
    "cost_big": BASE + f"max_cost_usd: {BIG}\n",
    "conc_big": BASE + f"concurrency: {BIG}\n",
    "temp_big": _sub(("temperature: 0", f"temperature: {BIG}")),
    "max_tokens_big": _sub(("max_tokens: 300", f"max_tokens: {BIG}")),
    "max_tokens_float": _sub(("max_tokens: 300", "max_tokens: 300.0")),
    "judge_thr_big": _sub(("judge_threshold: 0.5", f"judge_threshold: {BIG}")),
    "case_thr_big": _sub(("  quality_threshold: 0.5", f"  quality_threshold: {BIG}")),
    "weight_big": _sub(("weight: 0.5", f"weight: {BIG}")),
    "percent_big": _sub(("percent: 0.5", f"percent: {BIG}")),
    "huge_int": _sub(("quality_threshold: 50", f"quality_threshold: {HUGE}")),
    "nan_inf": _sub(
        ("temperature: 0", "temperature: .nan"), ("max_tokens: 300", "max_tokens: .inf")
    ),
    # judge_model must be a string even with no llm_judge case.
    "judge_model_int": _sub(("judge_model: judge/j", "judge_model: 5")).split(
        "  - {id: j1"
    )[0],
    "judge_model_list": _sub(("judge_model: judge/j", "judge_model: [a, b]")),
    "markup_in_error": _sub(("id: w1", 'id: "[/x]"'), ("id: j1", 'id: "[/x]"')),
    "surrogate_in_error": _sub(
        ("id: w1", 'id: "\\ud800"'), ("id: j1", 'id: "\\ud800"')
    ),
    "tmpl_undefined": _sub(('template: "Q: {{ q }}"', 'template: "{{ q.a.b }}"')),
    "empty": "",
    "list": "- a\n- b\n",
    "scalar": "42\n",
    "bad_yaml": "name: [unclosed\n",
    "python_tag": "!!python/object/apply:os.system ['echo hi']\n",
    "non_utf8": b"name: caf\xe9\n",
}


def _ok(content, **usage):
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "cost": 0.001, **usage},
    }


def _J(status, body):
    return lambda: httpx.Response(status, json=body)


def _raw(status, text):
    return lambda: httpx.Response(status, content=text.encode())


def _content_raw(content_json):
    """A 200 completion whose content is given as a raw JSON string literal."""
    return _raw(
        200,
        '{"choices":[{"message":{"content":' + content_json + "}}],"
        '"usage":{"prompt_tokens":1,"completion_tokens":1,"cost":0}}',
    )


def _exc(exc):
    return lambda: exc


HTML = "<html><body><h1>502 Bad Gateway</h1></body></html>"
_REQ = httpx.Request("POST", "https://openrouter.ai")

RESPONSES = {
    "ok_label": _J(200, _ok("positive")),
    "ok_list": _J(200, _ok('[{"use_case": "a", "percent": 0.5, "order": 1}]')),
    "ok_judge": _J(200, _ok('```json\n{"score": 4, "reasoning": "r"}\n```')),
    **{
        f"json_{c}": _J(c, {"error": {"code": c, "message": "nope"}})
        for c in (400, 401, 429, 500, 502)
    },
    **{f"html_{c}": _raw(c, HTML) for c in (200, 404, 503)},
    "503_no_error": _J(503, {}),
    "503_completion": _J(503, _ok("positive")),
    "empty_200": _raw(200, ""),
    "binary_200": lambda: httpx.Response(200, content=b"\xff\xfe\x00garbage"),
    "json_list": _J(200, [1, 2]),
    "json_null": _J(200, None),
    "error_str": _J(200, {"error": "broke"}),
    "error_code_str": _J(200, {"error": {"code": "429", "message": "x"}}),
    "error_null_500": _J(500, {"error": None}),
    "error_markup": _J(400, {"error": {"code": 400, "message": "[/x] [bold]oops"}}),
    "error_surrogate": _raw(400, '{"error":{"code":400,"message":"bad \\ud800"}}'),
    "deep_body": _raw(200, "[" * 50000 + "]" * 50000),
    "no_choices": _J(200, {"usage": {}}),
    "choices_str": _J(200, {"choices": "abc"}),
    "message_null": _J(200, {"choices": [{"message": None}]}),
    "null_content": _J(200, _ok(None)),
    "ws_content": _J(200, _ok("  \n")),
    "list_content": _J(200, _ok([{"type": "text"}])),
    "usage_missing": _J(200, {"choices": [{"message": {"content": "x"}}]}),
    "usage_str": _J(200, _ok("x", prompt_tokens="3", cost="0.1")),
    "usage_inf": _raw(
        200,
        '{"choices":[{"message":{"content":"positive"}}],'
        '"usage":{"prompt_tokens":1,"completion_tokens":1,"cost":Infinity}}',
    ),
    "usage_big": _raw(
        200,
        '{"choices":[{"message":{"content":"positive"}}],'
        f'"usage":{{"prompt_tokens":1,"completion_tokens":1,"cost":{BIG}}}}}',
    ),
    "usage_huge": _raw(
        200,
        '{"choices":[{"message":{"content":"positive"}}],'
        f'"usage":{{"prompt_tokens":{HUGE},"completion_tokens":1,"cost":0}}}}',
    ),
    "content_surrogate": _content_raw('"pos\\ud800itive"'),
    "content_markup": _J(200, _ok("[/x] [bold]positive :smile:")),
    "wm_big_percent": _J(
        200, _ok(f'[{{"use_case": "a", "percent": {BIG}, "order": 1}}]')
    ),
    "wm_huge_int": _J(200, _ok("[" + HUGE + "]")),
    "wm_inf": _J(200, _ok('[{"use_case": "a", "percent": 1e400, "order": 1}]')),
    "wm_bad_items": _J(200, _ok('[{"use_case": {"x": 1}, "percent": "a"}]')),
    "wm_deep": _J(200, _ok("[" * 100000 + "]" * 100000)),
    "judge_text": _J(200, _ok("I think it's a 4.")),
    "judge_99": _J(200, _ok('{"score": 99}')),
    "judge_inf": _J(200, _ok('{"score": Infinity}')),
    "judge_nan": _J(200, _ok('{"score": NaN}')),
    "judge_big": _J(200, _ok('{"score": 1e400}')),
    "judge_huge": _J(200, _ok('{"score": ' + HUGE + "}")),
    "judge_str": _J(200, _ok('{"score": "four"}')),
    "judge_list": _J(200, _ok("[4]")),
    "judge_reason_obj": _J(200, _ok('{"score": 5, "reasoning": {"a": [1]}}')),
    "judge_reason_markup": _J(200, _ok('{"score": 5, "reasoning": "[/x]"}')),
    "judge_reason_surrogate": _content_raw(
        '"{\\"score\\": 5, \\"reasoning\\": \\"r\\\\ud800\\"}"'
    ),
    "timeout": _exc(httpx.ReadTimeout("slow")),
    "connect": _exc(httpx.ConnectError("refused")),
    "remote_proto": _exc(httpx.RemoteProtocolError("peer closed")),
    "decoding": _exc(httpx.DecodingError("bad gzip")),
    "redirects": _exc(httpx.TooManyRedirects("loop")),
    "status_error": _exc(
        httpx.HTTPStatusError("x", request=_REQ, response=httpx.Response(500))
    ),
}

# Mixed sequences: one item per call, the last repeats.
SEQUENCES = {
    "mix_ok_then_errors": [
        "ok_label",
        "ok_list",
        "ok_judge",
        "json_500",
        "timeout",
        "content_markup",
    ],
    "mix_retry_then_ok": [
        "503_no_error",
        "ok_label",
        "json_429",
        "ok_list",
        "html_503",
        "ok_judge",
    ],
    "mix_hostile": [
        "content_surrogate",
        "wm_big_percent",
        "judge_reason_markup",
        "usage_big",
        "decoding",
    ],
}

CATALOGS = {
    "ok": _J(
        200,
        {
            "data": [
                {
                    "id": m,
                    "context_length": 8000,
                    "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                }
                for m in ("vendor/a", "other/b", "judge/j")
            ]
        },
    ),
    "negative": _J(
        200,
        {
            "data": [
                {"id": m, "pricing": {"prompt": "-1", "completion": "-1"}}
                for m in ("vendor/a", "other/b", "judge/j")
            ]
        },
    ),
    "weird": _J(
        200,
        {
            "data": [
                {
                    "id": "vendor/a",
                    "context_length": "lots",
                    "pricing": {"prompt": "abc"},
                },
                {"id": "other/b", "context_length": [1], "pricing": {"prompt": "-0.5"}},
                {"id": 5},
                "x",
                {"id": "[/m]", "pricing": None},
            ]
        },
    ),
    "data_str": _J(200, {"data": "abc"}),
    "raw_overflow": _raw(
        200,
        '{"data":[{"id":"vendor/a","context_length":1e400,"pricing":{"prompt":'
        + BIG
        + ',"completion":NaN}},{"id":"other/b","context_length":'
        + HUGE
        + "}]}",
    ),
    "error": _J(401, {"error": {"code": 401, "message": "[/x] bad key"}}),
    "html": _raw(502, HTML),
    "list": _J(200, [1]),
    "connect": _exc(httpx.ConnectError("refused")),
    "redirects": _exc(httpx.TooManyRedirects("loop")),
}

GOOD_RECORD = {
    "schema_version": 1,
    "run_id": "r1_x",
    "benchmark": "[bold]v2 \ud800",
    "timestamp": "2026-01-01_000000",
    "results": [
        {
            "model": "[/m]",
            "prompt_id": "[/p]",
            "score": 50.0,
            "success": 1,
            "scored": 2,
            "errors": 1,
            "total": 3,
            "total_prompt_tokens": 1,
            "total_completion_tokens": 1,
            "total_cost": 0.0,
            "test_cases": [
                {
                    "id": "[/c]",
                    "tags": ["[/t]"],
                    "match_result": True,
                    "match_score": 1.0,
                    "total_cost": 0.0,
                },
                {
                    "id": "c\ud800",
                    "tags": [],
                    "match_result": False,
                    "match_score": 0.0,
                    "total_cost": 0.0,
                },
                {
                    "id": "c3",
                    "match_result": False,
                    "match_score": None,
                    "total_cost": 0.0,
                },
            ],
        }
    ],
}


def _bad_record(**overrides):
    return {**GOOD_RECORD, **overrides}


RESULT_FILES = {
    "good.json": json.dumps(GOOD_RECORD),
    "not_json.json": "not json {{",
    "markdown.json": "# Report\n\n| a | b |\n",
    "list.json": "[1, 2]",
    "empty_obj.json": "{}",
    "non_utf8.json": b"\xff\xfe{",
    "deep.json": "[" * 100000 + "]" * 100000,
    "huge_int.json": '{"benchmark": "b", "timestamp": "t", "results": [], "x": '
    + HUGE
    + "}",
    "cost_null.json": json.dumps(
        _bad_record(results=[{**GOOD_RECORD["results"][0], "total_cost": None}])
    ),
    "score_nan.json": json.dumps(
        _bad_record(results=[{**GOOD_RECORD["results"][0], "score": float("nan")}])
    ),
    "score_big.json": json.dumps(
        _bad_record(results=[{**GOOD_RECORD["results"][0], "score": int(BIG)}])
    ),
    "results_dict.json": json.dumps(_bad_record(results={"a": 1})),
    "combo_str.json": json.dumps(_bad_record(results=["x"])),
    "cases_missing.json": json.dumps(
        _bad_record(
            results=[
                {
                    k: v
                    for k, v in GOOD_RECORD["results"][0].items()
                    if k != "test_cases"
                }
            ]
        )
    ),
    "case_id_int.json": json.dumps(
        _bad_record(
            results=[
                {
                    **GOOD_RECORD["results"][0],
                    "test_cases": [{"id": 1, "total_cost": 0}],
                }
            ]
        )
    ),
    "tags_dict.json": json.dumps(
        _bad_record(
            results=[
                {
                    **GOOD_RECORD["results"][0],
                    "test_cases": [{"id": "c", "tags": {"a": 1}, "total_cost": 0}],
                }
            ]
        )
    ),
    "run_id_int.json": json.dumps(_bad_record(run_id=5)),
    "insights_garbage.json": json.dumps(_bad_record(insights="x")),  # recomputed
    "run_id_path.json": json.dumps(_bad_record(run_id="../../escape")),
    "timestamp_int.json": json.dumps(_bad_record(timestamp=5)),
    "dir.json": None,  # a directory
}


@pytest.fixture
def cli(monkeypatch, tmp_path, capsys):
    """Run ``main.main(argv)`` offline from ``tmp_path``; return the exit code.

    ``serve(*names)`` queues fake responses for ``httpx.Client.post`` / ``get``
    (the last repeats). Any exception other than a clean ``SystemExit`` fails.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(openrouter, "_cache_write_warned", False)
    state = {"seq": [RESPONSES["ok_label"]], "n": 0}

    def fake(self, *a, **k):
        item = state["seq"][min(state["n"], len(state["seq"]) - 1)]()
        state["n"] += 1
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(httpx.Client, "post", fake)
    monkeypatch.setattr(httpx.Client, "get", fake)

    def serve(factories):
        state["seq"], state["n"] = list(factories), 0

    def run(argv, label):
        try:
            main.main(argv)
        except SystemExit as exc:
            code = exc.code
        except BaseException as exc:  # noqa: BLE001 — the point of the test
            pytest.fail(f"{label}: {' '.join(map(str, argv))} crashed: {exc!r}")
        else:
            pytest.fail(f"{label}: returned without exiting")
        capsys.readouterr()  # drop the output; keep memory flat
        assert code is None or isinstance(code, (int, str)), (label, code)
        return code

    run.serve = serve
    return run


def _write_config(tmp_path, label):
    text = CONFIGS[label]
    path = tmp_path / f"{label}.yaml"
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(text, encoding="utf-8", errors="surrogatepass")
    return str(path)


def _run_and_check_record(cli, tmp_path, path, label, extra=()):
    shutil.rmtree(tmp_path / "results", ignore_errors=True)
    cli(["run", path, *extra], label)
    [record] = (tmp_path / "results").glob("*.json")
    compare.load_record(str(record))  # readable, and in the shape we persist
    return str(record)


def _valid(cli, tmp_path, label):
    return cli(["validate", _write_config(tmp_path, label)], f"validate {label}") == 0


def test_validate_and_run_every_config(cli, tmp_path):
    valid = []
    for label in CONFIGS:
        path = _write_config(tmp_path, label)
        code = cli(["validate", path], f"validate {label}")
        if code == 0:
            valid.append(label)
            cli.serve([RESPONSES["ok_label"]])
            _run_and_check_record(cli, tmp_path, path, label, ["--no-cache"])
            for names in SEQUENCES.values():
                cli.serve([RESPONSES[n] for n in names])
                _run_and_check_record(cli, tmp_path, path, label, ["--no-cache"])
        else:
            # An invalid config must be refused before any API call.
            cli.serve([_exc(AssertionError("API called for an invalid config"))])
            assert cli(["run", path], f"run {label}") not in (0, None), label
    # The corpus must keep exercising both sides.
    assert {"base", "markup", "surrogates", "name_punct", "judge_ref_big"} <= set(valid)
    assert "judge_model_int" not in valid and "max_tokens_float" not in valid


def test_run_survives_every_hostile_response(cli, tmp_path):
    path = _write_config(tmp_path, "base")
    for name, factory in RESPONSES.items():
        cli.serve([factory])
        _run_and_check_record(cli, tmp_path, path, name, ["--no-cache"])


def test_run_survives_tampered_and_unwritable_cache(cli, tmp_path):
    path = _write_config(tmp_path, "markup")
    cli.serve([RESPONSES["ok_judge"]])
    _run_and_check_record(cli, tmp_path, path, "fill cache")
    for garbage in (
        '{"response": null}',
        '{"response": "x", "cost": "a"}',
        '{"response": "x", "prompt_tokens": 1, "completion_tokens": 1, "cost": '
        + BIG
        + "}",
        "[" * 100000,
        "\ud800",
    ):
        for entry in (tmp_path / ".evalblink_cache").glob("*.json"):
            if entry.name != "models.json":
                entry.write_text(garbage, encoding="utf-8", errors="surrogatepass")
        _run_and_check_record(cli, tmp_path, path, f"tampered {garbage[:20]!r}")
    shutil.rmtree(tmp_path / ".evalblink_cache")
    (tmp_path / ".evalblink_cache").write_text("a file, not a directory")
    _run_and_check_record(cli, tmp_path, path, "unwritable cache")


def test_dry_run_survives_every_catalog(cli, tmp_path):
    for label in ("base", "markup", "surrogates", "judge_ref_big"):
        path = _write_config(tmp_path, label)
        for name, factory in CATALOGS.items():
            cli.serve([factory])
            code = cli(["run", path, "--dry-run"], f"dry-run {label}/{name}")
            if name == "negative":
                # All pricing unknown: $0, nothing over budget — unless a model
                # isn't in the catalog at all (markup/surrogate ids), which fails.
                assert code == (0 if label in ("base", "judge_ref_big") else 1)
            shutil.rmtree(tmp_path / ".evalblink_cache", ignore_errors=True)


def test_models_survives_every_catalog(cli, tmp_path):
    for name, factory in CATALOGS.items():
        for extra in ([], ["--free"], ["--min-context", "1k"]):
            cli.serve([factory])
            cli(["models", "--no-cache", *extra], f"models {name} {extra}")
    for bad in ("xk", "1e400k", "nan", ""):
        with pytest.raises(SystemExit) as exc:
            main.main(["models", "--min-context", bad])
        assert exc.value.code == 2


def test_report_compare_history_survive_every_results_file(cli, tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    paths = {}
    for name, content in RESULT_FILES.items():
        path = results / name
        if content is None:
            path.mkdir()
        elif isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8", errors="surrogatepass")
        paths[name] = str(path)
    paths["missing.json"] = str(results / "missing.json")
    paths["report.md"] = str(results / "report.md")
    (results / "report.md").write_text("# not json")

    good = paths["good.json"]
    for name, path in paths.items():
        expect_ok = name in ("good.json", "run_id_path.json", "insights_garbage.json")
        assert (cli(["report", path], f"report {name}") == 0) == expect_ok, name
        for argv in (["compare", good, path], ["compare", path, good, "--detailed"]):
            assert (cli(argv, f"compare {name}") == 0) == expect_ok, name
    assert cli(["history"], "history") == 0
    # The traversal run_id wrote its report inside results/, not above it.
    assert not (tmp_path.parent / "escape.md").exists()
    assert os.path.isdir(results)
