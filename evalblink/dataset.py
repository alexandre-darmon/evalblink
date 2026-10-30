"""Test cases loaded from a CSV, TSV or JSONL file next to the benchmark YAML.

``test_cases: {file: cases.csv, evaluation: exact_match}`` is expanded into the
plain list the rest of the pipeline reads, before validation — the validator,
runner and reporter never see the file form.

A row's test-case fields (:data:`FIELDS`) keep their meaning; every other
column (CSV/TSV) or key (JSONL) becomes a template variable. In CSV, ``tags``
are separated by ``;`` and a ``weighted_match`` ``expected_output`` is JSON.
"""

from __future__ import annotations

import csv
import difflib
import json
import os

FIELDS = (
    "id",
    "evaluation",
    "expected_output",
    "criteria",
    "reference",
    "tags",
    "human_score",
)
SPEC_KEYS = ("file", "evaluation")
FORMATS = {".csv": ",", ".tsv": "\t", ".jsonl": None, ".ndjson": None}


class DatasetError(ValueError):
    """A test-case file that can't be read, with the file and line in the message."""


def expand(config, base_dir: str) -> tuple:
    """``(config, warnings)`` with a ``test_cases: {file: …}`` spec expanded.

    Any other config is returned unchanged. Raises :class:`DatasetError`.
    """
    spec = config.get("test_cases") if isinstance(config, dict) else None
    if not isinstance(spec, dict):
        return config, []
    unknown = [str(k) for k in spec if k not in SPEC_KEYS]
    if unknown:
        raise DatasetError(
            f"test_cases: unknown key {unknown[0]!r} — use 'file' and,"
            " optionally, 'evaluation' (the default for rows without one)"
        )
    name = spec.get("file")
    if not isinstance(name, str) or not name.strip():
        raise DatasetError(
            "test_cases.file must be the path of a .csv, .tsv or .jsonl file"
        )
    ext = os.path.splitext(name)[1].lower()
    if ext not in FORMATS:
        raise DatasetError(
            f"{name}: unsupported file type {ext or '(none)'!r} — use .csv, .tsv or .jsonl"
        )
    path = os.path.join(base_dir, name)
    rows = _jsonl(path, name) if FORMATS[ext] is None else _delimited(path, name, ext)

    default = spec.get("evaluation")
    cases, warnings, seen_columns = [], [], set()
    for line, row in rows:
        case = {k: row.pop(k) for k in FIELDS if k in row}
        variables = row.pop("variables", None)
        if variables is not None and not isinstance(variables, dict):
            raise DatasetError(f"{name}:{line}: 'variables' must be an object")
        variables = {**(variables or {}), **row}
        seen_columns.update(row)
        if variables:
            case["variables"] = variables
        case.setdefault("id", f"line_{line}")
        if default is not None:
            case.setdefault("evaluation", default)
        if case.get("evaluation") == "weighted_match" and isinstance(
            case.get("expected_output"), str
        ):
            try:
                case["expected_output"] = json.loads(case["expected_output"])
            except ValueError as exc:
                raise DatasetError(
                    f"{name}:{line}: a weighted_match expected_output must be JSON ({exc})"
                ) from None
        cases.append(case)
    if not cases:
        raise DatasetError(f"{name} has no test cases")

    for column in sorted(map(str, seen_columns)):
        close = difflib.get_close_matches(column, FIELDS, n=1, cutoff=0.8)
        if close:
            warnings.append(
                f"{name}: column {column!r} is passed as a template variable"
                f" — did you mean {close[0]!r}?"
            )
    return {**config, "test_cases": cases}, warnings


def _open(path, name):
    try:
        # utf-8-sig: Excel's "CSV UTF-8" starts with a byte-order mark.
        return open(path, encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise DatasetError(f"Cannot read {name}: {exc.strerror or exc}") from None


def _delimited(path, name, ext):
    """``[(line, {column: value})]`` from a CSV/TSV file with a header row."""
    try:
        with _open(path, name) as f:
            reader = csv.reader(f, delimiter=FORMATS[ext])
            header = next(reader, None)
            if header is None:
                raise DatasetError(
                    f"{name} is empty — the first row must name the columns"
                )
            header = [h.strip() for h in header]
            if "" in header:
                raise DatasetError(
                    f"{name}:1: column {header.index('') + 1} has no name"
                )
            dupes = sorted({h for h in header if header.count(h) > 1})
            if dupes:
                raise DatasetError(f"{name}:1: duplicate column {dupes[0]!r}")
            rows = []
            for cells in reader:
                line = reader.line_num
                if not any(c.strip() for c in cells):
                    continue  # blank line
                if len(cells) > len(header):
                    raise DatasetError(
                        f"{name}:{line}: {len(cells)} cells but {len(header)} columns"
                        " — quote values that contain the delimiter"
                    )
                row: dict = dict(zip(header, cells))
                # Empty cells mean "not set" for test-case fields; a variable
                # keeps its (possibly empty) value.
                for field in FIELDS:
                    if field in row and not row[field].strip():
                        del row[field]
                if "tags" in row:
                    row["tags"] = [
                        t.strip() for t in row["tags"].split(";") if t.strip()
                    ]
                if "human_score" in row:
                    try:
                        row["human_score"] = int(row["human_score"])
                    except ValueError:
                        raise DatasetError(
                            f"{name}:{line}: human_score must be a whole number"
                        ) from None
                rows.append((line, row))
    except csv.Error as exc:
        raise DatasetError(f"{name}: {exc}") from None
    except UnicodeDecodeError:
        raise DatasetError(f"{name} is not UTF-8 text — save it as UTF-8") from None
    return rows


def _jsonl(path, name):
    """``[(line, object)]`` from a file with one JSON object per line."""
    rows = []
    try:
        with _open(path, name) as f:
            for line, text in enumerate(f, start=1):
                if not text.strip():
                    continue
                try:
                    row = json.loads(text)
                except (ValueError, RecursionError) as exc:
                    raise DatasetError(f"{name}:{line}: invalid JSON ({exc})") from None
                if not isinstance(row, dict):
                    raise DatasetError(
                        f"{name}:{line}: each line must be a JSON object, not"
                        f" {type(row).__name__}"
                    )
                rows.append((line, row))
    except UnicodeDecodeError:
        raise DatasetError(f"{name} is not UTF-8 text — save it as UTF-8") from None
    return rows
