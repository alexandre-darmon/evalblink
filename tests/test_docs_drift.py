"""Drift guards: docs, shipped benchmarks and the CLI must agree with the code.

Offline. Catches a README/PRD example that no longer validates, a shipped
benchmark that breaks, or a flag documented in the CLI reference that the
parser doesn't have.
"""

from __future__ import annotations

import argparse
import re
import shlex
from pathlib import Path

import pytest
import yaml

from evalblink import main
from evalblink.validator import validate

ROOT = Path(__file__).resolve().parent.parent
DOCS = ["README.md", "PRD.md"]
BENCHMARKS = sorted((ROOT / "benchmarks").glob("*.yaml"))


def _yaml_blocks(doc):
    text = (ROOT / doc).read_text()
    return re.findall(r"```yaml\n(.*?)```", text, flags=re.DOTALL)


def _complete_configs():
    """Every doc YAML block that is a whole benchmark config (not a fragment)."""
    for doc in DOCS:
        for i, block in enumerate(_yaml_blocks(doc)):
            data = yaml.safe_load(block)
            if (
                isinstance(data, dict)
                and {"name", "models", "prompts", "test_cases"} <= data.keys()
            ):
                yield pytest.param(data, id=f"{doc}#{i}")


@pytest.mark.parametrize("path", BENCHMARKS, ids=lambda p: p.name)
def test_shipped_benchmarks_validate(path):
    errors, _ = validate(yaml.safe_load(path.read_text()))
    assert errors == []


def test_docs_contain_complete_examples():
    # Guards the guard: if the regex stops matching, the test below goes vacuous.
    assert len(list(_complete_configs())) >= 4


@pytest.mark.parametrize("config", list(_complete_configs()))
def test_doc_examples_validate(config):
    errors, _ = validate(config)
    assert errors == []


def _subcommands(parser):
    """{"run": subparser, "cache clear": subparser, ...} including nested ones."""
    out = {}
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                out[name] = sub
                out.update({f"{name} {k}": v for k, v in _subcommands(sub).items()})
    return out


SUBCOMMANDS = _subcommands(main.build_parser())


@pytest.mark.parametrize("name", sorted(SUBCOMMANDS))
def test_every_subcommand_has_help(name, capsys):
    with pytest.raises(SystemExit) as exc:
        main.build_parser().parse_args([*name.split(), "--help"])
    assert exc.value.code == 0


def _documented_invocations():
    """`evalblink ...` lines from the README CLI reference block."""
    text = (ROOT / "README.md").read_text()
    section = text.split("## CLI Reference", 1)[1]
    block = re.search(r"```bash\n(.*?)```", section, flags=re.DOTALL).group(1)
    for line in block.splitlines():
        line = line.split("#", 1)[0].strip()
        if line.startswith("evalblink "):
            yield shlex.split(line)[1:]


def test_cli_reference_is_parsed():
    assert len(list(_documented_invocations())) >= 10


@pytest.mark.parametrize("argv", list(_documented_invocations()), ids=" ".join)
def test_documented_flags_exist(argv):
    words = [w for w in argv if not w.startswith("-")]
    # Longest documented subcommand prefix, e.g. "cache clear" before "cache".
    name = next(
        " ".join(words[:n]) for n in (2, 1) if " ".join(words[:n]) in SUBCOMMANDS
    )
    known = {s for a in SUBCOMMANDS[name]._actions for s in a.option_strings}
    flags = {w.split("=", 1)[0] for w in argv if w.startswith("-")}
    assert flags <= known, f"`evalblink {' '.join(argv)}`: unknown {flags - known}"
