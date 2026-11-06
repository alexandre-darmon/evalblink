"""Drift guards: docs, shipped benchmarks and the CLI must agree with the code.

Offline. Catches a README/PRD example that no longer validates, a shipped
benchmark that breaks, or a flag documented in the CLI reference that the
parser doesn't have.
"""

from __future__ import annotations

import argparse
import re
import shlex
import xml.etree.ElementTree as ET
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
    # load_config: a benchmark may load its test cases from a file.
    errors, _ = validate(main.load_config(str(path)))
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


# ── README visuals ────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not (ROOT / "docs" / "assets").is_dir(), reason="the sdist ships no docs/assets"
)
def test_readme_visuals_exist_and_are_self_contained():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    images = re.findall(r"!\[([^\]]*)\]\(([^)]+)\)", readme)
    base = (
        "https://raw.githubusercontent.com/alexandre-darmon/evalblink/main/docs/assets/"
    )
    assert len(images) == 4
    for alt, url in images:
        assert alt, url
        # Absolute: PyPI shows the README without the repo around it.
        assert url.startswith(base), url
        svg = (ROOT / "docs" / "assets" / url.removeprefix(base)).read_text("utf-8")
        ET.fromstring(svg)  # well-formed
        # An SVG in <img> can't load anything, so nothing may depend on it.
        assert not re.search(
            r"<script|<text|<image|<style|https?://(?!www\.w3\.org)", svg
        )
    # evalblink deliberately doesn't track latency; the banner once claimed it.
    assert "latency" not in (ROOT / "docs" / "assets" / "banner.svg").read_text("utf-8")
