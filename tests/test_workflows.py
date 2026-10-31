"""The release path in .github/workflows stays the one CI exercises.

publish.yml only runs on a tag, so a broken artifact hand-off would first
show up mid-release. CI calls the same build workflow and downloads its
artifact the same way; these checks keep it that way, and keep every
publish behind the release gates.
"""

from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parent.parent / ".github" / "workflows"
BUILD = "./.github/workflows/build.yml"
PUBLISH = "pypa/gh-action-pypi-publish"

# The sdist ships the tests but not .github/.
pytestmark = pytest.mark.skipif(
    not WORKFLOWS.is_dir(), reason="no .github/workflows (running from the sdist)"
)


def _load(name):
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def _all():
    return {f.name: _load(f.name) for f in sorted(WORKFLOWS.glob("*.y*ml"))}


def _steps(workflow, action):
    """``(job_id, step)`` for every step of ``workflow`` using ``action``."""
    return [
        (job_id, step)
        for job_id, job in workflow["jobs"].items()
        for step in job.get("steps", [])
        if step.get("uses", "").split("@")[0] == action
    ]


def _needs(job):
    needs = job.get("needs", [])
    return {needs} if isinstance(needs, str) else set(needs)


def _uploaded():
    uploads = _steps(_load("build.yml"), "actions/upload-artifact")
    assert len(uploads) == 1, "build.yml must upload exactly one artifact"
    step = uploads[0][1]["with"]
    assert step.get("if-no-files-found") == "error"
    return step["name"], step["path"]


def test_build_workflow_is_reusable_and_builds_both_dists():
    build = _load("build.yml")
    # PyYAML reads a bare `on:` key as True; a quoted "on" stays a string.
    on = build.get(True, build.get("on"))
    assert ({on} if isinstance(on, str) else set(on)) == {"workflow_call"}
    runs = [
        s.get("run", "") for j in build["jobs"].values() for s in j.get("steps", [])
    ]
    assert "python -m build" in runs  # wheel and sdist
    assert any(r.startswith("twine check --strict") for r in runs)


def test_only_the_shared_workflow_builds_and_both_callers_use_it():
    for name, workflow in _all().items():
        if name != "build.yml":
            assert not _steps(workflow, "actions/upload-artifact"), name
    for name in ("ci.yml", "publish.yml"):
        assert _load(name)["jobs"]["build"].get("uses") == BUILD, name


def test_every_download_matches_the_uploaded_artifact():
    name, path = _uploaded()
    downloads = {
        workflow: _steps(_load(workflow), "actions/download-artifact")
        for workflow in ("ci.yml", "publish.yml")
    }
    # CI's package job, and both publish jobs.
    assert [j for j, _ in downloads["ci.yml"]] == ["package"]
    assert sorted(j for j, _ in downloads["publish.yml"]) == [
        "publish-pypi",
        "publish-testpypi",
    ]
    for workflow, steps in downloads.items():
        jobs = _load(workflow)["jobs"]
        for job_id, step in steps:
            assert step["with"] == {"name": name, "path": path}, (workflow, job_id)
            assert "build" in _needs(jobs[job_id]), (workflow, job_id)


def test_every_publish_waits_for_the_release_gates():
    """A release only uploads the shared build, after lint, tests and tag checks."""
    publishers = [
        (name, job_id)
        for name, workflow in _all().items()
        for job_id, _ in _steps(workflow, PUBLISH)
    ]
    assert sorted(publishers) == [
        ("publish.yml", "publish-pypi"),
        ("publish.yml", "publish-testpypi"),
    ]
    jobs = _load("publish.yml")["jobs"]
    assert "checks" in _needs(jobs["build"])
    runs = [s.get("run", "") for s in jobs["checks"]["steps"]]
    for gate in ("ruff check .", "ruff format --check .", "pytest -q"):
        assert gate in runs, gate
    gates = " ".join(runs)
    assert "merge-base --is-ancestor" in gates  # the tag is on main
    assert "__version__" in gates  # the tag matches the package version
    # PyPI on a v* tag push only; TestPyPI on a manual run only.
    assert jobs["publish-pypi"]["environment"] == "pypi"
    assert jobs["publish-pypi"]["if"] == (
        "github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')"
    )
    assert jobs["publish-testpypi"]["environment"] == "testpypi"
    assert jobs["publish-testpypi"]["if"] == "github.event_name == 'workflow_dispatch'"


def test_each_action_has_one_version_across_workflows():
    """CI must run the same action versions the release does."""
    versions: dict = {}
    for workflow in _all().values():
        for job in workflow["jobs"].values():
            for step in job.get("steps", []):
                action, _, ref = step.get("uses", "").partition("@")
                if ref:
                    versions.setdefault(action, set()).add(ref)
    assert {a: v for a, v in versions.items() if len(v) > 1} == {}
