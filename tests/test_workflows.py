"""The release path in .github/workflows stays the one CI exercises.

publish.yml only runs on a tag, so a broken artifact hand-off would first
show up mid-release. CI calls the same build workflow and downloads its
artifact the same way; these checks keep it that way.
"""

from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parent.parent / ".github" / "workflows"
BUILD = "./.github/workflows/build.yml"

# The sdist ships the tests but not .github/.
pytestmark = pytest.mark.skipif(
    not WORKFLOWS.is_dir(), reason="no .github/workflows (running from the sdist)"
)


def _load(name):
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def _steps(workflow, action):
    """``(job_id, step)`` for every step of ``workflow`` using ``action``."""
    return [
        (job_id, step)
        for job_id, job in workflow["jobs"].items()
        for step in job.get("steps", [])
        if step.get("uses", "").split("@")[0] == action
    ]


def _uploaded():
    uploads = _steps(_load("build.yml"), "actions/upload-artifact")
    assert len(uploads) == 1, "build.yml must upload exactly one artifact"
    step = uploads[0][1]["with"]
    assert step.get("if-no-files-found") == "error"
    return step["name"], step["path"]


def test_build_workflow_is_reusable_and_builds_both_dists():
    build = _load("build.yml")
    # PyYAML reads the bare `on:` key as True.
    assert set(build[True]) == {"workflow_call"}
    runs = [s.get("run", "") for j in build["jobs"].values() for s in j["steps"]]
    assert "python -m build" in runs  # wheel and sdist
    assert any(r.startswith("twine check --strict") for r in runs)


def test_ci_and_publish_both_build_with_the_shared_workflow():
    for name in ("ci.yml", "publish.yml"):
        jobs = _load(name)["jobs"]
        assert jobs["build"].get("uses") == BUILD, name
        # Only the shared workflow builds or uploads a dist.
        assert not _steps(_load(name), "actions/upload-artifact"), name


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
            needs = jobs[job_id]["needs"]
            assert "build" in ([needs] if isinstance(needs, str) else needs)


def test_each_action_has_one_version_across_workflows():
    """CI must run the same action versions the release does."""
    versions: dict = {}
    for f in WORKFLOWS.glob("*.yml"):
        for job in yaml.safe_load(f.read_text(encoding="utf-8"))["jobs"].values():
            for step in job.get("steps", []):
                action, _, ref = step.get("uses", "").partition("@")
                if ref:
                    versions.setdefault(action, set()).add(ref)
    assert {a: v for a, v in versions.items() if len(v) > 1} == {}
