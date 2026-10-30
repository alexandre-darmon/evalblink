# Contributing

```bash
git clone https://github.com/alexandre-darmon/evalblink && cd evalblink
pip install -e ".[dev]"
pytest -q                         # offline, ~6 s: no API key, no network
ruff check . && ruff format .
```

- **Tests stay offline.** `tests/conftest.py` fails any real HTTP request; use the `client_factory` fake or a fake `httpx.Client.post`. The one network test (`tests/test_models_live.py`) only runs with `EVALBLINK_NETWORK_TESTS=1`, weekly in CI.
- **Docs are tested.** `tests/test_docs_drift.py` checks that the YAML examples in README.md and PRD.md validate and that documented flags exist. Update the docs with the code.
- **Architecture.** CLAUDE.md describes each module's job and the config schema the code actually reads. `openrouter.py` is the only module that calls the API.
- **Changes.** One topic per pull request, with a test for each behaviour change and an entry under "Unreleased" in CHANGELOG.md.

## Releasing

1. Bump `__version__` in `evalblink/__init__.py` and date the CHANGELOG section.
2. Merge to `main` with CI green.
3. `git tag -a vX.Y.Z -m "evalblink X.Y.Z" && git push origin vX.Y.Z`. The publish workflow checks the tag against the version and that it is on `main`, then waits for approval of the `pypi` environment.
