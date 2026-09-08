# Contributing

Thanks for looking at `lerobot-align`. This document covers the mechanics:
how to get a working development environment, what to run before opening a
pull request, and which parts of the tree are deliberately frozen.

## Development install

The package is not on PyPI; run these commands from a source checkout or
extracted source distribution, with Python 3.12, uv and FFmpeg available.

```bash
uv sync --locked --extra dev
```

The `dev` extra pulls in `pytest`, `pytest-cov`, `ruff`, the serving extra and
`matplotlib`, so it covers everything the checks below need.

**FFmpeg must be on `PATH`.** Without it the video fixtures in
`tests/test_frames.py` skip rather than fail, which quietly reduces coverage of
exactly the code path most likely to break.

## Checks before a pull request

These are the same steps `.github/workflows/ci.yml` runs, in the same order.
Run them from the repository root.

```bash
# Unit tests
uv run --no-sync pytest -q

# End-to-end smoke run, both invocation forms CI exercises
uv run --no-sync python -m tests.run_e2e_smoke
uv run --no-sync python tests/run_e2e_smoke.py

# Lint
uv run --no-sync ruff check src tests evaluation

# Shell syntax for the reference host scripts
for script in scripts/*.sh evaluation/scripts/*.sh; do bash -n "$script"; done

# Build the wheel and sdist
uv build
bash scripts/check_distributions.sh
```

CI installs the wheel and extracted sdist into separate fresh environments,
runs both test suites and the visual HTTP smoke test in each, and checks CLI
entry points without importing the checkout. It also opens the built wheel and asserts that `lerobot_align/jobs.py`
and the prompt files are packaged and that the `lerobot-align` console script
is registered. If you move a prompt file or rename an entry point, expect that
step to catch it.

Under `uv`, prefix the Python commands with `uv run` (`uv run pytest -q`), or
call the virtualenv's interpreter directly (`.venv/bin/python -m pytest -q`).

### The evaluation harness suite

`testpaths = ["tests"]` in `pyproject.toml` limits a bare `pytest -q` to the
fast unit lane, so the 104 harness tests under `evaluation/` are not collected
by it — and CI runs them in a separate step. Run them
explicitly whenever you touch `evaluation/metrics/` or `evaluation/scripts/`:

```bash
uv run --no-sync pytest -q evaluation
```

The metric tests assert properties the study relies on — that over-segmentation is
penalised, that order-crossing matches are rejected, that error metrics have
the right polarity, that the semantic matcher does not treat antonyms as
equivalent. Treat a failure there as a result-invalidating bug, not a flaky
test.

## `evaluation/` holds frozen research artifacts

The development archive's `evaluation/` is a record of a pre-registered study, not a test fixture
directory. The plan, arm definitions, scores, reports and reviews describe runs
that were executed once, on specific hardware, against specific dataset
revisions.

The public source export contains the harness and protocol but excludes those
historical artifacts and internal notes. See [release/README.md](release/README.md)
for the committed-file export boundary, and
[DEPENDENCY_SECURITY.md](DEPENDENCY_SECURITY.md) for dependency advisory scope
and the regression checks that guard it.

- Do not regenerate results casually. Re-running a stage overwrites artifacts
  that published claims cite, and inference is stochastic — the report itself
  records a repeated seed fit that selected a different set of cohorts.
- Do not edit a report to match new numbers. Deviations from the
  pre-registration are recorded in `evaluation/analysis/deviations.md`; that
  is where a change of protocol belongs.
- Corrections to the metric *code* are welcome, and should come with a test in
  `evaluation/metrics/` that fails before the fix.

## Style

- `ruff` with the repository's configuration checks Python lint; `line-length = 100`, target `py312`.
- Prose in this repository is plain and specific, and states limitations
  rather than working around them. Match that in docstrings, comments and
  documentation.
- Comments should explain why something is the way it is, especially where the
  reason is a silent failure mode in a dependency. Several of the longest
  comments in this tree exist because the alternative was a wrong result that
  looked fine.
