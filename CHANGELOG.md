# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `lerobot-align-fit` accepts the append-only `.jsonl` sidecar written by
  `lerobot-align-eval-batch` wherever it accepted the `--out` JSON array, so a
  scoring run that dies partway no longer has to be re-aggregated by hand.
  Because the sidecar is append-only, a repeated `(episode, format)` is a later
  observation rather than the duplicate-row caller error the array form still
  reports: the last row wins, except that an `error` row never displaces a
  scored one — a re-run against a flaky endpoint would otherwise shrink the
  fitting cohort and blame the model for an episode whose prediction is sitting
  in the same file. Every resolved collision is printed and recorded in the
  calibration under `resolved_collisions`, alongside `source_results_format`.
- `lerobot-align-eval-batch` stamps a per-run `run_id` on every row it writes.
  The sidecar is append-only and carries no other run identity, so without it
  two scoring passes at different `--model`, `--fps` or `--camera` are
  indistinguishable once they share a file.

### Fixed

- `lerobot-align-eval-batch --out <name>.jsonl` derived a sidecar path equal to
  the output file, so the run appended rows to the file it then overwrote with
  the final array. The sidecar now falls back to `<name>.jsonl.rows.jsonl` in
  that collision only; every other `--out` keeps its existing sidecar name.
- `lerobot-align-fit` reported malformed results files with a bare
  `JSONDecodeError`; it now names the file and, for the sidecar, the line, and
  rejects a row with no `episode` key instead of raising `KeyError`.

## [0.1.0] - <RELEASE-DATE>

First public release. There is no earlier release history; entries below
describe the initial contents rather than changes against a predecessor.

### Added

- `lerobot-align`: steerable multimodal annotation, given-subtask alignment and
  boundary calibration for LeRobot v3.x datasets, driven through any
  OpenAI-compatible vision-language endpoint.
- Native video encoding for the subtask generation and alignment stages, using
  the model's own video timeline instead of contact-sheet clock badges.
- Motion-adaptive keyframe sampling and stacked synchronized multiview frames.
- Dynamic-programming boundary calibration with dataset-specific duration
  priors, plus the fitter's acceptance gates.
- Subtask import from SARM metadata, previous `language_persistent` runs, raw
  `task_index` runs, and `meta/lerobot_annotations.json`.
- Diagnostic console scripts: `lerobot-align-fit`, `lerobot-align-eval`,
  `lerobot-align-eval-batch`, `lerobot-motion-viz`.
- Dispatch to Hugging Face Jobs via `--job.target`.
- `evaluation/`: the pre-registered comparison against upstream
  `lerobot-annotate`, its metric library, and the frozen results and reports.
