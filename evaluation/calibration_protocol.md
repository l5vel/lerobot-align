# Shared alignment evaluation for Corpus A and Corpus B

Both datasets use **one evaluation implementation and one protocol**. Raw source
conversion, source paths, task membership and camera feature names belong to
preprocessing. Fitting, routing, inference, timing baselines, scoring, failure
handling and statistical comparisons are shared.

Allocate **exactly ten distinct annotated trajectories total per task** across
all source components, cameras and segment-count cohorts. Fit once using only
those trajectories, then freeze the parameters for every remaining trajectory.
Failed or unusable seeds are not replaced. Tasks with fewer than eleven selected
trajectories are rejected, because they cannot supply ten seeds plus evaluation.

## Run either dataset

Use the same commands, changing only the source manifest and output directories:

```bash
CORPUS=corpus_a  # or corpus_b
.venv/bin/python evaluation/scripts/prepare_alignment_study.py \
  --manifest "evaluation/configs/${CORPUS}_sources.json" \
  --out "/path/to/prepared/${CORPUS}" --dry-run

# Remove --dry-run to create the evaluation datasets. This copies/splits media.
.venv/bin/python evaluation/scripts/prepare_alignment_study.py \
  --manifest "evaluation/configs/${CORPUS}_sources.json" \
  --out "/path/to/prepared/${CORPUS}"

.venv/bin/python evaluation/scripts/evaluate_alignment_study.py \
  --study "/path/to/prepared/${CORPUS}" --out "/path/to/runs/${CORPUS}" \
  --model "$MODEL_ID" --ports 8001,8002,8003,8004,8005,8006
```

Run in a shell with read access to your prepared input directories. Set the
`ALIGN_CORPUS_A_ROOT` / `ALIGN_CORPUS_A_GT` or `ALIGN_CORPUS_B_ROOT` /
`ALIGN_CORPUS_B_GT` environment variables used by the selected manifest.
The same model must already be serving on the supplied ports. This driver does
not start servers or select GPUs.
`--dry-run` on the evaluator prints the shared commands without launching jobs.
`--stage fit|run|score|aggregate` supports stage-by-stage execution; completed
fits cannot be overwritten, so resume later stages instead of refitting.
Interrupted preparation requires a fresh directory.

The bundled source manifests reference environment-configured prepared LeRobot
v3 datasets. Corpus A explicitly groups its 16 source components into four task
families: clean-table, mobile-door, bag-place and fridge-drink. This is the task
unit for both its ten-seed budget and its bootstrap, rather than assuming each
folder is a separate task. Corpus B uses the 39 named tasks in its selected
study root. Task definitions are an explicit study-design input; change the
manifest before preparation if a different task taxonomy is intended.

Metadata-only validation against the real NFS sources found:

| Corpus | Tasks | Allocated seeds | Remaining evaluation trajectories | Evaluation components |
|---|---:|---:|---:|---:|
| A | 4 | 40 | 757 | 36 |
| B | 39 | 390 | 1,305 | 141 |

Components partition evaluation by task, segment count and physical source;
they do **not** get extra seeds or become bootstrap units. These are preparation
counts, not new performance measurements. Historical A/B scores use different
splits or policies and cannot stand in for the new evaluation. Both datasets
need fresh fits and inference to compare performance under this protocol.

## Shared settings and evaluation contract

`configs/alignment_protocol_arms.yaml` defines six VLM arms and three timing
baselines. The preparer resolves its two camera roles into `arms.yaml`. Both
corpora use temperature 0.2, 2 sampled frames/second, and caps of 60/300 frames
for contact sheets and 300 for video. The shared configuration carries the
1.5-second minimum-duration instruction used by generated-label prompts, but
this fixed-label study supplies the labels and the alignment/calibration solver
does not enforce that value; it optimizes boundaries on the 0.1-second grid
described below.
Native dataset FPS and camera resolution are preserved during preparation;
common inference sampling and resizing are applied afterwards. Splitting a
shared video shard may re-encode selected episodes with LeRobot's splitter.

All arms receive the same per-episode ordered label texts, with no boundary
timestamps in the supplied label file and no modal/default label fallback.
Count pooling is valid only when positions represent comparable task phases;
sharing code does not establish that assumption for either dataset.

Both calibrated camera arms use the same ten seed identities but their own raw
predictions and fitted parameters. The shared workflow uses count scope, at
least three usable seeds per count, and the recommendation gate (3 percentage
points leave-one-out tIoU gain and 1.5s initial first-boundary MAE). It does not
force rejected fits. Unsupported counts and rejected fits use the exact matching
uncalibrated configuration.

`align_floors.py` uses those same ten seeds and matching-count pooling for the
median-relative-boundary and cumulative-duration-prior baselines. Counts with
fewer than three usable seeds receive uniform timing. The uniform baseline is
anchored at the annotated episode start. Episode extent is supplied to these
controls; their endpoint advantage must be considered when interpreting tIoU.

Every expected task/count/source/arm job is declared before dispatch. Missing
job files are errors; explicit failures remain in the scoring population.
`score_alignment.py` computes identical metrics for all nine arms.
`aggregate.py` uses task clusters, paired comparisons, and the same eight
contrasts × three primary metrics with Holm correction from
`configs/alignment_protocol_contrasts.yaml`. The headline includes all scored
trajectories, including failed outputs. A second aggregate uses the same
seed-count eligibility mask for every arm, including baselines; it never filters
on actual calibration application. Four versus 39 task clusters still implies
different statistical precision even with an identical protocol.

## Source manifest and frozen provenance

A minimal manifest for one task is:

```json
{
  "version": 1,
  "name": "my_corpus",
  "cameras": {"primary": "observation.images.wrist", "secondary": "observation.images.external"},
  "sources": [
    {"id": "part1", "root": "/data/part1", "ground_truth": "/data/part1-gt.json", "task": "task1"},
    {"id": "part2", "root": "/data/part2", "ground_truth": "/data/part2-gt.json", "task": "task1"}
  ]
}
```

Both parts share ten seeds in total. Each source can instead declare
`"tasks": {"0": "task1", "1": "task2"}` for a mixed-task root. `episodes` can
explicitly select available local IDs; otherwise all exported GT IDs are used.
Exported GT has `{"episodes": {"0": [{"text": "...", "start": 0, "end": 1}]}}`.
It must agree with the source's `meta/lerobot_annotations.json` sidecar, which
seed inference reads. The preparer verifies cameras, version, labels, duplicate
physical trajectories, task membership and seed/evaluation separation.

The default for **both bundled corpora** selects ten seeds per task by sorting
SHA-256 of `1729:task:source_id:local_episode`. Selection uses identities, not
annotations or prediction quality. Optional `seed_episodes` lists support an
already fixed allocation; if used, every source must declare a list (possibly
empty), and the combined allocation must still be exactly ten per task.

The preparer assigns global study IDs with an explicit source/local/task
registry, constructs a seed-only fitting sidecar, and physically splits all
remaining episodes into evaluation-only components. Source-local seed batches
are translated back into global IDs before one pooled task/count fit. Floor
inputs, merged VLM predictions and scoring all use those same global IDs.
The historical field name `original_index` in component maps means a global
study ID here; recover the native source ID through `preparation_plan.json`.

`study.json` hashes preparation, GT, splits, identity maps, seed sources and
resolved arm settings. The evaluator rejects modified prepared artifacts or
changed source GT/annotation/info metadata. `run_manifest.json` pins the study,
model, protocol, arm configuration, contrasts and bootstrap count. Calibration
manifests additionally record source mappings, inference settings, retained seed
IDs and content-addressed fit/prediction files. These checks pin metadata and
parameters; they do not hash every source video byte or a remote model's weights.

Runtime events record actual calibration application and skip reasons.
`calibration_eligible` denotes shared seed-count support; it is distinct from an
accepted fit and from `calibration_applied`. Failures remain visible in both the
scored population and the application report.

## What is fitted

For complete seed prediction `i`, let `T_i` be its predicted episode duration,
`p_ij` its internal boundary starts, and `g_ij` the corresponding annotated
starts. The fitter estimates positional offsets

`offset_j = median_i((p_ij - g_ij) / T_i)`.

It also estimates median annotated segment durations divided by `T_i`, and
pooled residual scales. At inference, boundary targets are shifted by
`offset_j * T`. A dynamic program balances those targets against segment-duration
priors on a 0.1-second grid. Its weight is selected from
`{0, 0.25, 0.5, 1, 2, 4}` using leave-one-out scores on the retained seeds only.
The resulting prior fractions are not renormalized by the calibrator; the
model-free cumulative-prior floor does normalize them. Final runtime frame
snapping is not reproduced by the seed diagnostic.

`--label-scope exact` retains the most frequent exact ordered label tuple.
`--label-scope segment_count` instead retains differently worded examples with
the same count, within the caller's task. It assumes corresponding positions
represent comparable task phases. It rejects mixed counts within a fit instead
of silently choosing one. Generated-label realignment still requires an exact
label schema; count scope is for supplied ordered labels.

Incomplete predictions, wrongly ordered labels, duplicate rows, failed calls,
invalid durations, and nonfinite timestamps are excluded and recorded. The task
workflow requires at least three usable seeds per count. Other counts and
rejected fits use the matching uncalibrated arm. This can leave much of a diverse
task uncalibrated despite a full ten-trajectory allocation.

The recommendation gate uses seed leave-one-out gain (default at least three
tIoU percentage points) and initial first-boundary MAE (default at least 1.5s).
This gain is optimistic because the same folds select the weight; it is not an
independent performance estimate. `--force` explicitly overrides that gate,
but never the seed budget, scope, or minimum usable count. Choose this policy
before examining evaluation results.

## Shared modules and historical compatibility

- `prepare_alignment_study.py`: source normalization and shared task allocation.
- `evaluate_alignment_study.py`: the common entry point for both corpora.
- `fit_task_calibration.py`: seed-only inference, pooling and frozen fits.
- `make_alignment_labels.py`, `run_alignment_arms.py`: supplied labels and routed inference.
- `merge_alignment_predictions.py`: identity translation and task-level merge.
- `align_floors.py`, `score_alignment.py`, `aggregate.py`: shared controls and analysis.

The old `fit_corpus_b_calibration.py`, `run_corpus_b_arms.py`,
`merge_corpus_b_predictions.py` and `make_corpus_b_oracle_labels.py` filenames are
thin compatibility wrappers around these shared implementations. Their CLI
naming defaults preserve historical B commands; new studies use the common
entry point. Raw Corpus B conversion/selection scripts remain preprocessing.
The historical `run_all.sh` alignment stages require `ALIGN_LEGACY_PROTOCOL=1`
for explicit reproduction and are not the default protocol for new studies.
Historical offline ablation/diagnostic scripts analyze their saved runs and are
not alternative production evaluation pipelines.
