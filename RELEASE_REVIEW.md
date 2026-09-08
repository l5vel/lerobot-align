# Release readiness after implementation and integration checks

Reviewed 2026-09-08. **Recommendation: Ready with minor fixes for the scoped
Linux source/package release, using the clean export only.** The historical
study is still not independently reproducible, and hosted Jobs/Hub service
qualification remains outstanding. Those limitations must remain explicit.
Do not publish the original development repository's history.

Implementation reviewed: `b57ac71`. The six original user modifications remain
preserved in `ebdfcfe`. Follow-up work is separated into visual/Jobs runtime
qualification, dependency boundaries, source export and the additional Jobs
publication fix. Metadata/documentation follow-up `40351cb` verifies and pins
all historical source revisions; application code is unchanged. Unrelated local
notes are excluded from the release manifest.
Nothing has been pushed, uploaded or published.

## 1. Release blockers

### B1/B2 — Unsafe dataset publication and replacement of source licensing: resolved

[publication.py](src/lerobot_align/publication.py) and
[cli.py](src/lerobot_align/cli.py) publish a private snapshot built from an
explicit canonical-file manifest. Staging, caches, `.env`, symlinks and unrelated
files cannot enter that manifest. Source cards, attribution and source licensing
are preserved; missing or conflicting licenses fail rather than inheriting the
software license. [Publication tests](tests/test_publication.py) cover these cases.

The final review found a second path: [jobs.py](src/lerobot_align/jobs.py) called
LeRobot's helper that automatically uploaded an entire local dataset cache.
Jobs now require an existing accessible Hub dataset and fail before provisioning
when it is missing. They never invoke that helper. Both new Jobs regression
cases failed before the fix: one opened the sensitive local cache for upload;
the other lost the resolved visual transport settings. All Jobs tests pass now.

### B3/R2 — Uncleared source data and historical Git objects: release boundary resolved

The two removed archives contained 299 and 340 members respectively: 40/56
source-annotation files, 216/228 generated or mixed files, 39/55 selection files
and 4/1 software files. The exact path, size, classification and hash inventory
is retained in [release_artifact_inventory.json](evaluation/release_artifact_inventory.json).
No source labels or timestamp arrays were restored. Nine additional structured
artifacts remain redacted in the development archive. No redistribution rights
were inferred; [NOTICE](NOTICE) preserves separate software and dataset terms.

[export_release.py](scripts/export_release.py) reads only the 189 exact regular
files named by [source-files.txt](release/source-files.txt) from a selected Git
commit. It ignores working-tree changes and unlisted files, copies no Git objects,
stages the complete export, and records SHA-256 hashes. Historical results,
analysis documents and engineering notes are excluded. Regression tests cover
old deleted source material, tracked/untracked private files, symlinks, traversal,
existing destinations and interrupted object reads.

**Required publication action:** use the prepared clean repository or built
source distribution. Never push this development branch's history. The clean
repository must have one new root commit and no remote to the development tree;
[release instructions](release/README.md) make that boundary repeatable.

### B4/B5/B6 — Integrity, transactions and executable onboarding: resolved

Visual loading failures and invalid timestamps/configuration are rejected;
they cannot become camera-less or text-only annotations. Camera-less data is
distinguished from a failed visual provider.
The writer stages and validates all transformed shards, retains checksummed
originals plus a durable journal, publishes metadata last, and recovers interrupted
commits before reading. Failure injection covers exceptions and hard process death
between shard and metadata replacement. See [integrity tests](tests/test_annotation_integrity.py)
and [transaction tests](tests/test_transaction.py).

The canonical [README](README.md) workflow was executed from an allowlisted
export in a fresh environment. Real GPU qualification found and fixed native-video
HTTP 400 failures: Qwen2.5's processor needs timing metadata when vLLM supplies
decoded arrays. The client now sends actual encoded FPS/frame indices when
`video_metadata_source=client` and preserves preselected frames. Qwen3 adapters
use `server` mode to avoid duplicate metadata arguments. The actual upstream
processor regression checks frame timestamps, not just request dictionary shape.
Evaluation fingerprints include this mode. Strict visual fallback remains enabled
in qualification tests; no failed visual input was replaced with text-only inference.

### R1 — Dependency advisories: scoped non-applicability substantiated

The final unfiltered audit reports **six findings across five packages**, including
one duplicate: Accelerate 1.14.0, datasets 4.8.5, setuptools 81.0.0, torch 2.11.0
and Transformers 5.5.4. The audit remains red. These releases were not represented
as patched, incompatible upgrades were not forced, and no advisory was ignored.

[DEPENDENCY_SECURITY.md](DEPENDENCY_SECURITY.md) records each affected operation,
upstream reference, installed constraints and supported workflow disposition.
Fourteen test-only tripwires cover all exported Accelerate checkpoint-loader
aliases, folder dataset builders, tokenizer/processor template saves and
TorchScript compilation. Positive controls exercise every guard. Real sharded
model loading through the embedding evaluator and Transformers ModelManager,
and both complete visual CLI workflows, reach none of the affected operations.
Optional vLLM/Transformers server source paths were also inspected.

This assessment covers the tested Linux annotation/evaluation and documented
serving workflows. Custom Python models, remote-code execution, training, model
export, folder builders, macOS builds and different dependency resolutions require
a new assessment. Compatible upstream patches remain preferable when available.

## 2. Important improvements still outstanding

### I8 — Public historical reproduction remains incomplete

Locations: [Corpus A manifest](evaluation/configs/corpus_a_sources.json),
[Corpus B manifest](evaluation/configs/corpus_b_sources.json), and
[reproduction guide](evaluation/REPRODUCING.md).

Local shared-storage inspection recovered all 16 Corpus A repository IDs and
immutable revisions. Every cached annotation file matches its historical working
copy, and every normalized source ground-truth export matches the historical
counterpart. These datasets contain 799 episodes, of which 797 have annotations;
historical valid study selections have their own exclusions and counts.

Corpus B's 168 local download records all identify one immutable revision.
The original Parquet annotations reconstruct all 1,695 selected historical
ground-truth episodes exactly; source metadata also matches their episode lengths.
Evidence retained locally contains identifiers, index mappings and hashes, never
copied source annotation text.

With explicit lookup approval, anonymous Hub metadata requests on 2026-09-08
returned HTTP 200 and the exact requested commit for all **17 sources**. Every
response reported `private=false` and `gated=false`. The manifests now include
repository IDs, immutable revisions, verification observations and local matching
evidence; Corpus A also includes the original sidecar checksums. No dataset
content was uploaded or downloaded by this metadata check. The public Corpus B
card was read separately to reconcile its access-form declaration with the API;
NOTICE now describes both observations without inferring redistribution rights.

**Remaining action:** obtain any required author permissions and run preparation
from newly downloaded snapshots. Saved historical predictions and fits are also
still required; hashes do not reconstruct them, and fresh inference is a new
experiment. The release guide and README state these limits rather than claiming
full reproduction. Missing external source identifiers are resolved; independent
reconstruction of the historical experiment remains incomplete.

### Hosted Jobs and Hub publication still need service-level qualification

Locations: [jobs.py](src/lerobot_align/jobs.py),
[Jobs constraints](src/lerobot_align/jobs-constraints.txt), and
[Jobs documentation](ANNOTATION_PIPELINE.md#running-on-hugging-face-jobs).

The exact local image qualification found and fixed a missing `python` command
(the image provides `python3`) and a missing Cairo dependency. Setup now uses
an immutable image digest, an exact LeRobot commit and 259 Python constraints,
then runs strict dependency/import checks. Video metadata configuration is
forwarded explicitly and credentials stay in Jobs secrets. Local configuration
files and implicit local-cache uploads remain unsupported and fail early.

**Remaining action:** on an approved account, submit one bounded disposable Job
and publish original synthetic data to an explicitly chosen private Hub repository.
Local Docker bootstrap, mocked submission/publication tests and GPU inference
qualify implementation and runtime compatibility; they do not exercise hosted
scheduling, service permissions, upload transport or billing. No paid Job or
actual Hub publication was performed during this review.

## 3. Nice-to-have improvements

- Split the large [plan module](src/lerobot_align/modules/plan_subtasks_memory.py)
  only after release; preserve existing generation/alignment regressions.
- Update deprecated DatasetInfo and tiny-fixture tokenizer APIs when upgrading
  dependencies. Current test warnings do not indicate failed assertions.
- Keep future evaluation manifests portable and store immutable model revisions
  beside served aliases. Do not rewrite historical results to appear reproducible.

## 4. What is in good shape

| Original finding | Status and concrete change | Regression evidence |
| --- | --- | --- |
| B1: unsafe dataset publication | Resolved. [publication.py](src/lerobot_align/publication.py) builds an explicit canonical-file manifest and a private upload snapshot; unrelated files, custom staging, caches and symlinks are excluded/rejected. New destinations default private. Existing tag protection remains. | [test_publication.py](tests/test_publication.py), [test_cli.py](tests/test_cli.py) |
| B2: replacement of source licensing | Resolved. Full source card/YAML, attribution and license are preserved. Missing licenses require an explicit verified value; conflicting overrides fail. Software licensing is never substituted for data licensing. | Source card preservation, missing-license and conflicting-license tests in [test_publication.py](tests/test_publication.py) |
| B3: uncleared evaluation sources | Resolved in current packages/tree; public publication must use the clean export below. Complete member inventory, archive removal, structured redactions, corrected NOTICE and author-download/preparation instructions are committed. | [test_release_artifacts.py](tests/test_release_artifacts.py) |
| B4: visual failures treated as usable input | Resolved. [frames.py](src/lerobot_align/frames.py), CLI preflight and visual modules reject initialization/decode/missing-frame failures. Genuinely camera-less datasets remain distinguishable; visual annotation does not silently continue without required frames. | [test_annotation_integrity.py](tests/test_annotation_integrity.py), [test_frames.py](tests/test_frames.py) |
| B5: interrupted multi-shard rewrites | Resolved with a recoverable transaction. [transaction.py](src/lerobot_align/transaction.py) stages and validates transformed data before replacement, keeps fsynced originals and a journal, publishes metadata last, rolls back failures and recovers before the next read. A separate recovery CLI is included. | [test_transaction.py](tests/test_transaction.py): exceptions, hard process death after each replacement, staging/metadata failure, corruption and interrupted recovery |
| B6: unusable onboarding/placeholders | Resolved for the canonical CPU workflow. [README](README.md#installation), [uv.lock](uv.lock), package metadata, LICENSE and NOTICE contain an executable workflow and no unfilled release identities. Client/server model names, activation, endpoint keys and native-video settings are consistent. | Exact documented commands executed in a clean source export with a fresh environment |
| I1: installed distribution failures | Resolved. The sdist includes evaluation code required by tests and excludes research inputs/results. [check_distributions.sh](scripts/check_distributions.sh) runs tests outside the checkout against installed artifacts. Installed-package code/prompt fingerprinting no longer assumes `src/` or confuses `lerobot` with `lerobot_align`. | Full wheel and extracted-sdist suites; two added installed-fingerprint regressions in [test_metrics.py](evaluation/metrics/test_metrics.py) |
| I2: remote Jobs configuration | Resolved in code and local runtime qualification. Jobs require an existing accessible Hub dataset and never auto-upload a local cache; remote service qualification remains pending. | [test_jobs.py](tests/test_jobs.py) |
| I3: evaluation authentication and built-in labels | Resolved. Evaluation CLIs support an environment-sourced key and explicit ordered label files; missing/empty inputs fail early. Keys are absent from result provenance. | [test_evaluation_safety.py](tests/test_evaluation_safety.py) |
| I4: successful exit after failed evaluation | Resolved. Batch failures, missing episodes and empty results return nonzero while preserving completed JSONL rows and actionable failure records. | Failure, empty-selection and recovery regressions in [test_evaluation_safety.py](tests/test_evaluation_safety.py) and [test_eval_align_batch.py](tests/test_eval_align_batch.py) |
| I5: invalid annotation/numeric values | Resolved. Config, dataset timestamps and model span parsers reject non-finite/malformed values, invalid structures and invalid explicit labels before inference/writing. | [test_annotation_integrity.py](tests/test_annotation_integrity.py), [test_config_and_execution_safety.py](tests/test_config_and_execution_safety.py) |
| I6: ambiguous evaluation/calibration provenance | Resolved for new batch runs. Results fingerprint effective settings, source bytes, code, loaded prompts (including environment overrides), credential-free endpoint identity, package versions and ground truth. The fitter rejects mixed conditions and mismatched ground truth; unverifiable legacy input requires an explicit flag and remains marked unverified. Remote model aliases still require a separately recorded immutable weight revision. | [test_evaluation_safety.py](tests/test_evaluation_safety.py), [test_fit_align_calibration.py](tests/test_fit_align_calibration.py) |
| I7: exposed unauthenticated server defaults | Resolved. Built-in server commands and Docker host publication default to loopback. | [test_vlm_client.py](tests/test_vlm_client.py), [test_jobs.py](tests/test_jobs.py) |
| I8: external dataset identifiers | Resolved: all 17 source identities and immutable revisions are recorded and anonymously verified; historical ground truth matches local source bytes. Complete historical experiment reconstruction remains outstanding. | Two manifest regressions failed before the identifiers/verification were added; portable-root checks remain enabled |
| I9: visual end-to-end CI gap | Resolved for practical CPU CI. Original synthetic MP4s are decoded and sent through real HTTP as JPEG contact sheets or native MP4; the installed CLI runs all annotation modules and validates the rewritten dataset. CI runs both invocation forms and distribution tests. | [run_e2e_smoke.py](tests/run_e2e_smoke.py), [CI](.github/workflows/ci.yml) |
| Additional media-path disclosure bug | Resolved. A dataset metadata path cannot read or transmit media outside its root, including absolute paths, traversal and outside symlink targets. | Three new path-boundary regressions in [test_frames.py](tests/test_frames.py) |


Transactional limits remain explicit: external readers must stop during commit
and recovery, since POSIX cannot make multiple file replacements simultaneously
visible. Recovery needs retained originals/journal; reserve up to twice the
shard size as additional staging/backup space.

## Verification

Linux, Python 3.12.13, FFmpeg present. Final CPU reruns set `OMP_NUM_THREADS=2`;
no paid service or third-party annotation data is needed by the test suites.
The updated packages built successfully offline from cached build dependencies
after a sandbox DNS restriction. The fresh wheel's full suite passed inside the
sandbox; its HTTP smoke needed a separate run with localhost socket permission.
The sdist checks use a separate fresh environment with that permission. All 187
non-generated files in the rebuilt sdist match the reviewed allowlisted commit
exactly. The new download snippets passed shell/embedded-Python syntax checks;
the large dataset downloads themselves were not executed.

| Check | Exact result |
| --- | --- |
| Full unit suite: `pytest -q tests -p no:cacheprovider` | **628 passed**, 10 warnings |
| Full evaluation suite: `pytest -q evaluation -p no:cacheprovider` | **106 passed** |
| Anonymous historical source metadata verification | **17/17** exact revision matches, HTTP 200, public and ungated as reported by the Hub; complete file downloads not tested |
| Source-identity regression and shared preparation checks | **18 passed**; the two new identity cases failed before the manifest update |
| Full fresh installed-wheel suite, outside the checkout | **734 passed**, 10 warnings; all six console help commands passed |
| Extracted final sdist, fresh environment | **734 passed**, 10 warnings; lint and all six shell syntax checks passed |
| Regular real-media smoke in both distribution environments | Contact sheets and native video passed, including all modules and transactional writes |
| Guarded real-media smoke in both distribution environments | Both visual formats passed with **14 active advisory tripwires**, zero affected-operation calls |
| Original sharded embedding/Transformers model-load tests and guard positive control | **3 passed**; included in all final full suites |
| Publication, CLI, interrupted migration and Jobs rerun | **71 passed** (`test_publication.py`, `test_cli.py`, `test_transaction.py`, `test_jobs.py`) |
| Source export boundary tests | **8 passed**, including deleted historical material, local secrets, symlinks, traversal and failure injection; included in full suites |
| Runtime-only wheel before the NOTICE-only wheel update (identical application bytes) | Installation, CLI help and both visual formats passed without development/serving extras |
| Canonical README from an allowlisted source export and new environment (commands/runtime unchanged by this metadata follow-up) | Exact `uv sync --locked --extra dev`, CLI help and visual smoke commands passed |
| Lint | `ruff check src tests evaluation scripts/export_release.py` passed |
| Shell checks | `bash -n` passed on all six tracked shell scripts; ShellCheck was not installed/run |
| Whitespace | `git diff --check` passed |
| Credential-pattern/content-boundary scan | No live credentials identified across 189 release-source files and both artifacts. Two file-level URL flags are the same synthetic `.invalid` test fixture in source and sdist; zero unresolved flags. No `.env`, evaluation result archives or source annotations in either package |
| Final dependency audit | **6 findings across 5 packages**, one duplicate, 124 distributions inspected; unfiltered results retained, scoped assessment above |
| Local Jobs installation prelude in exact Docker image | Strict `pip check`, all runtime imports and installed CLI help passed with the 259-distribution constraints |
| Live GPU integration | **4 passed**: generation and fixed-label alignment, each with contact sheets and native video, using original synthetic MP4s |

The live model check used an installed wheel outside the source checkout on one
H100. It ran vLLM 0.19.1 image digest
`sha256:2622f38a0aa646c15ccc27bd5033911a58fd94ac69fd8f86aba0692d77cfe5b9`
and `Qwen/Qwen2.5-VL-7B-Instruct` weights at
`cc594898137f460bfe9f0759e9844b3ce807cfb5`. Model-quality scoring was not performed.
Its client/configuration/harness bytes are identical to the final implementation;
the last subsequent production change only closes the Jobs submission/upload
path and forwards its environment. That change is covered by the final full
wheel/sdist suites and the reproduced Jobs regressions. All review containers
were stopped and GPU memory released.

Warnings: seven existing DatasetInfo deprecations, two tiny-fixture tokenizer
deprecations and one host CUDA auto-detection warning during CPU embedding tests.
The GPU serving check used the compatible CUDA stack in Docker. A pattern scan
is not proof that arbitrary sensitive content is absent and does not clear old
Git history for publication.

Final artifacts (source/NOTICE follow-up `40351cb`, application code unchanged from
`b57ac71`; this report is not inside the packages):

- `lerobot_align-0.1.0-py3-none-any.whl`: **57 files**, **209142 bytes**,
  SHA-256 `4dbb95598b5792f3846f122d91ffe19dc9e2490266b92668b7b90b706620ce91`.
- `lerobot_align-0.1.0.tar.gz`: **188 files**, **924869 bytes**,
  SHA-256 `44d47c997be06650ea7f951459a28727df5cb0f73357d2c4536d757ee30f89eb`.

## Prioritized release checklist

- [x] Preserve the six pre-existing user modifications and unrelated local notes.
- [x] Fix B1-B6 with publication, integrity, migration and onboarding regressions.
- [x] Close the additional implicit Jobs cache-upload path.
- [x] Qualify contact-sheet/native-video generation and fixed-label alignment on a real model.
- [x] Pin and qualify the local Jobs runtime; retain strict dependency checks.
- [x] Substantiate scoped disposition of all current dependency findings; keep the raw audit visible.
- [x] Add an explicit committed-file source export excluding inherited history.
- [x] Recover and locally validate both corpora's historical source provenance.
- [x] Complete final full suites, wheel/sdist installs, runtime-only install, critical regressions and credential scan.
- [x] Verify all 17 recovered Hub IDs/revisions anonymously after explicit lookup approval; record immutable IDs, observations and download instructions.
- [ ] Run disposable hosted Jobs and private Hub publication checks on an approved account.
- [ ] For independently reproducible historical claims, provide authorized source access and the original saved predictions/fits.
- [ ] Publish only the reviewed clean source repository or built packages; keep the original development history private.

## 5. Final release recommendation

**Ready with minor fixes for the scoped source/package release.** The reproduced
correctness, data-integrity and publication bugs are fixed and the release has a
clean source publication path. Retain the dependency assessment and the explicit
historical/hosted-service limitations. This recommendation does not clear the
original Git history for publication or certify the historical study as externally
reproducible.
