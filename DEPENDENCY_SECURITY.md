# Dependency advisory assessment

Reviewed 2026-09-08 for the locked Linux/Python 3.12 annotation client,
evaluation harness and documented Qwen serving workflow. This is a scoped
reachability assessment, not a declaration that these dependency releases have
been patched. The final dependency audit reports six findings across five
packages, including one duplicate advisory. No finding is suppressed.

## Scope and evidence

Supported inputs are LeRobot v3 Parquet/MP4 datasets. The annotation client
loads their numeric tables and decoded frames, calls an HTTP endpoint, and
writes annotations. Evaluation loads a sentence embedding model for encoding.
The documented server loads published model weights with remote code disabled.
Training, model/tokenizer export, folder-based dataset builders, arbitrary
TorchScript classes, custom Python model code and macOS source builds are not
covered by this assessment.

| Package in local lock | Advisory and affected operation | Disposition for this scope |
| --- | --- | --- |
| accelerate 1.14.0 | CVE-2026-69112: sharded checkpoint index paths can escape their directory or name blocking special files. [Upstream report/fix proposal](https://github.com/huggingface/accelerate/pull/4070) | Not reached. Neither current Transformers model loading, SentenceTransformer encoding nor vLLM calls Accelerate's `load_checkpoint_in_model` / `load_checkpoint_and_dispatch`. Guarded tests load original sharded weights through both actual supported model-loading paths. No patched version is advertised by the audit. |
| datasets 4.8.5 | CVE-2026-66007: folder-builder metadata paths escape the dataset directory. [Upstream fix](https://github.com/huggingface/datasets/commit/f989ef9) | Not reached. `reader.py` reads fixed Parquet paths; `frames.py` uses LeRobot metadata, whose `load_nested_dataset` uses `Dataset.from_parquet`. No imagefolder/audiofolder/videofolder/pdffolder builder is used. |
| setuptools 81.0.0 | CVE-2026-59890 / GHSA-h35f-9h28-mq5c: Unicode normalization bypasses source-file exclusions on macOS. [Maintainer advisory](https://github.com/pypa/setuptools/security/advisories/GHSA-h35f-9h28-mq5c) | Not reached. This project builds with Hatchling on Linux. Its public source exporter reads exact Git blobs from an explicit manifest. This says nothing about a consumer using setuptools to build other projects on macOS. |
| torch 2.11.0 | CVE-2025-3000: TorchScript compilation of certain bare list/tuple class annotations causes crashes/memory corruption. [Upstream fix](https://github.com/pytorch/pytorch/commit/b90c949) | Not reached from supported input processing. Annotation uses tensors for video frames; embedding evaluation uses model forward/encode. Neither compiles input-supplied Python classes. vLLM 0.19.1 source has no `torch.jit.script` or `_script_impl` call. |
| transformers 5.5.4 | CVE-2026-9856: saving named chat templates can write outside the chosen directory. [Upstream fix](https://github.com/huggingface/transformers/commit/eaaaf84) | Not reached. Annotation is an HTTP client. SentenceTransformer evaluation and Transformers/vLLM serving load processors/models; they do not save tokenizers, processors or templates. Remote custom code remains disabled. |

LeRobot 0.6.1 constrains datasets below 5, torch below 2.12 and setuptools below
82. This project constrains Transformers below 5.6 for its tested local/remote
stack. The scanner's advertised fixed versions (5.0.1, 2.13.0, 83.0.0 and
5.10.0 respectively) therefore cannot be applied as blind upgrades. Keep
monitoring upstream compatibility and verify maintainer patch information when
selecting replacements. [Published LeRobot metadata](https://pypi.org/pypi/lerobot/json)

## Executable regression evidence

```bash
uv run --no-sync pytest -q tests/test_dependency_boundaries.py
uv run --no-sync python -m tests.run_dependency_boundaries
```

The tests install tripwires on fourteen affected entry points: all six exported
aliases of the two Accelerate checkpoint loaders, three folder-builder
methods, tokenizer save/template save, processor save, and both public/internal
TorchScript compilation entries. A positive control calls every entry and checks
that it is rejected and later restored. Separate tests create original random
tiny sharded BERT weights locally, then exercise the actual embedding evaluator and
Transformers serving ModelManager's processor/model loading and forward pass.
No downloaded model or third-party training data is needed by these tests.

The guarded smoke runs the installed CLI through real MP4 decoding, HTTP visual
transport, all annotation modules and transactional writes, for both contact
sheets and native video. Fourteen guards are active during input processing; both
runs record zero calls. Guards are test-only, installed after trusted module
bootstrap and before CLI inputs. They neither harden nor monkeypatch the shipped
application. CI and fresh wheel/sdist checks rerun them.

Static optional-server inspection additionally covered Transformers 5.5.4
`cli/serving/model_manager.py`, SentenceTransformers 6.0.1's transformer module,
and the complete vLLM 0.19.1 `vllm/` source. The vLLM runtime image uses torch
2.10, Transformers 5.5.3 and Accelerate 1.13.0, which are also unpatched for the corresponding
advisories; the same unused-operation reasoning applies. The documented Qwen
model was separately exercised on an H100 in that image with real visual inputs.
These checks are compatibility evidence, not a model quality assessment.

## Release decision and revalidation

The listed advisories are non-applicable to the exact supported operations
above based on source inspection and runtime boundary checks. The dependency
audit itself remains red. Do not reuse this assessment for custom servers,
training, exporting models, untrusted Python code or a different dependency
resolution. Adding an affected operation, enabling remote code, changing the
platform or changing dependency versions requires a fresh assessment before
publication. A compatible patched stack remains preferable when available.

Run an unfiltered dependency audit against the final installed environment and
retain its results with the release evidence. Review newly reported advisories;
this document is not an allowlist for all future findings in these packages.
