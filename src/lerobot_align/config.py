#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from lerobot.configs.default import JobConfig

SubtaskGenerationFrameFormat = Literal["contact_sheet", "video"]


def _number(name: str, value: Any, *, minimum: float = 0, integer: bool = False,
            positive: bool = False, maximum: float | None = None) -> None:
    valid = (not isinstance(value, bool) and isinstance(value, (int, float))
             and math.isfinite(value) and value >= minimum
             and (not positive or value > 0)
             and (not integer or isinstance(value, int))
             and (maximum is None or value <= maximum))
    if not valid:
        raise ValueError(f"{name} must be a finite {'integer' if integer else 'number'} "
                         f"{'greater than 0' if positive else f'>= {minimum}'}"
                         f"{f' and <= {maximum}' if maximum is not None else ''}; got {value!r}")

# The annotation pipeline boots its own vLLM server, so the pod starts from the
# official vLLM runtime rather than the prebuilt `lerobot-gpu` training image;
# `lerobot` is pip-installed on top by this package's remote-job launcher.
# v0.19.1 is the first vLLM release image with Transformers 5.5.x, which can
# share LeRobot 0.6.1's Hugging Face Hub 1.x runtime; it retains the tested
# CUDA 12.9 / PyTorch 2.10 stack. Pin it so `latest` cannot alter a run.
DEFAULT_ANNOTATE_JOB_IMAGE = (
    "vllm/vllm-openai@sha256:2622f38a0aa646c15ccc27bd5033911a58fd94ac69fd8f86aba0692d77cfe5b9"
)


@dataclass
class AnnotationJobConfig(JobConfig):
    """`JobConfig` with the annotation runtime's defaults.

    Adds `lerobot_ref` because the vLLM image ships no lerobot: the pod installs
    it from git, and the ref selects the compatible LeRobot API. Point it at a
    branch/tag/SHA explicitly when testing a different LeRobot build remotely.
    """

    image: str = DEFAULT_ANNOTATE_JOB_IMAGE
    # Annotation is a bounded pass over a dataset; a tighter cap than training's
    # "2d" keeps a wedged vLLM server from burning a day of GPU time.
    timeout: str | None = "2h"
    lerobot_ref: str = "7e241bd630a3719a56157a497ce5d08f244784f1"  # v0.6.1
    # Explicit immutable wheel URL (+SHA256) or VCS requirement (+full commit)
    # required for remote Jobs. None fails before provisioning.
    package_spec: str | None = None


@dataclass
class PlanConfig:
    """``plan`` module: subtasks + plan + memory + task augmentation."""

    enabled: bool = True

    # Target number of free-form alternatives, plus the canonical ``task_aug``
    # at t=0 (renderer rotates ${task} among them); 0 disables. Filtering and
    # deduplication can leave fewer alternatives; validation warns on shortfalls.
    n_task_rephrasings: int = 10

    # Derive the task from video instead of episode_task: off / if_short / always.
    # Affects prompts only; ``meta/tasks.parquet`` is untouched.
    derive_task_from_video: str = "if_short"
    derive_task_min_words: int = 3

    # Visual representation for open-ended task derivation and subtask
    # generation. ``contact_sheet`` is the established default; ``video``
    # sends uniformly sampled frames as a native clip so video-capable VLMs
    # can use temporal tokens. Seeded relabeling remains contact-sheet based.
    subtask_generate_frame_format: SubtaskGenerationFrameFormat = "contact_sheet"

    # Policy when a stage explicitly configured as native ``video`` cannot
    # build a genuine video_url input. The established behavior falls back to
    # timestamped contact sheets. ``error`` makes the modality a hard contract:
    # fail before sending an alternate VLM request. This applies to generation
    # and fixed-label alignment, but not to seeded relabeling (which is always
    # an explicitly contact-sheet-based stage).
    subtask_video_fallback: Literal["contact_sheet", "error"] = "contact_sheet"

    # By default, the subtask describe/segment passes render the episode as
    # macrodata/refiner-style contact sheets: sampled frames packed into JPEG
    # grids with each frame's timestamp burned into its corner, so the VLM
    # cites the exact source time of a boundary directly. ``video`` keeps the
    # same frame budget and episode/window timeline.
    #
    # ``frames_per_second`` is the sampling rate: 2.0 = one frame every 0.5s.
    frames_per_second: float = 2.0
    # Frame budget per VLM call (= columns × rows × sheets). Subtask
    # generation currently uses one camera. When a whole episode sampled at
    # ``frames_per_second`` exceeds this, the episode is
    # AUTOMATICALLY split into consecutive windows that each honor
    # ``max_frames_per_prompt`` (one describe→segment call per window, still at
    # the full ``frames_per_second`` density), and the per-window spans are
    # combined + stitched into one contiguous cover. The
    # final two windows share the remaining intervals when a greedy split would
    # otherwise leave a tiny tail.
    # Windows are independent, so an event crossing a seam can be split or
    # repeated. An episode of any length is still covered at full density.
    max_frames_per_prompt: int = 60
    contact_sheet_columns: int = 5
    contact_sheet_frames_per_sheet: int = 20
    contact_sheet_frame_width: int = 224
    contact_sheet_quality: int = 84

    min_subtask_seconds: float = 1.5
    plan_max_steps: int = 8

    # Narrate-only grounding pass before segmenting — best defense against subtasks
    # invented from the task text (+1 VLM call/episode).
    subtask_describe_first: bool = True

    # Seeded relabeling: after segmentation, re-label each span with a focused
    # pass that sees the previous / current / next segment contact sheets and
    # minimally corrects the seed label (macrodata's best end-to-end labeling
    # step). Costs +1 VLM call per subtask; off by default.
    subtask_seeded_relabel: bool = False
    # Frames sampled uniformly per segment sheet in the relabel pass.
    subtask_relabel_frames: int = 5

    # Optional generate -> align chain. After open-ended generation (and the
    # optional seeded-relabel pass), pass only the ordered generated labels
    # through the whole-episode fixed-label aligner used by ``subtasks_path``. All
    # ``subtask_align_*`` settings apply, including native-video input and
    # calibration. Replace boundaries only when alignment preserves every
    # ordered label; otherwise warn and retain the complete generated spans.
    # Off by default so the established generation path is unchanged.
    subtask_realign_generated: bool = False

    # --- Fixed-label alignment (supplied or generated labels) --------------
    # Path to a JSON file supplying the subtask labels. When set, the VLM never
    # writes subtask text: it is shown the whole episode in the configured
    # alignment format plus the ordered list, and returns a start/end for each
    # label in ONE call.
    # Labels are authoritative; only the boundaries are predicted. A label the
    # episode never performs comes back unplaced and is stitched over (and
    # counted against ``subtask_align_min_fraction``).
    #
    # ``max_frames_per_prompt`` caps the camera frames for this call. Contact
    # sheets share the budget across selected views (two cameras and a budget
    # of 300 means at most 150 timestamps per view); native video stacks the
    # views within each frame and keeps the full timestamp budget. Alignment is
    # never split across calls, since a chunk cannot tell which part of the
    # episode it covers.
    #
    # The file is either a flat list applied to every episode:
    #     ["pick up the cup", "pour the water", "put the cup down"]
    # or an object keyed by episode index, with an optional "default":
    #     {"default": [...], "0": [...], "7": [...]}
    #
    # ``subtask_describe_first`` and ``subtask_seeded_relabel`` do not apply in
    # this mode (there is nothing to ground or relabel) and are skipped.
    subtasks_path: Path | None = None

    # Optional ordered camera list used only by fixed-label alignment. Empty
    # keeps the historical single-camera behaviour (``--vlm.camera_key`` or the
    # provider default). Keys are matched against the dataset's decodable video
    # features at runtime; missing optional views are warned about and skipped.
    # Camera roles are intentionally not inferred from names such as ``wrist``
    # or ``side`` because LeRobot metadata does not standardize those roles.
    subtask_align_camera_keys: tuple[str, ...] = ()

    # ``uniform`` is the established timestamp grid. ``motion_stratified``
    # preserves equal-time coverage but uses task-agnostic numeric observation
    # changes to choose a more informative real frame within each time stratum.
    # It falls back to the exact uniform grid when suitable signals are absent.
    subtask_align_sampling: Literal["uniform", "motion_stratified"] = "uniform"

    # Optional explicit numeric observation columns for motion-stratified
    # sampling. Empty means discover low-dimensional floating-point
    # ``observation.*`` features from meta/info.json without assuming a robot
    # morphology, joint naming convention, or vector length.
    subtask_align_motion_feature_keys: tuple[str, ...] = ()

    # How the sampled frames reach the model during fixed-label alignment.
    # ``contact_sheet`` is the established path: frame grids with the source
    # time burned into each tile, which the model reads back as text.
    # ``video`` sends one re-encoded clip instead and lets a video-native model
    # (Qwen3-VL and family) carry time itself — those models interleave literal
    # ``<12.3 seconds>`` tokens ahead of each frame, so the timestamp never
    # round-trips through pixels. The clip is encoded so that clip time equals
    # episode time. Synchronized multiview input is vertically stacked into
    # each video frame so the timeline still advances once per timestamp.
    subtask_align_frame_format: Literal["contact_sheet", "video"] = "contact_sheet"

    # Per-boundary timing offsets subtracted from the model's answer, as a JSON
    # file written by the ``lerobot-align-fit`` diagnostic command.
    #
    # The model's boundary error against a human reference is largely a fixed
    # offset rather than noise: it marks a subtask as starting several seconds
    # after the annotator does, consistently, in the same direction. Measured on
    # 50 fridge episodes the first boundary sat +5.86s late with a median that
    # did not move when the prompt was rewritten to ask for motion onset, nor
    # when the model was given nine times as many frames. Subtracting a
    # calibration fit on ten labelled episodes recovered about +9 macro-tIoU
    # points on held-out episodes.
    #
    # This is a per-corpus adapter, not a general correction: the offsets encode
    # one annotation convention and must be refit for a new task or a new
    # labelling team. The file records the label list it was fit against and is
    # ignored, with a warning, for episodes whose labels differ. Legacy files
    # without labels remain accepted for direct fixed-label alignment, but are
    # ignored in generated-label realignment because their schema is unknown.
    subtask_align_calibration_path: Path | None = None

    # Minimum fraction of the fixed labels that must be placed before an
    # episode is accepted; below it the run fails instead of emitting spans.
    # A dropped subtask is stitched over, so a mostly failed alignment still
    # covers the episode and passes validation — wrong output that looks
    # right. Set to 0 to restore the stitch-and-warn behaviour.
    subtask_align_min_fraction: float = 0.5

    # --- Import subtasks the dataset already records ------------------------
    # Sources already carry their own timings, so importing costs no VLM call.
    #   "off"        default: generate (or align, when subtasks_path is set)
    #   "auto"       try every source below in order, else generate
    #   "sarm"       {dense,sparse}_subtask_* columns on meta/episodes/*.parquet
    #   "lerobot_annotations" legacy spans or v2 atoms in meta/lerobot_annotations.json
    #   "language"   style="subtask" rows a previous annotate run wrote
    #   "task_index" runs of the per-frame task_index column
    # An explicit source never silently falls back to generation; "auto" does.
    # Mutually exclusive with ``subtasks_path``.
    subtask_import: str = "off"

    # Preferred SARM granularity; the other prefix and the legacy unprefixed
    # columns are accepted as fallbacks.
    subtask_import_sarm_prefix: str = "dense"

    # Emit ``style="plan"`` rows at each boundary; False = subtasks + memory only.
    emit_plan: bool = True

    # Emit ``style="memory"`` rows at each boundary; False = subtasks (+ plan) only.
    # Symmetric counterpart of ``emit_plan``.
    emit_memory: bool = True

    # (subtask spans are always stitched to a contiguous full-episode cover; not configurable.)

    # Optional EgoMimic-style 5-axis task augmentation; replaces n_task_rephrasings.
    task_aug_axes: TaskAugAxesConfig = field(default_factory=lambda: TaskAugAxesConfig())

    def __post_init__(self) -> None:
        for name in ("derive_task_min_words", "plan_max_steps", "contact_sheet_columns",
                     "contact_sheet_frames_per_sheet", "contact_sheet_frame_width"):
            _number(f"plan.{name}", getattr(self, name), integer=True, positive=True)
        _number("plan.frames_per_second", self.frames_per_second, positive=True)
        _number("plan.min_subtask_seconds", self.min_subtask_seconds)
        _number("plan.contact_sheet_quality", self.contact_sheet_quality, integer=True,
                positive=True, maximum=100)
        _number("plan.subtask_align_min_fraction", self.subtask_align_min_fraction, maximum=1)
        if self.derive_task_from_video not in {"off", "if_short", "always"}:
            raise ValueError("plan.derive_task_from_video must be off, if_short or always")
        if self.subtask_generate_frame_format not in {"contact_sheet", "video"}:
            raise ValueError(
                "plan.subtask_generate_frame_format must be 'contact_sheet' or 'video', "
                f"got {self.subtask_generate_frame_format!r}."
            )
        if self.subtask_align_frame_format not in {"contact_sheet", "video"}:
            raise ValueError(
                "plan.subtask_align_frame_format must be 'contact_sheet' or 'video', "
                f"got {self.subtask_align_frame_format!r}."
            )
        if self.subtask_video_fallback not in {"contact_sheet", "error"}:
            raise ValueError(
                "plan.subtask_video_fallback must be 'contact_sheet' or 'error', "
                f"got {self.subtask_video_fallback!r}."
            )
        if (
            isinstance(self.n_task_rephrasings, bool)
            or not isinstance(self.n_task_rephrasings, int)
            or self.n_task_rephrasings < 0
        ):
            raise ValueError(
                "plan.n_task_rephrasings must be a non-negative integer, "
                f"got {self.n_task_rephrasings!r}."
            )
        if (
            isinstance(self.subtask_relabel_frames, bool)
            or not isinstance(self.subtask_relabel_frames, int)
            or self.subtask_relabel_frames <= 0
        ):
            raise ValueError(
                "plan.subtask_relabel_frames must be a positive integer, "
                f"got {self.subtask_relabel_frames!r}."
            )
        if (
            isinstance(self.max_frames_per_prompt, bool)
            or not isinstance(self.max_frames_per_prompt, int)
            or self.max_frames_per_prompt <= 0
        ):
            raise ValueError(
                "plan.max_frames_per_prompt must be a positive integer, "
                f"got {self.max_frames_per_prompt!r}."
            )


@dataclass
class TaskAugAxesConfig:
    """5-axis t=0 task augmentation (EgoMimic-style): synonym / omit_arm /
    omit_orientation / omit_grasp_method / combined. Replaces n_task_rephrasings
    when enabled; each variant becomes a ``task_aug`` row. Axes with nothing to
    omit emit fewer entries. Defaults (3+3+2+2+2) match EgoMimic."""

    enabled: bool = False

    synonym_paraphrase: int = 3
    omit_arm: int = 3
    omit_orientation: int = 2
    omit_grasp_method: int = 2
    combined_omissions: int = 2

    def __post_init__(self) -> None:
        for name in (
            "synonym_paraphrase",
            "omit_arm",
            "omit_orientation",
            "omit_grasp_method",
            "combined_omissions",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"plan.task_aug_axes.{name} must be a non-negative integer, got {value!r}."
                )


@dataclass
class InterjectionsConfig:
    """``interjections`` module: interjections + paired speech."""

    enabled: bool = True

    # Each emits a paired (interjection, speech) row at a boundary that already
    # carries the deterministic remaining-subtasks plan.
    max_interjections_per_episode: int = 3
    interjection_min_t: float = 2.0

    # Frame window centered on the timestamp so the VLM sees motion, not one frame.
    interjection_window_seconds: float = 2.0
    interjection_window_frames: int = 4

    # Minimum accepted speech atoms per selected episode. The initial
    # acknowledgement is attempted for every non-empty episode, so one catches
    # an invalid/empty VLM response without requiring any optional mid-episode
    # interjections. Set to 0 to make this module best-effort.
    min_speech_atoms_per_episode: int = 1

    def __post_init__(self) -> None:
        _number("interjections.max_interjections_per_episode", self.max_interjections_per_episode,
                integer=True)
        _number("interjections.interjection_min_t", self.interjection_min_t)
        _number("interjections.interjection_window_seconds", self.interjection_window_seconds,
                positive=True)
        _number("interjections.interjection_window_frames", self.interjection_window_frames,
                positive=True, integer=True)
        if (
            isinstance(self.min_speech_atoms_per_episode, bool)
            or not isinstance(self.min_speech_atoms_per_episode, int)
            or self.min_speech_atoms_per_episode < 0
        ):
            raise ValueError(
                "interjections.min_speech_atoms_per_episode must be a non-negative integer, "
                f"got {self.min_speech_atoms_per_episode!r}."
            )


@dataclass
class VqaConfig:
    """``vqa`` module: general VQA."""

    enabled: bool = True
    vqa_emission_hz: float = 1.0
    K: int = 1
    """Consecutive frames per emission tick. The VLM grounds on the FIRST frame,
    so K>1 smears stale labels onto moved frames. Default 1 (no smear)."""
    question_types: tuple[str, ...] = ("bbox", "keypoint", "count", "attribute", "spatial")

    # True: ground VQA only on --vlm.camera_key (default: every camera).
    restrict_to_default_camera: bool = False

    # Minimum complete (user, assistant) VQA pairs per selected episode when
    # the dataset exposes at least one camera and K schedules frames. Set to 0
    # for best-effort VQA; camera-less datasets are always exempt.
    min_pairs_per_episode: int = 1

    def __post_init__(self) -> None:
        _number("vqa.K", self.K, integer=True)
        if any(q not in {"bbox", "keypoint", "count", "attribute", "spatial"} for q in self.question_types):
            raise ValueError("vqa.question_types contains an unsupported question type")
        try:
            valid_frequency = (
                not isinstance(self.vqa_emission_hz, bool)
                and math.isfinite(self.vqa_emission_hz)
                and self.vqa_emission_hz > 0
            )
        except TypeError:
            valid_frequency = False
        if not valid_frequency:
            raise ValueError(
                "vqa.vqa_emission_hz must be a positive finite number, "
                f"got {self.vqa_emission_hz!r}."
            )
        if not self.question_types:
            raise ValueError("vqa.question_types must contain at least one question type.")
        if (
            isinstance(self.min_pairs_per_episode, bool)
            or not isinstance(self.min_pairs_per_episode, int)
            or self.min_pairs_per_episode < 0
        ):
            raise ValueError(
                "vqa.min_pairs_per_episode must be a non-negative integer, "
                f"got {self.min_pairs_per_episode!r}."
            )


@dataclass
class VlmConfig:
    """Shared Qwen-VL client configuration."""

    # Only ``openai`` (an OpenAI-compatible vLLM server, spawned locally only
    # when auto_serve=True); ``stub`` is for tests.
    backend: str = "openai"
    # The model the evaluation study and scripts/start_qwen38.sh serve. The
    # diagnostics default to this same value so that scoring a run does not
    # silently use a different model from the one that produced it.
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"

    # OpenAI-compatible endpoint; ``EMPTY`` key works for local servers.
    api_base: str = "http://localhost:8000/v1"
    api_key: str = field(default_factory=lambda: os.environ.get("LEROBOT_VLM_API_KEY", "EMPTY"), repr=False)

    # Off by default: supply ``--vlm.api_base`` pointing at an already-running
    # OpenAI-compatible endpoint. Pass ``--vlm.auto_serve=true`` to have the
    # tool spawn one locally instead (it needs a local GPU and `vllm serve` or
    # `transformers serve` installed). With auto_serve off, an unreachable
    # api_base fails fast rather than silently starting a local server.
    #
    # This default is LOCAL-ONLY. A run submitted with ``--job.target=<flavor>``
    # lands on a fresh GPU pod where nothing is listening and nothing else will
    # start a server, so ``jobs.build_pod_command`` sends
    # ``--vlm.auto_serve=true`` unless the submitter set the flag explicitly.
    auto_serve: bool = False
    serve_port: int = 8000
    # Override the auto-serve command; ``{port}`` substituted per replica.
    serve_command: str | None = None

    # Independent servers for round-robin routing (one per GPU). num_gpus=0 = one each.
    parallel_servers: int = 1
    num_gpus: int = 0
    client_concurrency: int = 16
    serve_ready_timeout_s: float = 600.0

    max_new_tokens: int = 512
    temperature: float = 0.2

    # Auto-serve context length (None → 32768); other vLLM flags go in serve_command.
    max_model_len: int | None = None

    # Camera for keyframes; None → first ``observation.images.*`` key.
    camera_key: str | None = None
    # Forwarded as extra_body.chat_template_kwargs (e.g. {"enable_thinking": false}).
    chat_template_kwargs: dict[str, Any] | None = None

    # vLLM's Qwen2.5 adapter omits timing metadata for decoded arrays; use
    # "client" with that adapter. Qwen3 supplies its own metadata, and sending
    # it again causes duplicate processor arguments. Keep that mode explicit.
    video_metadata_source: Literal["server", "client"] = field(
        default_factory=lambda: os.environ.get("LEROBOT_VLM_VIDEO_METADATA_SOURCE", "server")
    )

    # OpenAI-style thinking budget hint ("low"/"medium"/"high"); forwarded to
    # the server when set. Used to cap a thinking model's reasoning so it
    # leaves tokens for the actual JSON answer on OpenAI-compatible endpoints.
    reasoning_effort: str | None = None

    def __post_init__(self) -> None:
        if self.video_metadata_source not in {"server", "client"}:
            raise ValueError("vlm.video_metadata_source must be 'server' or 'client'")
        for name in ("parallel_servers", "client_concurrency", "max_new_tokens"):
            _number(f"vlm.{name}", getattr(self, name), integer=True, positive=True)
        _number("vlm.num_gpus", self.num_gpus, integer=True)
        _number("vlm.serve_port", self.serve_port, integer=True, positive=True, maximum=65535)
        _number("vlm.serve_ready_timeout_s", self.serve_ready_timeout_s, positive=True)
        _number("vlm.temperature", self.temperature, maximum=2)
        if self.max_model_len is not None:
            _number("vlm.max_model_len", self.max_model_len, integer=True, positive=True)


@dataclass
class ExecutorConfig:
    """Executor settings (intra-process episode concurrency; distribution via HF Jobs)."""

    # Episodes processed concurrently per phase; main knob for saturating the servers.
    episode_parallelism: int = 16

    # Skip incomplete episodes, but abort before writing if their fraction
    # exceeds this limit. 0 restores strict batch failure; 1 permits any
    # partial success. An entirely incomplete run always fails, even at 1.
    max_incomplete_episode_fraction: float = 1.0

    def __post_init__(self) -> None:
        _number("executor.episode_parallelism", self.episode_parallelism, integer=True, positive=True)
        value = self.max_incomplete_episode_fraction
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError(
                "executor.max_incomplete_episode_fraction must be a finite number in [0, 1], "
                f"got {value!r}."
            )


@dataclass
class AnnotationPipelineConfig:
    """Top-level config for ``lerobot-align`` (rewrites data shards in place)."""

    # Hub dataset: download source when ``root`` unset; push target when push_to_hub
    # is on and ``new_repo_id`` unset.
    repo_id: str | None = None

    # Separate push target (matches the LeRobot edit tools). Unset → push in place.
    new_repo_id: str | None = None

    root: Path | None = None

    # Defaults to ``<root>/.annotate_staging/``.
    staging_dir: Path | None = None

    seed: int = 1729

    plan: PlanConfig = field(default_factory=PlanConfig)
    interjections: InterjectionsConfig = field(default_factory=InterjectionsConfig)
    vqa: VqaConfig = field(default_factory=VqaConfig)

    vlm: VlmConfig = field(default_factory=VlmConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)

    # Where the annotation runs: omitted / "local" annotates on this machine, any
    # other value is an HF Jobs flavor (e.g. "h200") and submits the run there.
    # List flavors + pricing with `hf jobs hardware`.
    job: AnnotationJobConfig = field(default_factory=AnnotationJobConfig)

    skip_validation: bool = False
    only_episodes: tuple[int, ...] | None = None

    # Keyframe decode backend forwarded to ``decode_video_frames``. None →
    # library default (torchcodec when available, else PyAV). Or pin
    # ``"torchcodec"`` / ``"pyav"`` explicitly.
    video_backend: str | None = None

    # Upload to the Hub (new_repo_id if set, else repo_id; one must be set).
    push_to_hub: bool = False
    push_private: bool = True
    # Only for a source whose original license is missing from its card.
    dataset_license: str | None = None
    push_commit_message: str | None = None

    # Moving an existing Hub version tag requires a recoverable delete/create
    # transaction. Keep that destructive operation opt-in; without this flag,
    # push preflight aborts before creating or uploading anything when the
    # target repository already has the dataset's version tag.
    allow_version_tag_move: bool = False

    def __post_init__(self) -> None:
        _number("seed", self.seed, integer=True)
        if self.only_episodes is not None:
            for episode in self.only_episodes:
                _number("only_episodes", episode, integer=True)
            if not self.only_episodes or len(set(self.only_episodes)) != len(self.only_episodes):
                raise ValueError("only_episodes must be non-empty and contain unique episode IDs")

    def resolved_staging_dir(self, root: Path) -> Path:
        return self.staging_dir if self.staging_dir is not None else root / ".annotate_staging"
