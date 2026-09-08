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
"""Plot motion-stratified sampling over robot state, base pose, and subtasks."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from lerobot_align import motion_sampling as motion_sampling_module
from lerobot_align.config import PlanConfig
from lerobot_align.modules.plan_subtasks_memory import (
    PlanSubtasksMemoryModule,
)
from lerobot_align.reader import EpisodeRecord, iter_episodes
from lerobot_align.vlm_client import StubVlmClient
from lerobot.datasets.io_utils import load_info

TRACE_BLUE = "#245B91"
SAMPLE_ORANGE = "#D97706"
INK = "#20242A"
MID_GREY = "#7A828C"
GRID_GREY = "#D8DDE3"
SEGMENT_FILLS = ("#F7F8FA", "#E9EDF2")


def _feature_names(feature: dict[str, Any] | None, width: int, fallback: str) -> list[str]:
    """Extract ordered metadata names without assuming a schema convention."""
    names: Any = (feature or {}).get("names")
    if isinstance(names, dict):
        flattened = next((value for value in names.values() if isinstance(value, list)), [])
    elif isinstance(names, list):
        flattened = names
    else:
        flattened = []
    return [
        str(flattened[index]) if index < len(flattened) and flattened[index] else f"{fallback}[{index}]"
        for index in range(width)
    ]


def _ground_truth(root: Path, episode_index: int) -> list[dict[str, Any]]:
    path = root / "meta" / "lerobot_annotations.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    episodes = payload.get("episodes", {})
    episode = episodes.get(str(episode_index), {}) if isinstance(episodes, dict) else {}
    subtasks = episode.get("subtasks", []) if isinstance(episode, dict) else []
    return [span for span in subtasks if isinstance(span, dict)]


def _motion_salience_by_group(
    timestamps: np.ndarray,
    groups: dict[str, list[Any]],
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Reproduce the selector's exact group and aggregate salience inputs."""
    prepared = motion_sampling_module._prepare_timestamps(timestamps)  # noqa: SLF001
    if prepared is None:
        return timestamps, {}, np.zeros_like(timestamps)
    sorted_timestamps, row_indices, original_count = prepared
    window = motion_sampling_module._smoothing_window_for_timestamps(  # noqa: SLF001
        sorted_timestamps,
        0.1,
    )
    group_scores: dict[str, np.ndarray] = {}
    for key, values in groups.items():
        prepared_group = motion_sampling_module._prepare_signal_groups(  # noqa: SLF001
            {key: values},
            sorted_timestamps,
            row_indices=row_indices,
            original_count=original_count,
            min_finite_fraction=0.8,
        )
        if prepared_group:
            group_scores[key] = motion_sampling_module._motion_transition_salience(  # noqa: SLF001
                sorted_timestamps,
                prepared_group[0],
                window,
            )
    if not group_scores:
        return sorted_timestamps, {}, np.zeros_like(sorted_timestamps)
    return sorted_timestamps, group_scores, np.max(list(group_scores.values()), axis=0)


def _make_sampler(root: Path, *, timestamp_budget: int, fps: float, sampling: str):
    return PlanSubtasksMemoryModule(
        vlm=StubVlmClient(responder=lambda _messages: {}),
        config=PlanConfig(
            frames_per_second=fps,
            max_frames_per_prompt=timestamp_budget,
            subtask_align_sampling=sampling,
            n_task_rephrasings=0,
            emit_plan=False,
            emit_memory=False,
        ),
        root=root,
    )


def _distinct_state_channels(
    state: np.ndarray,
    base: np.ndarray | None,
    state_names: list[str],
) -> tuple[list[tuple[str, np.ndarray]], str | None]:
    """Drop only unnamed state-tail channels proven to duplicate base pose."""
    named_count = sum(not name.startswith("state[") for name in state_names)
    duplicate_note = None
    keep_width = state.shape[1]
    if (
        base is not None
        and named_count < state.shape[1]
        and state.shape[1] - named_count == base.shape[1]
        and np.allclose(state[:, named_count:], base, equal_nan=True)
    ):
        keep_width = named_count
        duplicate_note = (
            f"state[{named_count}:{state.shape[1]}] exactly duplicates observation.base_pose "
            "and is shown only once."
        )
    return [(state_names[index], state[:, index]) for index in range(keep_width)], duplicate_note


def _shade_segments(
    axes: list[plt.Axes],
    spans: list[dict[str, Any]],
    *,
    episode_start: float,
    episode_end: float,
) -> None:
    for axis in axes:
        for index, span in enumerate(spans):
            start = max(episode_start, float(span["start"]))
            end = min(episode_end, float(span["end"]))
            if end <= start:
                continue
            axis.axvspan(start, end, color=SEGMENT_FILLS[index % 2], zorder=0)
            if index:
                axis.axvline(start, color=MID_GREY, linewidth=0.7, linestyle=(0, (2, 3)), zorder=1)


def _segment_sample_counts(samples: list[float], spans: list[dict[str, Any]]) -> list[int]:
    counts = []
    for index, span in enumerate(spans):
        start, end = float(span["start"]), float(span["end"])
        counts.append(
            sum(
                start <= timestamp <= end if index == len(spans) - 1 else start <= timestamp < end
                for timestamp in samples
            )
        )
    return counts


def _plot_episode(
    *,
    root: Path,
    record: EpisodeRecord,
    info_features: dict[str, dict[str, Any]],
    motion_sampler: PlanSubtasksMemoryModule,
    uniform_sampler: PlanSubtasksMemoryModule,
    output_path: Path,
    pdf: PdfPages,
    timestamp_budget: int,
    fps: float,
) -> None:
    feature_keys = motion_sampler._motion_feature_keys
    groups = record.frame_columns(feature_keys)
    timestamps = np.asarray(record.frame_timestamps, dtype=np.float64)
    state_values = groups.get("observation.state")
    base_values = groups.get("observation.base_pose")
    if state_values is None:
        raise ValueError("plot requires observation.state in the selected dataset")

    state = np.asarray(state_values, dtype=np.float64)
    base = np.asarray(base_values, dtype=np.float64) if base_values is not None else None
    state_names = _feature_names(
        info_features.get("observation.state"),
        state.shape[1],
        "state",
    )
    state_channels, duplicate_note = _distinct_state_channels(state, base, state_names)
    channels = list(state_channels)
    if base is not None:
        base_names = _feature_names(
            info_features.get("observation.base_pose"),
            base.shape[1],
            "base_pose",
        )
        channels.extend((name, base[:, index]) for index, name in enumerate(base_names))

    motion_timestamps = motion_sampler._align_sample_timestamps(record)
    uniform_timestamps = uniform_sampler._align_sample_timestamps(record)
    salience_timestamps, group_saliences, aggregate_salience = _motion_salience_by_group(
        timestamps,
        groups,
    )
    spans = _ground_truth(root, record.episode_index)

    row_count = len(channels) + 2
    figure, axes_array = plt.subplots(
        row_count,
        1,
        figsize=(15.5, 2.6 + row_count * 1.28),
        sharex=True,
        gridspec_kw={"height_ratios": [1.55, 0.9, *([1.0] * len(channels))]},
    )
    axes = list(np.atleast_1d(axes_array))
    _shade_segments(
        axes,
        spans,
        episode_start=float(timestamps[0]),
        episode_end=float(timestamps[-1]),
    )

    salience_axis = axes[0]
    group_styles = {
        "observation.state": (TRACE_BLUE, (0, (5, 2)), "state-group salience"),
        "observation.base_pose": (SAMPLE_ORANGE, (0, (1, 2)), "base-group salience"),
    }
    for key, score in group_saliences.items():
        color, linestyle, label = group_styles.get(
            key,
            (MID_GREY, (0, (3, 2)), f"{key}-group salience"),
        )
        salience_axis.plot(
            salience_timestamps,
            score,
            color=color,
            linewidth=0.8,
            linestyle=linestyle,
            alpha=0.82,
            label=label,
            zorder=2,
        )
    salience_axis.plot(
        salience_timestamps,
        aggregate_salience,
        color=INK,
        linewidth=1.0,
        label="final max salience",
        zorder=3,
    )
    motion_salience = np.interp(motion_timestamps, salience_timestamps, aggregate_salience)
    salience_axis.scatter(
        motion_timestamps,
        motion_salience,
        s=20,
        color=SAMPLE_ORANGE,
        edgecolor=INK,
        linewidth=0.35,
        zorder=5,
    )
    salience_axis.set_ylabel("motion\nsalience")
    salience_axis.set_ylim(-0.03, 1.08)
    salience_axis.yaxis.set_major_locator(MaxNLocator(4))
    salience_axis.legend(loc="upper right", ncol=3, frameon=False, fontsize=7.8)

    for index, span in enumerate(spans, start=1):
        midpoint = (float(span["start"]) + float(span["end"])) / 2
        salience_axis.text(
            midpoint,
            1.02,
            f"S{index}",
            transform=salience_axis.get_xaxis_transform(),
            ha="center",
            va="bottom",
            color=INK,
            fontsize=8.5,
            fontweight="bold",
        )

    shift_axis = axes[1]
    pair_count = min(len(uniform_timestamps), len(motion_timestamps))
    paired_uniform = np.asarray(uniform_timestamps[:pair_count])
    paired_motion = np.asarray(motion_timestamps[:pair_count])
    shifts = paired_motion - paired_uniform
    shift_axis.axhline(0.0, color=MID_GREY, linewidth=0.8, zorder=2)
    shift_axis.vlines(
        paired_uniform,
        0.0,
        shifts,
        color=SAMPLE_ORANGE,
        linewidth=0.75,
        alpha=0.8,
        zorder=3,
    )
    shift_axis.scatter(
        paired_uniform,
        np.zeros(pair_count),
        marker="o",
        s=15,
        facecolor="white",
        edgecolor=MID_GREY,
        linewidth=0.65,
        zorder=4,
    )
    shift_axis.scatter(
        paired_uniform,
        shifts,
        marker="o",
        s=16,
        color=SAMPLE_ORANGE,
        edgecolor=INK,
        linewidth=0.25,
        zorder=5,
    )
    max_shift = max(0.1, float(np.max(np.abs(shifts), initial=0.0)) * 1.18)
    shift_axis.set_ylim(-max_shift, max_shift)
    shift_axis.set_ylabel("shift from\nuniform (s)")
    shift_axis.yaxis.set_major_locator(MaxNLocator(3))

    for axis, (name, values) in zip(axes[2:], channels, strict=True):
        axis.plot(timestamps, values, color=TRACE_BLUE, linewidth=0.85, zorder=2)
        axis.scatter(
            motion_timestamps,
            np.interp(motion_timestamps, timestamps, values),
            s=14,
            color=SAMPLE_ORANGE,
            edgecolor=INK,
            linewidth=0.25,
            zorder=4,
        )
        axis.set_ylabel(name, rotation=0, ha="right", va="center")
        axis.yaxis.set_major_locator(MaxNLocator(3))

    for axis in axes:
        axis.grid(axis="y", color=GRID_GREY, linewidth=0.55, alpha=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color(MID_GREY)
        axis.spines["bottom"].set_color(MID_GREY)
        axis.tick_params(colors=INK, labelsize=8)
        axis.margins(x=0)
        axis.set_xlim(float(timestamps[0]), float(timestamps[-1]))
    axes[-1].set_xlabel("episode-relative time (seconds)", color=INK)

    legend_handles = [
        Line2D([0], [0], color=TRACE_BLUE, linewidth=1.4, label="recorded trajectory"),
        Line2D(
            [0],
            [0],
            marker="o",
            markersize=5.5,
            markerfacecolor=SAMPLE_ORANGE,
            markeredgecolor=INK,
            linewidth=0,
            label="motion-stratified sample",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            markersize=5.5,
            markerfacecolor="white",
            markeredgecolor=MID_GREY,
            linewidth=0,
            label="uniform anchor (shift row)",
        ),
        Line2D(
            [0],
            [0],
            color=MID_GREY,
            linewidth=0.8,
            linestyle=(0, (2, 3)),
            label="human subtask boundary",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.948),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    duration = float(timestamps[-1] - timestamps[0])
    figure.suptitle(
        f"Episode {record.episode_index}: motion-stratified sampling over joints and base pose",
        x=0.08,
        y=0.986,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color=INK,
    )
    figure.text(
        0.08,
        0.956,
        (
            f"{len(timestamps):,} source frames at ~{len(timestamps) / max(duration, 1e-9):.1f} Hz; "
            f"{len(motion_timestamps)} samples from a {timestamp_budget}-timestamp cap "
            f"and {fps:g} fps request. Selection = 65% motion salience + 35% stratum-center proximity."
        ),
        ha="left",
        va="top",
        fontsize=9.5,
        color=INK,
    )

    segment_text = "  •  ".join(
        f"S{index}: {span.get('label', '')}" for index, span in enumerate(spans, start=1)
    )
    wrapped_segments = "\n".join(textwrap.wrap(segment_text, width=170))
    notes = [wrapped_segments] if wrapped_segments else []
    motion_counts = _segment_sample_counts(motion_timestamps, spans)
    uniform_counts = _segment_sample_counts(uniform_timestamps, spans)
    notes.append(
        f"Median |motion − uniform| shift: {np.median(np.abs(shifts)):.3f}s; "
        f"maximum: {np.max(np.abs(shifts), initial=0.0):.3f}s. "
        f"Samples per human segment — motion {motion_counts}, uniform {uniform_counts}."
    )
    if duplicate_note:
        notes.append(
            f"{duplicate_note} The top score still uses the production groups, including those "
            "dimensions inside the state-group RMS."
        )
    notes.append(
        "Source: meta/lerobot_annotations.json and parquet observation state; "
        "orange points are exact timestamps selected by motion_sampling.py. Human segments are "
        "diagnostic overlays and are not inputs to the sampler. Axes use recorded units."
    )
    figure.text(
        0.08,
        0.012,
        "\n".join(notes),
        ha="left",
        va="bottom",
        fontsize=8.3,
        color=INK,
    )
    figure.tight_layout(rect=(0.075, 0.09, 0.99, 0.925), h_pad=0.28)
    figure.savefig(output_path, dpi=180, facecolor="white", bbox_inches="tight")
    pdf.savefig(figure, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--episodes", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--timestamp-budget",
        type=int,
        default=60,
        help="per-camera timestamp cap; 60 keeps individual sampling decisions legible",
    )
    parser.add_argument("--fps", type=float, default=3.0, help="requested alignment sampling rate")
    parser.add_argument("--output-dir", type=Path, default=Path("motion_sampling_plots"))
    args = parser.parse_args()
    if args.timestamp_budget < 2:
        parser.error("--timestamp-budget must be at least 2 for a multi-frame plot")

    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = {
        record.episode_index: record for record in iter_episodes(root, only_episodes=tuple(args.episodes))
    }
    missing = [episode for episode in args.episodes if episode not in records]
    if missing:
        raise ValueError(f"episode(s) not found: {missing}")

    info_features = load_info(root).features
    motion_sampler = _make_sampler(
        root,
        timestamp_budget=args.timestamp_budget,
        fps=args.fps,
        sampling="motion_stratified",
    )
    uniform_sampler = _make_sampler(
        root,
        timestamp_budget=args.timestamp_budget,
        fps=args.fps,
        sampling="uniform",
    )

    episode_slug = "-".join(f"{episode:03d}" for episode in args.episodes)
    pdf_path = output_dir / (
        f"motion_sampling_budget_{args.timestamp_budget:03d}_episodes_{episode_slug}.pdf"
    )
    with PdfPages(pdf_path) as pdf:
        for episode_index in args.episodes:
            output_path = output_dir / (
                f"episode_{episode_index:03d}_motion_sampling_budget_{args.timestamp_budget:03d}.png"
            )
            _plot_episode(
                root=root,
                record=records[episode_index],
                info_features=info_features,
                motion_sampler=motion_sampler,
                uniform_sampler=uniform_sampler,
                output_path=output_path,
                pdf=pdf,
                timestamp_budget=args.timestamp_budget,
                fps=args.fps,
            )
            print(output_path)
    print(pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
