#!/usr/bin/env python3
"""Validate and plot paired Qwen visual-token measurements by frame format.

The input is one JSON object per held-out trajectory.  Each row contains both
the contact-sheet and native-video count for the same sampled frames.  The
script deliberately plots processor-expanded visual placeholders rather than
the modality-dependent token counters reported by an inference server.
"""

from __future__ import annotations

import argparse
import gzip
import html
import json
import math
from pathlib import Path
import re
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "evaluation/results/frame_format_tokens"
EXPECTED_CORPORA = ("corpus_a", "corpus_b")
CORPUS_LABELS = {"corpus_a": "Corpus A", "corpus_b": "Corpus B"}
COLORS = {"corpus_a": "#0d9488", "corpus_b": "#3b82f6", "all": "#334155"}
MARKERS = {"corpus_a": "o", "corpus_b": "D"}
INK = "#0f172a"
MUTED = "#64748b"
GRID = "#e2e8f0"


class PlotInputError(ValueError):
    """Raised when an input cannot support the claimed comparison."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot paired contact-sheet versus native-video Qwen processor visual-token counts. "
            "The command refuses to write a figure when rows are invalid or either corpus has "
            "non-positive workload reduction."
        )
    )
    parser.add_argument(
        "--rows",
        type=Path,
        default=DEFAULT_OUT / "rows.jsonl.gz",
        help="Paired measurement rows (.jsonl or .jsonl.gz).",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_OUT / "summary.json",
        help="Measurement summary and processor provenance JSON.",
    )
    parser.add_argument(
        "--svg",
        type=Path,
        default=None,
        help="Output SVG (default: figure3.svg beside --summary).",
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=None,
        help="Output PNG (default: figure3.png beside --summary).",
    )
    return parser


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise PlotInputError(f"rows file does not exist: {path}")
    opener = gzip.open if path.suffix == ".gz" else open
    rows: list[dict[str, Any]] = []
    try:
        with opener(path, "rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise PlotInputError(f"{path}:{line_number}: row is not a JSON object")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise PlotInputError(f"could not read {path}: {exc}") from exc
    if not rows:
        raise PlotInputError(f"rows file is empty: {path}")
    return rows


def _load_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PlotInputError(f"summary file does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlotInputError(f"could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PlotInputError(f"summary is not a JSON object: {path}")
    return value


def _positive_integer(row: dict[str, Any], key: str, row_number: int) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlotInputError(f"row {row_number}: {key} must be a positive integer")
    if not math.isfinite(float(value)) or int(value) != value or value <= 0:
        raise PlotInputError(f"row {row_number}: {key} must be a positive integer")
    return int(value)


def _nonnegative_integer(row: dict[str, Any], key: str, row_number: int) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlotInputError(f"row {row_number}: {key} must be a non-negative integer")
    if not math.isfinite(float(value)) or int(value) != value or value < 0:
        raise PlotInputError(f"row {row_number}: {key} must be a non-negative integer")
    return int(value)


def _aliased_integer(
    row: dict[str, Any],
    canonical: str,
    aliases: tuple[str, ...],
    row_number: int,
    *,
    allow_zero: bool,
) -> int:
    present = [key for key in (canonical, *aliases) if key in row]
    if not present:
        alternatives = ", ".join(repr(key) for key in (canonical, *aliases))
        raise PlotInputError(f"row {row_number}: expected one of {alternatives}")
    validator = _nonnegative_integer if allow_zero else _positive_integer
    values = {key: validator(row, key, row_number) for key in present}
    if len(set(values.values())) != 1:
        raise PlotInputError(f"row {row_number}: conflicting aliases {values!r}")
    return next(iter(values.values()))


def _close(left: float, right: float, *, tolerance: float = 1e-9) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def _validate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    required_text = ("corpus", "task", "component", "identity_namespace", "source")
    identities: set[tuple[str, str, int]] = set()
    source_identities: set[tuple[str, str, int]] = set()
    schema_versions: set[int] = set()
    cleaned: list[dict[str, Any]] = []

    for row_number, raw in enumerate(rows, start=1):
        for key in required_text:
            if not isinstance(raw.get(key), str) or not raw[key].strip():
                raise PlotInputError(f"row {row_number}: {key} must be a non-empty string")
        if raw.get("valid") is not True:
            raise PlotInputError(
                f"row {row_number}: valid must be explicitly true; refusing to publish"
            )
        if raw.get("same_frames") is not True:
            raise PlotInputError(
                f"row {row_number}: same_frames must be explicitly true; refusing to publish"
            )

        version = _positive_integer(raw, "schema_version", row_number)
        schema_versions.add(version)
        episode = _aliased_integer(
            raw,
            "component_episode",
            ("episode",),
            row_number,
            allow_zero=True,
        )
        identity = (raw["corpus"], raw["component"], episode)
        if identity in identities:
            raise PlotInputError(f"row {row_number}: duplicate paired identity {identity!r}")
        identities.add(identity)

        namespace = raw["identity_namespace"]
        source_episode = _nonnegative_integer(raw, "source_episode", row_number)
        source_identity = (raw["corpus"], namespace, source_episode)
        if source_identity in source_identities:
            raise PlotInputError(f"row {row_number}: duplicate source identity {source_identity!r}")
        source_identities.add(source_identity)

        sheet = _positive_integer(raw, "contact_sheet_visual_tokens", row_number)
        video = _positive_integer(raw, "video_visual_tokens", row_number)
        n_frames = _aliased_integer(
            raw,
            "n_frames",
            ("n_sampled_frames",),
            row_number,
            allow_zero=False,
        )
        expected_ratio = video / sheet
        for key, expected in (
            ("video_to_contact_ratio", expected_ratio),
            ("contact_to_video_ratio", sheet / video),
            ("video_reduction_fraction", 1.0 - expected_ratio),
        ):
            if key not in raw:
                continue
            value = raw[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise PlotInputError(f"row {row_number}: {key} must be numeric")
            if not math.isfinite(float(value)) or not _close(float(value), expected):
                raise PlotInputError(
                    f"row {row_number}: {key}={value!r} is inconsistent with the token counts"
                )

        cleaned.append(
            {
                **raw,
                "component_episode": episode,
                "contact_sheet_visual_tokens": sheet,
                "video_visual_tokens": video,
                "n_frames": n_frames,
            }
        )

    if len(schema_versions) != 1:
        raise PlotInputError(f"rows mix schema versions: {sorted(schema_versions)}")
    if schema_versions != {1}:
        raise PlotInputError(f"unsupported row schema_version={next(iter(schema_versions))!r}")
    corpora = {row["corpus"] for row in cleaned}
    if corpora != set(EXPECTED_CORPORA):
        raise PlotInputError(
            f"the blog figure requires exactly corpus_a and corpus_b; found {sorted(corpora)}"
        )
    return cleaned


def _unique_number(rows: list[dict[str, Any]], key: str) -> float:
    values: set[float] = set()
    for index, row in enumerate(rows, start=1):
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PlotInputError(f"row {index}: {key} must be numeric")
        if not math.isfinite(float(value)):
            raise PlotInputError(f"row {index}: {key} must be finite")
        values.add(float(value))
    if len(values) != 1:
        raise PlotInputError(f"rows do not share one {key}: {sorted(values)}")
    return next(iter(values))


def _unique_text(rows: list[dict[str, Any]], key: str) -> str:
    values = {row.get(key) for row in rows}
    if len(values) != 1 or not all(isinstance(value, str) and value for value in values):
        raise PlotInputError(f"rows do not share one non-empty {key}: {sorted(map(str, values))}")
    return str(next(iter(values)))


def _summary_for(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sheet = sum(row["contact_sheet_visual_tokens"] for row in rows)
    video = sum(row["video_visual_tokens"] for row in rows)
    if sheet <= 0 or video <= 0:
        raise PlotInputError("aggregate token totals must be positive")
    reductions = sorted(
        1.0 - row["video_visual_tokens"] / row["contact_sheet_visual_tokens"] for row in rows
    )
    middle = len(reductions) // 2
    if len(reductions) % 2:
        median = reductions[middle]
    else:
        median = (reductions[middle - 1] + reductions[middle]) / 2
    return {
        "n_episodes": len(rows),
        "n_tasks": len({row["task"] for row in rows}),
        "n_components": len({row["component"] for row in rows}),
        "contact_sheet_visual_tokens": sheet,
        "video_visual_tokens": video,
        "video_to_contact_ratio": video / sheet,
        "contact_to_video_ratio": sheet / video,
        "video_reduction_fraction": 1.0 - video / sheet,
        "mean_episode_reduction_fraction": sum(reductions) / len(reductions),
        "median_episode_reduction_fraction": median,
        "video_lower_pairs": sum(
            row["video_visual_tokens"] < row["contact_sheet_visual_tokens"] for row in rows
        ),
    }


def _validate_summary(
    summary: dict[str, Any], derived: dict[str, dict[str, Any]], row_count: int
) -> None:
    if "schema_version" not in summary:
        raise PlotInputError("summary is missing schema_version")
    if summary["schema_version"] != 1:
        raise PlotInputError(f"unsupported summary schema_version={summary['schema_version']!r}")

    population = summary.get("population_validation")
    if not isinstance(population, dict) or population.get("status") != "passed":
        raise PlotInputError("summary population_validation must explicitly report status='passed'")
    for key in ("rows", "unique_local_identities", "unique_source_identities"):
        if population.get(key) != row_count:
            raise PlotInputError(
                f"summary population_validation.{key}={population.get(key)!r} "
                f"does not match {row_count:,} rows"
            )

    actual_pairs = summary.get("actual_pair_validation")
    actual_corpora = actual_pairs.get("corpora") if isinstance(actual_pairs, dict) else None
    if (
        not isinstance(actual_pairs, dict)
        or actual_pairs.get("status") != "passed"
        or not isinstance(actual_pairs.get("pairs"), int)
        or isinstance(actual_pairs.get("pairs"), bool)
        or not isinstance(actual_corpora, dict)
        or set(actual_corpora) != set(EXPECTED_CORPORA)
        or any(
            isinstance(actual_corpora[corpus], bool)
            or not isinstance(actual_corpora[corpus], int)
            or actual_corpora[corpus] < 1
            for corpus in EXPECTED_CORPORA
        )
    ):
        raise PlotInputError(
            "summary must report passed decoded-media validation with at least one pair per corpus"
        )

    validation = summary.get("validation")
    if isinstance(validation, bool) and not validation:
        raise PlotInputError("summary marks the measurement invalid")
    if isinstance(validation, dict):
        if validation.get("valid") is False or validation.get("all_rows_valid") is False:
            raise PlotInputError("summary validation reports invalid measurement rows")
        status = str(validation.get("status", "")).lower()
        if status in {"error", "failed", "invalid", "rejected"}:
            raise PlotInputError(f"summary validation status is {status!r}")

    aggregates = summary.get("aggregates")
    if not isinstance(aggregates, dict):
        raise PlotInputError("summary is missing an aggregates object")
    corpus_aggregates = aggregates.get("corpora", {})
    if not isinstance(corpus_aggregates, dict):
        raise PlotInputError("summary aggregates.corpora is not an object")
    aliases = {"all": ("all", "pooled"), **{name: (name,) for name in EXPECTED_CORPORA}}
    for name, possible_keys in aliases.items():
        saved = next((aggregates[key] for key in possible_keys if key in aggregates), None)
        if saved is None and name in EXPECTED_CORPORA:
            saved = corpus_aggregates.get(name)
        if saved is None:
            raise PlotInputError(f"summary aggregates are missing {name!r}")
        if not isinstance(saved, dict):
            raise PlotInputError(f"summary aggregate {name!r} is not an object")
        expected = derived[name]
        for key in (
            "n_episodes",
            "n_tasks",
            "n_components",
            "contact_sheet_visual_tokens",
            "video_visual_tokens",
            "video_to_contact_ratio",
            "contact_to_video_ratio",
            "video_reduction_fraction",
        ):
            if key not in saved:
                raise PlotInputError(f"summary aggregate {name!r} is missing {key}")
            left = saved[key]
            right = expected[key]
            if isinstance(left, bool) or not isinstance(left, (int, float)):
                raise PlotInputError(f"summary aggregate {name!r}.{key} must be numeric")
            if not _close(float(left), float(right)):
                raise PlotInputError(
                    f"summary aggregate {name!r}.{key}={left!r} does not match rows ({right!r})"
                )
        saved_median = saved.get("median_episode_video_reduction_fraction")
        expected_median = expected["median_episode_reduction_fraction"]
        if (
            isinstance(saved_median, bool)
            or not isinstance(saved_median, (int, float))
            or not _close(float(saved_median), float(expected_median))
        ):
            raise PlotInputError(
                f"summary aggregate {name!r}.median_episode_video_reduction_fraction "
                f"does not match rows ({expected_median!r})"
            )

    if derived["all"]["n_episodes"] != row_count:
        raise PlotInputError("internal row-count inconsistency")


def _processor_label(summary: dict[str, Any]) -> str:
    processor = summary.get("processor")
    if not isinstance(processor, dict):
        raise PlotInputError("summary is missing a processor object")
    model = processor.get("model_id", processor.get("model"))
    source = processor.get("resolved_name_or_path", processor.get("requested_source"))
    if not isinstance(model, str) or not model:
        if not isinstance(source, str) or not source:
            raise PlotInputError("summary processor is missing model identity")
        cache_match = re.search(r"models--([^/]+)--([^/]+)/", source)
        model = (
            f"{cache_match.group(1)}/{cache_match.group(2)}" if cache_match else Path(source).name
        )
    revision = processor.get("revision", processor.get("model_revision"))
    if (not isinstance(revision, str) or not revision) and isinstance(source, str):
        snapshot_match = re.search(r"/snapshots/([^/]+)", source)
        if snapshot_match:
            revision = snapshot_match.group(1)
    if isinstance(revision, str) and revision:
        return f"{model} processor {revision[:8]}"
    return f"{model} processor"


def _format_token_tick(value: float, _position: float) -> str:
    if value < 1:
        return ""
    return f"{value:,.0f}"


def _format_compact_tokens(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,}"


def _write_accessible_svg(path: Path, title: str, description: str) -> None:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"<svg\b[^>]*>", text)
    if match is None:
        raise PlotInputError(f"matplotlib output is not an SVG document: {path}")
    opening = match.group(0)
    opening = re.sub(r'\swidth="[^"]+"', ' width="900"', opening, count=1)
    opening = re.sub(r'\sheight="[^"]+"', ' height="560"', opening, count=1)
    if " role=" not in opening:
        opening = opening[:-1] + ' role="img" aria-labelledby="figure3-title figure3-desc">'
    accessible = (
        opening
        + f'<title id="figure3-title">{html.escape(title)}</title>'
        + f'<desc id="figure3-desc">{html.escape(description)}</desc>'
    )
    body = text[match.end() :]
    # Matplotlib writes a direct-child title when Title metadata is supplied.
    # Replace that accessibility title instead of leaving two competing names.
    body = re.sub(r"^\s*<title>.*?</title>", "", body, count=1, flags=re.DOTALL)
    text = text[: match.start()] + accessible + body
    # Matplotlib leaves spaces after SVG path commands at line endings. Keep
    # generated publication artifacts clean under Git's whitespace checks.
    had_final_newline = text.endswith("\n")
    text = "\n".join(line.rstrip() for line in text.splitlines())
    if had_final_newline:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def _render(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    derived: dict[str, dict[str, Any]],
    svg_path: Path,
    png_path: Path,
) -> None:
    fps = _unique_number(rows, "sampling_fps")
    max_frames = int(_unique_number(rows, "max_frames"))
    frame_width = int(_unique_number(rows, "frame_width"))
    camera = _unique_text(rows, "camera")
    if not _close(fps, 2.0) or max_frames != 300 or frame_width != 224:
        raise PlotInputError(
            "Figure 3 is defined for the evaluation protocol (2 fps, <=300 frames, 224 px); "
            f"rows contain {fps:g} fps, <= {max_frames} frames, {frame_width} px"
        )
    if camera != "observation.images.wrist":
        raise PlotInputError(
            f"Figure 3 is defined for observation.images.wrist; rows contain camera={camera!r}"
        )

    processor = _processor_label(summary)
    count = len(rows)
    title = "Qwen3.8 processor-expanded visual tokens by frame format"
    subtitle = (
        f"Same wrist frames · {fps:g} fps · ≤{max_frames} frames · {frame_width} px · "
        f"{count:,} held-out trajectories"
    )
    description = (
        "A log-scale paired scatter compares contact-sheet and native-video visual tokens for "
        f"{count:,} held-out trajectories. A second panel compares normalized bars for summed "
        "contact-sheet and native-video workloads in Corpus A, Corpus B, and both corpora "
        "pooled, with absolute totals and ratio-of-sums reductions labeled."
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelcolor": "#334155",
            "axes.titlecolor": "#334155",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "svg.fonttype": "none",
        }
    )
    figure = plt.figure(figsize=(9, 5.6), dpi=100, facecolor="white")
    grid = figure.add_gridspec(1, 2, width_ratios=(1.35, 1.15), wspace=0.33)
    scatter_axis = figure.add_subplot(grid[0, 0])
    summary_axis = figure.add_subplot(grid[0, 1])

    figure.text(0.065, 0.955, title, fontsize=14, fontweight="bold", color=INK, va="top")
    figure.text(0.065, 0.912, subtitle, fontsize=9.5, color=MUTED, va="top")

    all_values: list[int] = []
    for corpus in EXPECTED_CORPORA:
        selected = [row for row in rows if row["corpus"] == corpus]
        x = [row["contact_sheet_visual_tokens"] for row in selected]
        y = [row["video_visual_tokens"] for row in selected]
        all_values.extend(x)
        all_values.extend(y)
        scatter_axis.scatter(
            x,
            y,
            s=13,
            alpha=0.34,
            color=COLORS[corpus],
            marker=MARKERS[corpus],
            edgecolors="none",
            label=f"{CORPUS_LABELS[corpus]} (n={len(selected):,})",
        )

    lower = min(all_values) * 0.78
    upper = max(all_values) * 1.28
    scatter_axis.plot([lower, upper], [lower, upper], color="#475569", lw=1.25, zorder=0)
    scatter_axis.text(
        0.97,
        0.94,
        "equal tokens",
        transform=scatter_axis.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color="#475569",
        rotation=45,
    )
    scatter_axis.set_xscale("log")
    scatter_axis.set_yscale("log")
    scatter_axis.set_xlim(lower, upper)
    scatter_axis.set_ylim(lower, upper)
    scatter_axis.set_aspect("equal", adjustable="box")
    scatter_axis.set_title("Paired trajectory counts", loc="left", fontsize=11, fontweight="bold")
    scatter_axis.set_xlabel("Contact sheets · visual tokens / trajectory")
    scatter_axis.set_ylabel("Native video · visual tokens / trajectory")
    locator = LogLocator(base=10, subs=(1.0, 2.0, 5.0))
    scatter_axis.xaxis.set_major_locator(locator)
    scatter_axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    scatter_axis.xaxis.set_major_formatter(FuncFormatter(_format_token_tick))
    scatter_axis.yaxis.set_major_formatter(FuncFormatter(_format_token_tick))
    scatter_axis.xaxis.set_minor_formatter(NullFormatter())
    scatter_axis.yaxis.set_minor_formatter(NullFormatter())
    scatter_axis.grid(which="major", color=GRID, linewidth=0.8)
    scatter_axis.grid(which="minor", visible=False)
    scatter_axis.spines[["top", "right"]].set_visible(False)
    legend = scatter_axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker=MARKERS[corpus],
                linestyle="none",
                markersize=6,
                markerfacecolor=COLORS[corpus],
                markeredgecolor="none",
                label=(f"{CORPUS_LABELS[corpus]} (n={derived[corpus]['n_episodes']:,})"),
            )
            for corpus in EXPECTED_CORPORA
        ],
        loc="lower right",
        frameon=False,
        fontsize=8.5,
        handletextpad=0.45,
    )
    for text in legend.get_texts():
        text.set_color("#475569")

    summary_axis.set_title(
        "Aggregate visual-token workload", loc="left", fontsize=11, fontweight="bold"
    )
    names = ("corpus_a", "corpus_b", "all")
    y_positions = (2, 1, 0)
    bar_offset = 0.18
    bar_height = 0.22
    summary_axis.axvline(0, color="#475569", linewidth=1.0)
    for name, y in zip(names, y_positions, strict=True):
        stats = derived[name]
        sheet_total = int(stats["contact_sheet_visual_tokens"])
        video_total = int(stats["video_visual_tokens"])
        video_percent = 100 * video_total / sheet_total
        reduction = 100 - video_percent
        summary_axis.barh(
            y + bar_offset,
            100,
            height=bar_height,
            color="#e2e8f0",
            edgecolor="#64748b",
            linewidth=0.9,
            hatch="///",
            zorder=2,
        )
        summary_axis.barh(
            y - bar_offset,
            video_percent,
            height=bar_height,
            color=COLORS[name],
            edgecolor=COLORS[name],
            linewidth=0.9,
            zorder=3,
        )
        summary_axis.text(
            97.5,
            y + bar_offset,
            f"Sheets {_format_compact_tokens(sheet_total)}",
            color=INK,
            fontsize=7.7,
            fontweight="bold",
            ha="right",
            va="center",
        )
        summary_axis.text(
            video_percent - 2.2,
            y - bar_offset,
            f"Video {_format_compact_tokens(video_total)}",
            color="white",
            fontsize=7.7,
            fontweight="bold",
            ha="right",
            va="center",
        )
        summary_axis.text(
            (video_percent + 100) / 2,
            y,
            f"{reduction:.1f}% fewer",
            color=INK,
            fontsize=7.8,
            fontweight="bold",
            ha="center",
            va="center",
        )
    summary_axis.set_yticks(y_positions, ("Corpus A", "Corpus B", "Pooled\n(A+B)"))
    summary_axis.set_xlim(0, 108)
    summary_axis.set_ylim(-0.48, 2.48)
    summary_axis.set_xlabel("Relative visual tokens · contact sheets = 100%")
    summary_axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0f}%"))
    summary_axis.set_xticks((0, 25, 50, 75, 100))
    summary_axis.grid(axis="x", color=GRID, linewidth=0.8)
    summary_axis.spines[["top", "right", "left"]].set_visible(False)
    summary_axis.tick_params(axis="y", length=0)

    figure.text(
        0.065,
        0.058,
        "Visual tokens = Σ prod(grid_thw) / merge_size² (merge_size = 2). "
        "Reduction = 1 − Σvideo / Σcontact-sheet.",
        fontsize=8.2,
        color=MUTED,
        va="bottom",
    )
    figure.text(
        0.065,
        0.027,
        "Complete evaluation census; no sampling interval.",
        fontsize=8.2,
        color=MUTED,
        va="bottom",
    )
    figure.text(0.935, 0.027, processor, fontsize=8.2, color=MUTED, ha="right", va="bottom")
    figure.subplots_adjust(left=0.075, right=0.965, top=0.83, bottom=0.18)

    svg_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        svg_path,
        format="svg",
        facecolor="white",
        metadata={"Title": title, "Description": description},
    )
    figure.savefig(png_path, format="png", dpi=100, facecolor="white")
    plt.close(figure)
    _write_accessible_svg(svg_path, title, description)


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    try:
        rows = _validate_rows(_load_rows(args.rows))
        summary = _load_summary(args.summary)
        derived = {
            corpus: _summary_for([row for row in rows if row["corpus"] == corpus])
            for corpus in EXPECTED_CORPORA
        }
        derived["all"] = _summary_for(rows)
        for corpus in EXPECTED_CORPORA:
            stats = derived[corpus]
            gates = {
                "ratio-of-sums workload reduction": stats["video_reduction_fraction"],
                "mean paired reduction": stats["mean_episode_reduction_fraction"],
                "median paired reduction": stats["median_episode_reduction_fraction"],
            }
            failures = [
                f"{label}={100 * value:+.2f}%" for label, value in gates.items() if value <= 0
            ]
            if failures:
                raise PlotInputError(
                    f"{CORPUS_LABELS[corpus]} fails the positive-reduction publication gate "
                    f"({'; '.join(failures)}); refusing to publish Figure 3"
                )
        _validate_summary(summary, derived, len(rows))
        svg_path = args.svg or args.summary.parent / "figure3.svg"
        png_path = args.png or args.summary.parent / "figure3.png"
        if svg_path.resolve() == png_path.resolve():
            raise PlotInputError("--svg and --png must be different paths")
        _render(rows, summary, derived, svg_path, png_path)
    except PlotInputError as exc:
        parser.error(str(exc))

    for name in (*EXPECTED_CORPORA, "all"):
        stats = derived[name]
        print(
            f"{name}: n={stats['n_episodes']:,} "
            f"reduction={100 * stats['video_reduction_fraction']:.2f}% "
            f"mean_pair={100 * stats['mean_episode_reduction_fraction']:.2f}% "
            f"median_pair={100 * stats['median_episode_reduction_fraction']:.2f}% "
            f"lower_pairs={stats['video_lower_pairs']:,}/{stats['n_episodes']:,}"
        )
    print(f"wrote {svg_path}")
    print(f"wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
