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
"""Keyframe extraction for the annotation pipeline.

Modules attach decoded camera frames to their VLM prompts so the model can
ground subtask decomposition, interjection scenarios, and VQA in actual
visual content. The pipeline shares one provider across modules and one
episode at a time, with a small per-episode cache so multiple modules
querying the same timestamp pay decode cost once.
"""

from __future__ import annotations

import io
import logging
import math
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol

import PIL.Image
import torch

from lerobot.configs import RGBEncoderConfig
from lerobot.datasets.video_utils import decode_video_frames, reencode_video

from .reader import EpisodeRecord, snap_to_frame

logger = logging.getLogger(__name__)


class FrameProviderError(RuntimeError):
    """A requested video frame set could not be made available."""


class FrameProvider(Protocol):
    """Decodes camera frames at episode-relative timestamps."""

    @property
    def camera_keys(self) -> list[str]:
        """All ``observation.images.*`` feature keys this provider can decode."""

    def frames_at(
        self,
        record: EpisodeRecord,
        timestamps: list[float],
        camera_key: str | None = None,
        *,
        fail_on_error: bool = False,
    ) -> list[Any]:
        """Return one decoded frame per timestamp from ``camera_key`` (or default).

        Frames are ``torch.Tensor`` (``C, H, W`` uint8) — the shape
        :func:`lerobot.datasets.video_utils.decode_video_frames` returns.
        :func:`to_image_blocks` converts them to PIL only at the VLM-message
        boundary.

        Empty list if the camera is unavailable. ``camera_key=None`` falls back
        to the provider's default camera so existing single-camera callers
        (the ``plan`` and ``interjections`` modules) keep working unchanged.
        ``fail_on_error=True`` instead raises :class:`FrameProviderError`, with
        the original initialization/decode exception chained as its cause.
        """

    def video_for_episode(
        self,
        record: EpisodeRecord,
        max_frames: int,
        camera_key: str | None = None,
    ) -> list[Any]:
        """Return up to ``max_frames`` decoded frames covering the whole episode.

        Sampling is uniform across the episode duration. Frames are
        ``torch.Tensor`` (``C, H, W`` uint8), ready to be wrapped into one
        ``{"type":"video", "video":<list>}`` block for a Qwen-VL-compatible
        model that pools temporally itself. Empty list if no camera available.
        """


@dataclass
class _NullProvider:
    """No-op provider used when the dataset has no video keys or in tests."""

    initialization_error: Exception | None = None

    @property
    def camera_keys(self) -> list[str]:
        if self.initialization_error is not None:
            raise FrameProviderError("Visual input initialization failed") from self.initialization_error
        return []

    def frames_at(
        self,
        record: EpisodeRecord,
        timestamps: list[float],
        camera_key: str | None = None,
        *,
        fail_on_error: bool = False,
    ) -> list[Any]:
        if self.initialization_error is not None:
            cause = self.initialization_error
            raise FrameProviderError(
                "video frame-provider initialization failed: "
                f"{type(cause).__name__}: {cause}. Check the dataset metadata/video paths "
                "and filesystem or Hugging Face cache permissions."
            ) from cause
        if fail_on_error:
            raise FrameProviderError(
                "no decodable video camera was found. Check meta/info.json for a video-backed "
                "observation.images.* feature or select one with --vlm.camera_key."
            )
        return []

    def video_for_episode(
        self,
        record: EpisodeRecord,
        max_frames: int,
        camera_key: str | None = None,
    ) -> list[Any]:
        return []


def null_provider(initialization_error: Exception | None = None) -> FrameProvider:
    return _NullProvider(initialization_error=initialization_error)


@dataclass
class VideoFrameProvider:
    """Decodes frames from the dataset's ``observation.images.*`` streams.

    By default the *first* camera key is used for the ``plan`` module
    (subtask decomposition) and the ``interjections`` module (interjection
    scenarios) — those prompts care about *what is happening*, not which
    angle. The ``vqa`` module instead iterates over every camera in
    :attr:`camera_keys` so each frame's
    grounded answer (bbox/keypoint/...) is tagged with the camera it was
    grounded against.

    ``camera_key`` overrides the default-camera choice but does not restrict
    :attr:`camera_keys`. Pass ``camera_key`` explicitly to ``frames_at`` /
    ``video_for_episode`` to read a non-default stream.

    Caches up to ``cache_size`` decoded frames per process so repeated module
    requests for the same episode timestamp do not decode twice.
    """

    root: Path
    camera_key: str | None = None
    tolerance_s: float = 1e-2
    cache_size: int = 256
    # Keyframe decode backend forwarded to
    # :func:`lerobot.datasets.video_utils.decode_video_frames`. ``None``
    # uses the library default (torchcodec when available, else PyAV).
    video_backend: str | None = None
    _meta: Any = field(default=None, init=False, repr=False)
    _cache: dict = field(default_factory=dict, init=False, repr=False)
    _camera_keys: list[str] = field(default_factory=list, init=False, repr=False)
    # Pipeline runs the three module phases under a ThreadPoolExecutor (see
    # ``ExecutorConfig.episode_parallelism``); guard the dict cache and the
    # one-shot warn flag against concurrent updates from worker threads.
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    # Serializes decode_video_frames calls: torchcodec hands out one
    # ``VideoDecoder`` per file from a process-wide cache, and the decoder
    # is not safe to drive from multiple threads at once.
    _decode_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _warned_decode_fail: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata  # noqa: PLC0415

        self._meta = LeRobotDatasetMetadata(repo_id="local", root=self.root)
        # Only ``video_keys`` are decodable here: the clip/decode paths read
        # ``videos/<key>/from_timestamp`` from episode metadata, which exists
        # only for video-stored cameras. Image-stored cameras (also in
        # ``camera_keys``) would KeyError, so restrict the list — and the
        # default — to video keys.
        # Depth cameras are excluded from the annotation pipeline for now.
        depth_keys = set(self._meta.depth_keys)
        keys = [key for key in self._meta.video_keys if key not in depth_keys]
        # Last-resort fallback: if metadata didn't surface any video keys but
        # the caller explicitly named a camera (``--vlm.camera_key=...``),
        # trust them — the key is by definition known to exist on the dataset.
        if not keys and self.camera_key:
            keys = [self.camera_key]
        self._camera_keys = keys
        if self.camera_key is None:
            self.camera_key = keys[0] if keys else None

    @property
    def camera_keys(self) -> list[str]:
        """All ``observation.images.*`` keys available on this dataset."""
        return list(self._camera_keys)

    def frames_at(
        self,
        record: EpisodeRecord,
        timestamps: list[float],
        camera_key: str | None = None,
        *,
        fail_on_error: bool = False,
    ) -> list[Any]:
        target = camera_key if camera_key is not None else self.camera_key
        if not timestamps:
            return []
        if target is None:
            if fail_on_error:
                raise FrameProviderError(
                    f"episode {record.episode_index}: no decodable video camera was resolved. "
                    "Check meta/info.json or pass --vlm.camera_key=<video-feature-key>."
                )
            return []
        # Snap each request to the nearest real frame timestamp: callers
        # sample uniform grids whose points land mid-frame, and
        # ``decode_video_frames`` rejects queries farther than
        # ``tolerance_s`` from a decodable frame. Snapping also dedupes
        # repeat queries through the cache.
        if record.frame_timestamps:
            timestamps = [snap_to_frame(float(ts), record.frame_timestamps) for ts in timestamps]

        out: list[Any] = []
        misses: list[float] = []
        miss_indices: list[int] = []
        with self._lock:
            for i, ts in enumerate(timestamps):
                key = (record.episode_index, target, round(float(ts), 6))
                cached = self._cache.get(key)
                if cached is not None:
                    out.append(cached)
                else:
                    out.append(None)
                    misses.append(float(ts))
                    miss_indices.append(i)

        if misses:
            # Parquet timestamps are the annotation clock for this episode,
            # while ``videos/<camera>/from_timestamp`` points at the start of
            # the episode inside its MP4 shard.  They normally both start at
            # zero, but imported datasets may preserve a non-zero annotation
            # origin.  Decode offsets must always be relative to the first
            # episode frame; otherwise a record spanning 10..12 s would seek
            # 10..12 s *past* its two-second source clip.
            episode_origin = float(record.frame_timestamps[0]) if record.frame_timestamps else 0.0
            decode_offsets = [ts - episode_origin for ts in misses]
            decoded = self._decode(
                record.episode_index,
                decode_offsets,
                target,
                fail_on_error=fail_on_error,
            )
            # ``_decode`` returns exactly one frame per requested timestamp,
            # or an empty list if decoding failed wholesale. Returning only
            # warm-cache survivors would discard their positions and let a
            # caller pair them with the wrong timestamps, so any incomplete
            # miss batch makes the entire request unavailable.
            if len(decoded) != len(miss_indices) or any(frame is None for frame in decoded):
                if fail_on_error:
                    raise FrameProviderError(
                        f"episode {record.episode_index}: video decoder returned "
                        f"{len(decoded)} frame(s) for {len(miss_indices)} requested timestamp(s) "
                        f"from camera {target!r}; a complete ordered frame set is required. "
                        "Check the video shard and episode timestamp metadata."
                    )
                return []
            with self._lock:
                for i, frame in zip(miss_indices, decoded, strict=True):
                    out[i] = frame
                    key = (record.episode_index, target, round(float(timestamps[i]), 6))
                    if len(self._cache) >= self.cache_size:
                        self._cache.pop(next(iter(self._cache)))
                    self._cache[key] = frame

        # Keep the positional contract explicit: callers either receive one
        # frame per requested timestamp, in order, or no frames at all.
        if any(frame is None for frame in out):
            return []
        return out

    def video_for_episode(
        self,
        record: EpisodeRecord,
        max_frames: int,
        camera_key: str | None = None,
    ) -> list[Any]:
        """Return up to ``max_frames`` frames uniformly sampled across the episode.

        The whole episode duration is covered; the model picks subtask
        boundaries from the temporal pooling it does internally. Frames are
        ``torch.Tensor`` (see :meth:`frames_at`).
        """
        target = camera_key if camera_key is not None else self.camera_key
        if max_frames <= 0 or target is None or not record.frame_timestamps:
            return []
        n_frames = min(max_frames, len(record.frame_timestamps))
        if n_frames == len(record.frame_timestamps):
            timestamps = list(record.frame_timestamps)
        else:
            t0 = record.frame_timestamps[0]
            t_last = record.frame_timestamps[-1]
            if t_last <= t0:
                timestamps = [float(t0)] * n_frames
            else:
                step = (t_last - t0) / (n_frames - 1) if n_frames > 1 else 0.0
                timestamps = [float(t0 + i * step) for i in range(n_frames)]
        return self.frames_at(record, timestamps, camera_key=target)

    # Public API: no in-repo production caller; exercised by tests/test_frames.py.
    def episode_clip_path(self, record: EpisodeRecord, cache_dir: Path) -> Path | None:
        """Extract the episode's subclip to ``cache_dir/ep_{idx:06d}.mp4``.

        Returns ``None`` if the dataset has no video tracks or extraction
        failed. Skips re-extract when the cached clip already exists.
        Re-encodes to H.264 via
        :func:`lerobot.datasets.video_utils.reencode_video` so the resulting
        mp4 is decodable by every downstream video processor — stream-copy
        would inherit the source codec (often AV1 in modern LeRobot
        datasets), which vllm's libav build cannot decode.
        """
        if self.camera_key is None:
            return None
        cache_dir.mkdir(parents=True, exist_ok=True)
        out_path = cache_dir / f"ep_{record.episode_index:06d}.mp4"
        if out_path.exists() and out_path.stat().st_size > 0:
            return out_path
        ep = self._meta.episodes[record.episode_index]
        from_timestamp = float(ep[f"videos/{self.camera_key}/from_timestamp"])
        to_timestamp = float(ep[f"videos/{self.camera_key}/to_timestamp"])
        src = self._video_path(record.episode_index, self.camera_key)
        encoder = RGBEncoderConfig(vcodec="h264", pix_fmt="yuv420p", g=None, crf=23, preset="ultrafast")
        try:
            reencode_video(
                src,
                out_path,
                video_encoder=encoder,
                overwrite=True,
                start_time_s=from_timestamp,
                end_time_s=to_timestamp,
            )
        except Exception:
            logger.warning(
                "clip extraction failed for episode %s (%s)", record.episode_index, src, exc_info=True
            )
            return None
        return out_path if out_path.exists() and out_path.stat().st_size > 0 else None

    def _video_path(self, episode_index: int, camera_key: str) -> Path:
        path = (self.root / self._meta.get_video_file_path(episode_index, camera_key)).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise FrameProviderError(
                f"episode={episode_index} camera={camera_key}: video path escapes the dataset root; "
                "refusing to read or transmit unrelated local media"
            )
        return path

    def _decode(
        self,
        episode_index: int,
        timestamps: list[float],
        camera_key: str,
        *,
        fail_on_error: bool = False,
    ) -> list[Any]:
        """Decode ``timestamps`` from the episode's video as ``(C, H, W)`` tensors.

        Delegates to :func:`lerobot.datasets.video_utils.decode_video_frames`
        (torchcodec when available, PyAV otherwise; ``video_backend`` pins
        one explicitly). Returns one frame per requested timestamp, or ``[]``
        if decoding failed — callers treat ``[]`` as "no frames available".
        """
        video_path: Path | str = f"<unresolved camera {camera_key}>"
        try:
            # Camera metadata can be globally advertised but absent for an
            # individual episode. Resolve it inside the guarded block so an
            # optional multiview camera degrades to the surviving view.
            ep = self._meta.episodes[episode_index]
            from_timestamp = ep[f"videos/{camera_key}/from_timestamp"]
            shifted = [from_timestamp + ts for ts in timestamps]
            video_path = self._video_path(episode_index, camera_key)
            # The module phases decode under a ThreadPoolExecutor (see
            # ``ExecutorConfig.episode_parallelism``) but torchcodec's cached
            # per-file decoder is single-threaded, so serialize decodes on a
            # dedicated lock. Frame extraction is a small fraction of episode
            # wall time (VLM calls dominate), so the contention is cheap.
            with self._decode_lock:
                # Stacked ``(N, C, H, W)`` uint8 tensor; one row per timestamp.
                decoded = decode_video_frames(
                    video_path, shifted, self.tolerance_s, backend=self.video_backend, return_uint8=True
                )
            return list(decoded)
        except Exception as exc:
            # Log loudly the first time so a silent vqa-module no-op (every
            # prompt skipped because frames_at returned []) is debuggable from
            # the job log instead of post-hoc parquet inspection. Subsequent
            # failures stay quiet.
            with self._lock:
                already_warned = self._warned_decode_fail
                if not already_warned:
                    self._warned_decode_fail = True
            if not already_warned:
                logger.warning(
                    "VideoFrameProvider._decode failed for episode=%s camera=%s video_path=%s backend=%s: %s",
                    episode_index,
                    camera_key,
                    video_path,
                    self.video_backend,
                    exc,
                    exc_info=exc,
                )
            if fail_on_error:
                raise FrameProviderError(
                    "video frame decode failed for "
                    f"episode={episode_index} camera={camera_key!r} video_path={video_path!s} "
                    f"backend={self.video_backend!r}: {type(exc).__name__}: {exc}. "
                    "Check the video shard, episode metadata, and filesystem or Hugging Face "
                    "cache permissions."
                ) from exc
            return []


def make_frame_provider(
    root: Path, camera_key: str | None = None, video_backend: str | None = None
) -> FrameProvider:
    """Build a :class:`VideoFrameProvider` if videos are present, else null."""
    info_path = root / "meta/info.json"
    if info_path.is_file():
        import json
        features = json.loads(info_path.read_text(encoding="utf-8")).get("features", {})
        cameras = [key for key, value in features.items() if value.get("dtype") in {"video", "image"}]
        if not cameras and camera_key is None:
            return null_provider()
    try:
        provider = VideoFrameProvider(root=root, camera_key=camera_key, video_backend=video_backend)
    except Exception as exc:
        logger.warning(
            "frame-provider initialization failed for root=%s camera=%r backend=%r: %s: %s",
            root,
            camera_key,
            video_backend,
            type(exc).__name__,
            exc,
            exc_info=exc,
        )
        raise FrameProviderError(
            f"Cannot initialize visual input for {root}: {type(exc).__name__}: {exc}. "
            "Check dataset metadata, video paths and permissions."
        ) from exc
    if provider.camera_key is None:
        raise FrameProviderError("Dataset declares cameras but no supported video camera is available")
    return provider


def _frame_to_pil(frame: Any) -> Any:
    """Materialise a decoded frame as a ``PIL.Image`` for the VLM message.

    Frames flow through the provider as ``torch.Tensor`` (``C, H, W`` uint8,
    straight from :func:`decode_video_frames`); PIL is only created here, at
    the VLM-message boundary, because the chat backends expect PIL images /
    data URLs. Non-tensor inputs (e.g. test stubs) pass through untouched.
    """
    if not isinstance(frame, torch.Tensor):
        return frame
    array = frame.detach().cpu()
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = array.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
    if array.shape[-1] == 1:
        array = array.squeeze(-1)
    return PIL.Image.fromarray(array.to(torch.uint8).numpy())


def to_image_blocks(frames: list[Any]) -> list[dict[str, Any]]:
    """Convert decoded frames to Qwen-VL-compatible image content blocks."""
    return [{"type": "image", "image": _frame_to_pil(frame)} for frame in frames]


def to_video_url_block(
    url: str | None,
    fps: float | None = 2.0,
    mm_kwargs: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Wrap a video file URL as one ``video_url`` block.

    Used by the ``openai`` backend (transformers serve / vllm serve /
    ktransformers serve), where the server handles frame sampling.
    Returns ``[]`` when ``url`` is ``None`` so the caller can splat.

    ``fps=None`` sends no sampling request at all, leaving the server to use
    every frame of the clip. Callers that have already chosen the frames — and
    encoded them so playback time carries meaning — need that: a resampling
    request would both drop frames and, on Qwen3-VL, shift the per-frame
    timestamps the model is told to read.

    ``mm_kwargs`` forwards arbitrary multimodal-processor arguments (e.g.
    ``{"num_frames": 280}``) the same way. It is the knob for the processor's
    own defaults, which are otherwise invisible: Qwen3-VL resamples to 2 fps,
    so a denser clip silently loses frames before the model sees it.
    """
    if not url:
        return []
    block: dict[str, Any] = {"type": "video_url", "video_url": {"url": url}}
    if fps is not None:
        block["fps"] = fps
    if mm_kwargs:
        block["mm_kwargs"] = dict(mm_kwargs)
    return [block]


def encode_frames_to_clip(
    frames: Sequence[Any],
    timestamps: Sequence[float],
    path: Path,
    *,
    frame_width: int = 336,
    crf: int = 0,
) -> float | None:
    """Encode sampled frames into an H.264 clip whose time is episode time.

    Video-native Qwen3-VL models carry frame time themselves, interleaving a
    literal ``<12.3 seconds>`` text token ahead of each frame's vision tokens
    rather than reading a burned-in badge out of the pixels. That only helps if
    the clip's own timeline matches the episode's, so the frame rate is derived
    from the sampled timestamps (``(N-1) / span``) instead of being chosen:
    playback second ``x`` is then episode second ``x + timestamps[0]``.

    This assumes evenly spaced ``timestamps`` — true of the uniform grid, not of
    motion-stratified sampling, which shifts representatives off the grid and
    would smear the mapping. Callers gate on that.

    Returns the encoded frame rate, or ``None`` if nothing could be encoded.
    """
    import av
    from PIL import Image

    if not frames or len(frames) != len(timestamps):
        return None

    images: list[PIL.Image.Image] = []
    for frame in frames:
        img = _frame_to_pil(frame)
        if not isinstance(img, PIL.Image.Image):
            continue
        img = img.convert("RGB")
        if img.width != frame_width:
            height = max(1, round(img.height * frame_width / img.width))
            img = img.resize((frame_width, height), resample=Image.Resampling.BILINEAR)
        images.append(img)
    if not images:
        return None

    # H.264 with yuv420p needs even dimensions, and every frame must match the
    # stream geometry, so the first frame sets it and the rest are conformed.
    width = images[0].width - (images[0].width % 2)
    height = images[0].height - (images[0].height % 2)
    if width < 2 or height < 2:
        return None
    images = [img if img.size == (width, height) else img.resize((width, height)) for img in images]

    span = float(timestamps[-1]) - float(timestamps[0])
    fps = (len(images) - 1) / span if len(images) > 1 and span > 0 else 1.0
    # A rational rate keeps libx264 and the server's decoder agreeing on the
    # timeline; the denominator bound holds the drift far below the sub-second
    # precision the boundary metrics are scored at.
    rate = Fraction(fps).limit_denominator(10000)

    container = av.open(str(path), mode="w")
    try:
        stream = container.add_stream("libx264", rate=rate)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.codec_context.time_base = Fraction(1, 1) / rate
        # Lossless (``crf=0``) so the decoded pixels are reproducible.
        #
        # libx264 is not deterministic here: repeat encodes of identical frames
        # differ, and not only in the container — two lossy encodes of one
        # episode decoded to pixels differing by up to 48 levels across 139 of
        # 280 frames. The model is sensitive enough to that to move a boundary
        # by seconds, and the same episode scored 40.6% / 74.3% / 84.0%
        # temporal IoU across three runs at temperature 0. Pinning threads
        # reduced but did not remove it.
        #
        # Lossless fixes it by construction rather than by tuning: the encoder
        # may still choose a different bitstream, but a lossless decode
        # reconstructs the input exactly either way, so the frames the model
        # sees are identical. Costs about 5x the file size (2.3 MB -> 11.9 MB
        # for a 280-frame episode), which is transient local data.
        stream.codec_context.thread_count = 1
        stream.codec_context.thread_type = "NONE"
        stream.options = {"crf": str(crf), "x264-params": "threads=1:sliced-threads=0"}
        for index, img in enumerate(images):
            frame = av.VideoFrame.from_image(img)
            frame.pts = index
            frame.time_base = stream.codec_context.time_base
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
    return float(rate)


def _draw_timestamp_badge(image: PIL.Image.Image, timestamp: float) -> PIL.Image.Image:
    """Burn ``timestamp`` (seconds) into the top-left corner of ``image``.

    A solid black badge with white text, so a VLM reading a contact sheet can
    cite the exact source time of each tile (e.g. ``012.50s``) directly,
    instead of the caller having to map tile position back to time. Mirrors
    the macrodata/refiner contact-sheet convention.
    """
    from PIL import ImageDraw, ImageFont

    result = image.copy()
    draw = ImageDraw.Draw(result)
    # Scale the timestamp to the tile so it stays legible after the model
    # downsamples the full sheet into 768px tiles — a tiny bitmap font blurs
    # at contact-sheet resolution and the VLM can no longer read the exact
    # source time, which is what the boundary score depends on. ``size=`` is
    # supported by Pillow's bitmap default since 10.1; fall back otherwise.
    badge_px = max(14, round(image.height * 0.12))
    try:
        font = ImageFont.load_default(size=badge_px)
    except TypeError:
        font = ImageFont.load_default()
    label = f"{timestamp:06.2f}s"
    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
    text_w, text_h = right - left, bottom - top
    pad = max(3, round(min(image.width, image.height) * 0.018))
    draw.rectangle((0, 0, text_w + pad * 2, text_h + pad * 2), fill=(0, 0, 0))
    draw.text((pad - left, pad - top), label, fill=(255, 255, 255), font=font)
    return result


def to_contact_sheet_blocks(
    frames: Sequence[Any],
    timestamps: Sequence[float],
    *,
    columns: int = 5,
    frames_per_sheet: int = 20,
    frame_width: int = 224,
    quality: int = 84,
) -> list[dict[str, Any]]:
    """Pack decoded frames into timestamped JPEG contact-sheet image blocks.

    Each frame is resized to ``frame_width`` wide, stamped with its
    episode-relative timestamp, and tiled row-major into grids of
    ``frames_per_sheet`` (``columns`` wide). One ``{"type":"image", ...}``
    block is returned per grid; many frames collapse into a few images, so a
    long episode's temporal coverage stays dense at a fraction of the vision
    tokens N separate frames would cost. ``frames`` and ``timestamps`` must be
    aligned and equal length. Returns ``[]`` for empty input.
    """
    from PIL import Image

    if not frames:
        return []
    columns = max(1, columns)
    frames_per_sheet = max(1, frames_per_sheet)
    rows_per_sheet = math.ceil(frames_per_sheet / columns)

    tiles: list[PIL.Image.Image] = []
    for ts, frame in zip(timestamps, frames, strict=False):
        img = _frame_to_pil(frame)
        if not isinstance(img, PIL.Image.Image):
            continue
        img = img.convert("RGB")
        if img.width != frame_width:
            height = max(1, round(img.height * frame_width / img.width))
            img = img.resize((frame_width, height), resample=Image.Resampling.BILINEAR)
        tiles.append(_draw_timestamp_badge(img, float(ts)))
    if not tiles:
        return []

    blocks: list[dict[str, Any]] = []
    for start in range(0, len(tiles), frames_per_sheet):
        chunk = tiles[start : start + frames_per_sheet]
        cell_w = max(tile.width for tile in chunk)
        cell_h = max(tile.height for tile in chunk)
        sheet = Image.new("RGB", (cell_w * columns, cell_h * rows_per_sheet), color=(0, 0, 0))
        for i, tile in enumerate(chunk):
            x = (i % columns) * cell_w
            y = (i // columns) * cell_h
            sheet.paste(tile, (x, y))
        # JPEG round-trip at ``quality`` to match the refiner convention and
        # shrink the wire payload; vision-token count is set by resolution, so
        # the real saving is the grid packing, not the codec.
        buf = io.BytesIO()
        sheet.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        blocks.append({"type": "image", "image": Image.open(buf).convert("RGB")})
    return blocks


def _draw_view_badge(image: PIL.Image.Image, label: str) -> PIL.Image.Image:
    """Burn a compact view identifier into the top-right of one camera frame."""
    from PIL import ImageDraw, ImageFont

    result = image.copy()
    draw = ImageDraw.Draw(result)
    badge_px = max(12, round(image.height * 0.09))
    try:
        font = ImageFont.load_default(size=badge_px)
    except TypeError:
        font = ImageFont.load_default()
    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
    text_w, text_h = right - left, bottom - top
    pad = max(3, round(min(image.width, image.height) * 0.018))
    x0 = max(0, image.width - text_w - pad * 2)
    draw.rectangle((x0, 0, image.width, text_h + pad * 2), fill=(0, 0, 0))
    draw.text((x0 + pad - left, pad - top), label, fill=(255, 255, 255), font=font)
    return result


def to_stacked_view_frames(
    camera_frames: Sequence[tuple[str, Sequence[Any]]],
    count: int,
    *,
    frame_width: int = 224,
) -> list[PIL.Image.Image]:
    """Vertically stack synchronized camera views into one image per timestamp.

    Each view is labeled ``VIEW 1``, ``VIEW 2``, and so on; the calling prompt
    maps those stable aliases to dataset camera keys. Composing the views for a
    moment into a single image is what stops the model from being handed several
    full timelines it could read end-to-end instead of side-by-side.

    Returns ``[]`` unless every view supplies exactly ``count`` decodable
    frames, so a partial view can never shift another view's timestamps.
    """
    from PIL import Image

    if not camera_frames or count <= 0:
        return []
    if any(len(frames) != count for _camera_key, frames in camera_frames):
        return []

    composites: list[PIL.Image.Image] = []
    for frame_index in range(count):
        views: list[PIL.Image.Image] = []
        for view_index, (_camera_key, frames) in enumerate(camera_frames, start=1):
            image = _frame_to_pil(frames[frame_index])
            if not isinstance(image, PIL.Image.Image):
                return []
            image = image.convert("RGB")
            if image.width != frame_width:
                height = max(1, round(image.height * frame_width / image.width))
                image = image.resize((frame_width, height), resample=Image.Resampling.BILINEAR)
            views.append(_draw_view_badge(image, f"VIEW {view_index}"))

        composite = Image.new(
            "RGB",
            (max(view.width for view in views), sum(view.height for view in views)),
            color=(0, 0, 0),
        )
        y = 0
        for view in views:
            composite.paste(view, (0, y))
            y += view.height
        composites.append(composite)
    return composites


def to_multiview_contact_sheet_blocks(
    camera_frames: Sequence[tuple[str, Sequence[Any]]],
    timestamps: Sequence[float],
    *,
    columns: int = 5,
    frames_per_sheet: int = 20,
    frame_width: int = 224,
    quality: int = 84,
) -> list[dict[str, Any]]:
    """Pack synchronized camera views into one chronological tile sequence.

    Every timestamp becomes one vertically stacked composite labeled ``VIEW 1``,
    ``VIEW 2``, and so on. The calling prompt maps those stable aliases to the
    dataset camera keys. Keeping all views for a timestamp inside one tile avoids
    presenting the VLM with several full timelines that it could concatenate.
    """
    composites = to_stacked_view_frames(camera_frames, len(timestamps), frame_width=frame_width)
    if not composites:
        return []
    return to_contact_sheet_blocks(
        composites,
        timestamps,
        columns=columns,
        frames_per_sheet=frames_per_sheet,
        frame_width=frame_width,
        quality=quality,
    )
