"""Resolve which camera a diagnostic reads, from the dataset itself.

These diagnostics used to default ``--camera`` to ``observation.images.wrist``,
the wrist key of one particular fridge dataset. Most LeRobot datasets do not
carry that key, so the default named a camera that does not exist. ``--camera``
now defaults to ``None`` and is resolved here: the dataset's first decodable
video camera, exactly as :func:`lerobot_align.frames.make_frame_provider`
resolves it from ``meta/info.json``, or an error naming the keys the dataset
actually has.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from lerobot_align.frames import make_frame_provider


def available_camera_keys(root: Path) -> list[str]:
    """The decodable video camera keys of the dataset at ``root``.

    Empty for genuinely camera-less metadata. Initialization failures propagate
    with their cause instead of appearing as an empty camera list.
    """
    return list(make_frame_provider(root).camera_keys)


def resolve_camera_keys(root: Path, requested: Sequence[str] | None) -> list[str]:
    """The camera keys to read, given whatever ``--camera`` supplied.

    ``requested`` empty or ``None`` means "whatever this dataset offers", which
    resolves to its first decodable video camera. A requested key is trusted
    when the dataset surfaced no keys of its own (the caller may know better
    than the metadata, as ``VideoFrameProvider`` itself assumes); otherwise it
    must be one the dataset has.

    Raises ``SystemExit`` with the dataset's own key list when the request
    cannot be met.
    """
    available = available_camera_keys(root)
    if requested:
        unknown = [key for key in requested if key not in available]
        if unknown and available:
            raise SystemExit(
                f"{root}: --camera {unknown} is not a decodable video feature of this dataset. "
                f"Available camera keys: {available}."
            )
        return list(requested)
    if not available:
        raise SystemExit(
            f"{root}: no decodable video camera was found, so --camera cannot be defaulted. "
            "Check that meta/info.json declares a video-backed observation.images.* feature "
            "(depth cameras are excluded), or name one with --camera."
        )
    return [available[0]]


def resolve_camera_key(root: Path, requested: str | None) -> str:
    """Single-camera form of :func:`resolve_camera_keys`."""
    return resolve_camera_keys(root, [requested] if requested else None)[0]


CAMERA_HELP = "camera key (default: the dataset's first decodable video camera)"
MULTI_CAMERA_HELP = (
    "camera key; repeat for synchronized multiview "
    "(default: the dataset's first decodable video camera)"
)
