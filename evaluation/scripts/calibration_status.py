"""Read runtime calibration decisions without inferring them from routing or spans."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

STATUS_VERSION = 1
MARKER = "ALIGN_CALIBRATION "


def parse_calibration_status(log: str) -> dict[str, dict[str, Any]]:
    """Parse one atomic decision per episode; refuse ambiguous instrumentation."""
    statuses: dict[str, dict[str, Any]] = {}
    for line in log.splitlines():
        if MARKER not in line:
            continue
        raw = json.loads(line.split(MARKER, 1)[1])
        if not isinstance(raw, dict):
            raise ValueError("calibration status must be an object")
        episode = raw.get("episode")
        if type(episode) is not int or episode < 0:
            raise ValueError("calibration status requires a non-negative integer episode")
        if type(raw.get("applied")) is not bool or not isinstance(raw.get("reason"), str):
            raise ValueError("calibration status requires boolean applied and a reason")
        if raw.get("method") not in {None, "offsets", "duration_prior"}:
            raise ValueError("unknown calibration method in runtime status")
        key = str(episode)
        if key in statuses:
            raise ValueError(f"duplicate calibration decisions for episode {episode}")
        statuses[key] = {**raw, "source": "runtime"}
    return statuses


def run_calibration_status(
    log: str,
    episodes: Iterable[int] | None,
    predictions: dict[str, list[dict[str, Any]]],
    *,
    configured: bool,
    job_ok: bool,
) -> dict[str, dict[str, Any]]:
    """Require an observed decision for every delivered calibrated prediction.

    No-output and failed episodes are recorded separately from a runtime skip.
    A changed answer alone is never evidence that calibration ran.
    """
    observed = parse_calibration_status(log)
    expected = ({str(int(e)) for e in episodes} if episodes is not None
                else set(observed) | set(predictions))
    if set(observed) - expected:
        raise ValueError("runtime calibration status contains an unrequested episode")
    statuses: dict[str, dict[str, Any]] = {}
    for key in sorted(expected, key=int):
        status = observed.get(key)
        if status and status["applied"] and not configured:
            raise ValueError(f"episode {key}: calibration applied without a calibration file")
        if not job_ok or not predictions.get(key):
            statuses[key] = {
                "episode": int(key), "applied": False,
                "reason": "job_failed" if not job_ok else "no_output", "method": None,
                "runtime_decision": status, "source": "delivery",
            }
        elif status is not None:
            statuses[key] = status
        elif configured:
            raise ValueError(f"episode {key}: calibrated output has no runtime decision")
        else:
            statuses[key] = {
                "episode": int(key), "applied": False,
                "reason": "not_configured", "method": None, "source": "configuration",
            }
    return statuses
