#!/usr/bin/env python
"""Run one (dataset, arm) job and record its predictions plus full provenance.

One process annotates one dataset under one arm against one vLLM endpoint.
Parallelism across GPUs is achieved by launching several of these, never by
threading inside one, so that a crash loses at most one job and the per-job
timing numbers stay clean enough to compare.

Caching
-------
The unit of caching is the (dataset, arm) pair. If the output record already
exists and its ``config_fingerprint`` matches what this invocation would
produce, the job is skipped.

The fingerprint covers the tool, every flag, the model id, the seed, the
episode split, and a **content hash of both tools' annotation source**. The
content hash is the important part: keying on the git commit alone means an
uncommitted edit leaves HEAD unchanged, the cache hits, and predictions
produced by different code are served as if current -- which a downstream C1
verdict would then certify while recording the *new* source hash. Hashing the
tree closes that loop, and nothing that cannot alter the output is included, so
re-running after an unrelated edit is still free.

Files supplied on the command line -- the timing calibration and, for
fixed-label alignment, the ordered label list -- are hashed by CONTENT, never
by path. Label files are regenerated in place during the alignment study, so a
path-keyed entry would serve predictions produced against the *previous*
labels while nothing else in the key moved.

Isolation
---------
Each job gets its own working root, because both tools rewrite the dataset's
parquet files in place. Roots are cheap: videos are symlinked by
``prepare_dataset.py`` and only the parquet is materialised.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import urllib.request
import sys
import time
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gate_fingerprint import source_hashes  # noqa: E402
from prepare_dataset import build_working_root, extract_ground_truth, verify_clean  # noqa: E402
from calibration_status import STATUS_VERSION, run_calibration_status  # noqa: E402

UPSTREAM_MODULE = "lerobot.scripts.lerobot_annotate"
ALIGN_MODULE = "lerobot_align.cli"


def git_commit(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def package_version(name: str, python: str) -> str:
    try:
        return subprocess.check_output(
            [python, "-c", f"import {name} as m; print(getattr(m,'__version__','unknown'))"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def content_digest(path: str | Path, option: str = "supplied file") -> str:
    """Short SHA-256 of a file supplied on the command line.

    The cache key must move when the file's CONTENT moves, not when its name
    does. Both supplied files are rewritten in place by their generators --
    ``lerobot-align-fit`` for the calibration, ``make_label_files.py`` for the
    labels -- so a path-keyed entry silently serves predictions made against
    the previous version of a file that still has the same name.
    """
    file = Path(path)
    if not file.is_file():
        raise SystemExit(f"{option} does not exist: {file}")
    return hashlib.sha256(file.read_bytes()).hexdigest()[:16]


def config_fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


VLLM_TOKEN_METRICS = ("vllm:prompt_tokens_total", "vllm:generation_tokens_total")


def replica_token_counters(base_url: str) -> dict[str, float] | None:
    """Cumulative token counters for the replica this job will use.

    The dispatcher gives each job exclusive use of one replica for its whole
    duration, so the difference between a reading taken before the job and one
    taken after is exactly that job's token consumption. Returns ``None`` when
    the endpoint is unreachable or unparseable, so a missing reading is
    recorded as unknown rather than silently as zero.
    """
    root = base_url.rstrip("/")
    for suffix in ("/v1", "/v1/"):
        if root.endswith(suffix):
            root = root[: -len(suffix)]
            break
    try:
        with urllib.request.urlopen(f"{root}/metrics", timeout=10) as response:
            body = response.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - instrumentation must never fail a job
        return None
    out: dict[str, float] = {}
    for line in body.splitlines():
        if line.startswith("#"):
            continue
        for metric in VLLM_TOKEN_METRICS:
            if line.startswith(metric + "{"):
                with contextlib.suppress(IndexError, ValueError):
                    out[metric] = float(line.rsplit(" ", 1)[1])
    return out or None


def build_command(
    *, tool: str, python: str, root: Path, flags: dict[str, Any], model_id: str, base_url: str
) -> list[str]:
    """Assemble the CLI for either tool.

    Both accept the same draccus-style ``--section.field=value`` arguments, so
    the arm's flag dict is rendered identically for both. That is what makes
    the equalised arms genuinely equalised rather than approximately so.
    """
    module = UPSTREAM_MODULE if tool == "upstream" else ALIGN_MODULE
    command = [python, "-m", module, f"--root={root}"]
    for key, value in sorted(flags.items()):
        if value is None:
            continue
        if isinstance(value, bool):
            value = "true" if value else "false"
        command.append(f"--{key}={value}")
    command.append(f"--vlm.model_id={model_id}")
    # The field is `vlm.api_base`, not `vlm.base_url`; draccus rejects unknown
    # keys, so the wrong name does not degrade to a default -- it aborts the run.
    command.append(f"--vlm.api_base={base_url}")
    command.append("--vlm.auto_serve=false")
    command.append("--push_to_hub=false")
    return command


def extract_predictions(root: Path) -> dict[int, list[dict[str, Any]]]:
    """Read back the subtask spans the tool wrote.

    Deliberately uses the same reader as ground-truth extraction, so the two
    sides of every comparison are parsed by identical code and a parsing quirk
    cannot masquerade as a quality difference.
    """
    return extract_ground_truth(sorted((root / "data").rglob("*.parquet")))


def episode_selection_flag(episodes: list[int]) -> str:
    """Render the evaluation split for ``--only_episodes``.

    Both tools accept ``only_episodes: tuple[int, ...]`` natively, which is
    strictly better than filtering the parquet ourselves: the dataset stays
    internally consistent (episode metadata, chunk indices and video references
    all keep matching), and the tools do the selection with the same code they
    would use in production. Rewriting the parquet to drop episodes risked
    changing what the tools saw in ways unrelated to the arms.
    """
    return "[" + ",".join(str(int(e)) for e in sorted(episodes)) + "]"


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--arms-config", type=Path, default=HERE.parent / "configs" / "arms.yaml")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="predictions JSON to write")
    parser.add_argument("--episodes", type=Path, default=None, help="JSON split file; runs the eval list")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--calibration-path", default=None,
        help="Timing calibration JSON. Substituted into every flag containing the "
             "{calibration_path} placeholder and hashed by content into the cache key.",
    )
    parser.add_argument(
        "--subtasks-path", default=None,
        help="Ordered label list JSON for fixed-label alignment. Substituted into every "
             "flag containing the {subtasks_path} placeholder and hashed by content into "
             "the cache key; refused when no flag consumes it, because the arm would then "
             "silently generate its own labels.",
    )
    parser.add_argument(
        "--repeat", type=int, default=0,
        help="Repeat index. Neither tool seeds its generation call, so repeats of the "
             "same arm differ; the index keeps their caches and outputs separate.",
    )
    parser.add_argument("--timeout", type=int, default=14400)
    parser.add_argument("--force", action="store_true", help="ignore an existing cached result")
    parser.add_argument("--keep-root", action="store_true")
    args = parser.parse_args()

    import yaml

    spec = yaml.safe_load(args.arms_config.read_text(encoding="utf-8"))
    arms = {a["name"]: a for a in spec["arms"]}
    if args.arm not in arms:
        raise SystemExit(f"unknown arm {args.arm}; known: {sorted(arms)}")
    arm = arms[args.arm]
    if arm["tool"] == "reference":
        raise SystemExit(
            f"{args.arm} is a model-free arm; use reference_arms.py (generation) or "
            "align_floors.py (fixed-label alignment)"
        )

    flags = dict(arm.get("flags") or {})
    # Which supplied files the arm's flags actually reference. Tracked rather
    # than assumed: a file no flag consumes cannot change the run, but it would
    # still change the cache key and sit in the provenance as though it had.
    consumed: set[str] = set()
    for key, value in list(flags.items()):
        if not isinstance(value, str):
            continue
        rendered = value
        if "{calibration_path}" in rendered:
            if not args.calibration_path:
                raise SystemExit(f"{args.arm} needs --calibration-path")
            rendered = rendered.replace("{calibration_path}", args.calibration_path)
            consumed.add("calibration_path")
        if "{subtasks_path}" in rendered:
            if not args.subtasks_path:
                raise SystemExit(f"{args.arm} needs --subtasks-path")
            rendered = rendered.replace("{subtasks_path}", args.subtasks_path)
            consumed.add("subtasks_path")
        flags[key] = rendered

    # An unconsumed label file is not a harmless extra argument. The arm then
    # runs in GENERATION mode -- the tool invents its own labels -- while the
    # cache key and the provenance both claim the supplied list was in use, and
    # the fixed-label study silently measures something else. Refuse.
    if args.subtasks_path and "subtasks_path" not in consumed:
        raise SystemExit(
            f"--subtasks-path={args.subtasks_path} was passed for arm {args.arm}, but none "
            "of its flags contain the {subtasks_path} placeholder, so the labels would "
            "never reach the tool and the arm would GENERATE labels instead of aligning "
            "the supplied ones. Give the arm a plan.subtasks_path flag, or drop the option."
        )
    if args.calibration_path and "calibration_path" not in consumed:
        # Deliberately a warning where the label file is an error: the sweep
        # resolves one calibration file per component and offers it to every arm
        # of that component, only some of which are calibrated. Ignoring it is
        # correct; recording it as if it had applied is not, so the digest below
        # is taken only for a file some flag consumed.
        print(
            f"[run] note: {args.arm} consumes no calibration file; ignoring "
            f"--calibration-path={args.calibration_path}",
            file=sys.stderr,
        )

    episodes: list[int] | None = None
    if args.episodes:
        split = json.loads(args.episodes.read_text(encoding="utf-8"))
        episodes = [int(e) for e in split["eval"]]

    align_repo = HERE.parent.parent
    calibration_digest = None
    if "calibration_path" in consumed and Path(args.calibration_path).exists():
        calibration_digest = content_digest(args.calibration_path, "--calibration-path")
    # The label file decides which labels the VLM is asked to place, so it is
    # part of the run. Hashed for the same reason the calibration is, and it
    # must exist now rather than being discovered missing after a working root
    # has been built and a subprocess launched.
    subtasks_digest = (
        content_digest(args.subtasks_path, "--subtasks-path") if args.subtasks_path else None
    )
    provenance = {
        "dataset": args.dataset,
        "arm": args.arm,
        "tool": arm["tool"],
        "flags": flags,
        "model_id": args.model_id,
        "align_commit": git_commit(align_repo),
        "lerobot_version": package_version("lerobot", args.python),
        # Git HEAD is NOT sufficient. An uncommitted edit to either tool leaves
        # HEAD unchanged, so the cache would serve predictions produced by
        # different code -- and a later C1 verdict would then certify those
        # outputs while claiming the current source. Content hashes close that.
        **source_hashes(),
        "supervision": arm["supervision"],
        "equalised": arm.get("equalised"),
        # Everything below can change the output and must therefore invalidate
        # the cache. Omitting the episode list was the worst of these: a rerun
        # with a different split would have silently served the old split's
        # predictions.
        "repeat": args.repeat,
        "source_root": str(args.source_root),
        "episodes": sorted(episodes) if episodes is not None else None,
        # Both supplied files enter the key by CONTENT. The path is not the
        # key: editing a label file in place -- routine here, the generators
        # rewrite the same filenames -- leaves the path identical, and a
        # path-keyed entry would serve predictions made against the old labels.
        "calibration_sha": calibration_digest,
        "calibration_status_version": STATUS_VERSION,
        "subtasks_sha": subtasks_digest,
    }
    fingerprint = config_fingerprint(provenance)

    if args.out.exists() and not args.force:  # noqa: E501 - see episode parse above
        try:
            existing = json.loads(args.out.read_text(encoding="utf-8"))
            if existing.get("config_fingerprint") == fingerprint and existing.get("ok"):
                print(f"[run] cached: {args.dataset} {args.arm} ({fingerprint})")
                return 0
            print(f"[run] stale cache for {args.dataset} {args.arm}, re-running")
        except Exception:
            pass

    root = args.work_dir / f"{args.dataset}__{args.arm}__r{args.repeat}"
    root.parent.mkdir(parents=True, exist_ok=True)
    build_working_root(args.source_root, root, link_videos=True, overwrite=True)
    if episodes is not None:
        flags["only_episodes"] = episode_selection_flag(episodes)

    leaks = verify_clean(root)
    if leaks:
        raise SystemExit(f"working root leaks annotation state: {leaks}")

    command = build_command(
        tool=arm["tool"], python=args.python, root=root,
        flags=flags, model_id=args.model_id, base_url=args.base_url,
    )
    log_path = args.out.with_suffix(".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[run] {args.dataset} {args.arm}: {' '.join(command)}")

    tokens_before = replica_token_counters(args.base_url)
    started = time.time()
    environment = dict(os.environ)
    # Required for the native-video path; harmless for every other arm and for
    # upstream, whose client ignores it. Set unconditionally so that no arm can
    # differ by whether an operator remembered to export it.
    environment["LEROBOT_OPENAI_SEND_MM_KWARGS"] = "1"
    timed_out = False
    with log_path.open("w", encoding="utf-8") as handle:
        try:
            process = subprocess.run(
                command, stdout=handle, stderr=subprocess.STDOUT,
                timeout=args.timeout, env=environment, check=False,
            )
            returncode = process.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = 124
            handle.write(f"\nTimeoutExpired: job exceeded {args.timeout} seconds\n")
    elapsed = time.time() - started
    tokens_after = replica_token_counters(args.base_url)
    if tokens_before and tokens_after:
        prompt_tokens = int(
            tokens_after.get('vllm:prompt_tokens_total', 0)
            - tokens_before.get('vllm:prompt_tokens_total', 0)
        )
        generation_tokens = int(
            tokens_after.get('vllm:generation_tokens_total', 0)
            - tokens_before.get('vllm:generation_tokens_total', 0)
        )
    else:
        prompt_tokens = generation_tokens = None

    ok = returncode == 0
    error = f"job exceeded {args.timeout} seconds" if timed_out else None
    predictions: dict[str, list[dict[str, Any]]] = {}
    if ok:
        try:
            predictions = {str(k): v for k, v in sorted(extract_predictions(root).items())}
        except Exception as exc:  # noqa: BLE001
            ok = False
            error = f"prediction extraction failed: {exc}"
            print(f"[run] prediction extraction failed: {exc}", file=sys.stderr)

    # Parameters are loaded before inference and must remain the same artifact
    # throughout the run. Runtime decisions, not cohort eligibility, establish
    # which delivered predictions actually used that artifact.
    if calibration_digest:
        try:
            unchanged = hashlib.sha256(Path(args.calibration_path).read_bytes()).hexdigest()[:16] == calibration_digest
        except OSError:
            unchanged = False
        if not unchanged:
            ok, error = False, "calibration file changed or became unreadable during inference"
    try:
        statuses = run_calibration_status(
            log_path.read_text(encoding="utf-8"),
            episodes,
            predictions, configured=calibration_digest is not None, job_ok=ok,
        )
    except (ValueError, TypeError) as exc:
        ok, error = False, f"invalid calibration instrumentation: {exc}"
        statuses = {}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                **provenance,
                "config_fingerprint": fingerprint,
                "repeat": args.repeat,
                "ok": ok,
                "returncode": returncode,
                "timed_out": timed_out,
                "error": error,
                "elapsed_seconds": elapsed,
                # Cost, measured not estimated. The dispatcher gives each job
                # exclusive use of one replica, so the delta in that replica's
                # cumulative counters is this job's consumption. ``None`` means
                # the endpoint was unreachable -- never silently zero.
                "prompt_tokens": prompt_tokens,
                "generation_tokens": generation_tokens,
                "total_tokens": (
                    None if prompt_tokens is None or generation_tokens is None
                    else prompt_tokens + generation_tokens
                ),
                "n_episodes_requested": len(episodes) if episodes is not None else None,
                "n_episodes_predicted": len(predictions),
                "command": command,
                "log": str(log_path),
                "episodes": predictions,
                "calibration_status": statuses,
                "calibration_applied": {k: s["applied"] for k, s in statuses.items()},
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(
        f"[run] {args.dataset} {args.arm}: ok={ok} rc={returncode} "
        f"{elapsed:.1f}s episodes={len(predictions)} "
        f"tokens={'?' if prompt_tokens is None else prompt_tokens + generation_tokens} "
        f"-> {args.out}"
    )

    if not args.keep_root:
        with contextlib.suppress(Exception):
            shutil.rmtree(root)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
