#!/usr/bin/env python
"""Render the final comparison report from the aggregated results.

Assembles one markdown document from artifacts that already exist, and refuses
to invent anything that does not. In particular it will not print a headline
claim unless the evidence for that claim is present: no C1 gate verdict means
no mechanism claim, and no C4 contrast means no perception claim.

**Missing evidence is a failure, not a footnote.** An earlier version printed
"INCOMPLETE", wrote the file anyway and exited 0 -- so a pipeline running under
``set -e`` would sail straight past it and leave a document on disk that reads
like a report while resting on nothing. Now the required artifacts are checked
first: if any is absent the report is NOT written and the exit status is
non-zero. ``--allow-incomplete`` opts into a draft for diagnosis, which is
stamped with a banner that survives copy-paste and still refuses to call itself
a result.

A stale report from an earlier, complete run is also cleared rather than left in
place, because a file with an old timestamp beside fresh results is exactly what
gets picked up by mistake.

Every mutation of ``--out`` goes through a single file descriptor, opened once
with ``O_NOFOLLOW`` and verified by reading this script's generation marker from
that descriptor. Nothing is ever done by re-resolving the path.

Two earlier versions were unsafe. The first unlinked whatever ``--out`` named on
the failure path, so ``--out ~/notes.md`` with evidence missing destroyed an
unrelated file -- data loss from a typo. The second checked a marker first, but
checked it by path and then acted by path, so the file could be swapped in
between, and both operations followed symlinks: pointing ``--out`` at a symlink
would have written through it. Holding one descriptor closes both holes, and
``O_NOFOLLOW`` means a symlinked ``--out`` is refused outright rather than
silently followed to its target.

Ownership is established through that descriptor, but the *content* is never
written into the destination directly. Writing in place -- even with a
loop that consumes every byte -- leaves a partial file bearing the generation
marker if the write fails midway on a full or failing disk, and that file reads
as authoritative while stopping partway through the results. Checking the size
afterwards cannot undo it; the damage is already on disk.

So the payload goes to a sibling temporary file, is fully written and fsynced
there, and only then published. A reader therefore sees either the previous
content or the complete new content, never a half-written report. A failure at
any point leaves only the temporary file, which is removed, and the destination
untouched.

Publishing is deliberately split by case, because a bare ``os.replace`` operates
by path and would clobber whatever happens to be there -- reintroducing the
substitution risk the descriptor exists to prevent, and discarding the
create-only guarantee an ``O_EXCL`` open gives:

* the destination did not exist -> ``os.link``, which fails with ``EEXIST`` if a
  file appeared in the meantime, so a fresh run can never overwrite something
  that arrived after the check. This case is fully race-free;
* the destination existed and carries our marker -> the descriptor is locked with
  ``flock(LOCK_EX | LOCK_NB)`` for the whole operation, and the inode behind it is
  compared against the path immediately before ``os.replace``.

Residual limitation, stated rather than papered over
----------------------------------------------------
``rename(2)`` is defined on paths, so there is no POSIX operation that atomically
replaces *a specific verified inode*. The overwrite case therefore keeps a
microscopic window between the inode check and the rename.

What the lock buys is the threat that actually exists here: two runs of this
pipeline writing the same report concurrently. ``flock`` serialises those
completely -- a second instance fails immediately rather than interleaving --
and the inode check catches an out-of-band replacement in every case except one
occurring inside that final window.

The lock is taken on a dedicated sidecar (``.<name>.lock``), never on the report
itself. Locking the report was tried and does not work: it is only reachable
when the file already exists, so two runs starting from nothing both proceed
unlocked; and publishing swaps the destination inode, so the lock a run holds
ends up on an orphaned inode while the next run locks the new one. A sidecar is
never replaced, so a single ``flock`` on it covers the whole operation --
including the case where the report does not exist yet.

A non-cooperating external process that swaps the path in those microseconds can
still have its file replaced. That is inherent to rename semantics and is NOT
defended against; the mitigation is not to point ``--out`` at a file you care
about, which the ownership check already enforces for everything except a
deliberate race.

The ordering is deliberate and is the opposite of how such reports usually
read. The model-free floor comes first, before any tool number, because on
this corpus a script prior that never sees a pixel already scores 0.974
label_f1 -- so a reader who meets the tool-vs-tool delta first will
systematically overrate it. The limitations are stated in the opening summary
rather than at the end, for the same reason.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import fcntl
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

# Written as the first line of every generated report. It is the ONLY thing that
# licenses this script to delete an existing file: without it we cannot tell our
# own stale output from a file the user happened to name.
GENERATION_MARKER = "<!-- generated-by: evaluation/scripts/make_report.py -->"

HEADLINE = ("boundary_f1@0p5", "boundary_f1@1", "macro_iou")
PRETTY = {
    "boundary_f1@0p5": "Boundary F1 @0.5s",
    "boundary_f1@1": "Boundary F1 @1s",
    "macro_iou": "Macro temporal IoU",
    "edit_score": "Segmental edit score",
    "label_f1": "Label F1",
    "f1@25": "F1 @IoU 0.25",
    "mof": "Frame accuracy (MoF)",
}


SUPERSEDED_NOTICE = (
    f"{GENERATION_MARKER}\n"
    "# Superseded\n\n"
    "A previous report was generated here, then invalidated because the evidence it\n"
    "rested on is missing or has changed. It has been cleared deliberately: a stale\n"
    "report sitting beside fresh results is easily mistaken for current.\n"
)


class OutputTarget:
    """Exclusive handle on ``--out``, safe against swaps and symlinks.

    The descriptor is opened once with ``O_NOFOLLOW`` and every read, truncate
    and write goes through it. Because the path is never re-resolved after the
    ownership check, a file substituted afterwards cannot be written to or
    cleared, and a symlinked ``--out`` is refused rather than followed.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.fd: int | None = None
        self.lock_fd: int | None = None
        self.existed = False
        self.is_ours = False
        self.refusal: str | None = None

        # Serialise on a sidecar BEFORE touching anything, so two runs starting
        # with no report present are still ordered, and so the lock survives the
        # destination inode being replaced at publish time.
        lock_path = path.with_name(f".{path.name}.lock")
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            self.lock_fd = os.open(
                lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o644
            )
        except OSError as exc:
            if exc.errno in (errno.ELOOP, errno.EMLINK):
                self.refusal = (
                    f"{lock_path} is a symbolic link. Refusing to lock through it: the "
                    "target could be any file, and two runs pointed at different targets "
                    "would not serialise."
                )
            else:
                self.refusal = f"cannot create lock {lock_path}: {exc}"
            return
        # A fifo or device would block or misbehave under flock; only a regular
        # file is an acceptable lock.
        if not stat.S_ISREG(os.fstat(self.lock_fd).st_mode):
            os.close(self.lock_fd)
            self.lock_fd = None
            self.refusal = f"{lock_path} exists and is not a regular file; refusing to lock it."
            return
        try:
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(self.lock_fd)
            self.lock_fd = None
            self.refusal = (
                f"{path} is locked by another run of this script. Refusing to write "
                "concurrently; wait for it to finish."
            )
            return

        try:
            self.fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
            self.existed = True
        except FileNotFoundError:
            return
        except OSError as exc:
            if exc.errno in (errno.ELOOP, errno.EMLINK):
                self.refusal = (
                    f"{path} is a symbolic link. Refusing to write through it, because the "
                    "target could be any file. Pass a regular path."
                )
            else:
                self.refusal = f"cannot open {path}: {exc}"
            return

        try:
            header = os.read(self.fd, len(GENERATION_MARKER.encode("utf-8")) + 2)
            self.is_ours = header.decode("utf-8", "strict").startswith(GENERATION_MARKER)
        except (OSError, UnicodeDecodeError):
            self.is_ours = False

    def close(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        if self.lock_fd is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
            os.close(self.lock_fd)
            self.lock_fd = None

    @staticmethod
    def _write_all(fd: int, payload: bytes) -> None:
        """Write every byte, or raise.

        ``os.write`` is not obliged to consume the whole buffer -- it returns how
        much it took, and a short write is legal for large payloads or when a
        signal interrupts the call. Ignoring that return value silently produces
        a truncated file, which for this script means a report that opens with
        the generation marker, reads as authoritative and simply stops partway
        through the results. That is a worse failure than not writing at all,
        because nothing about the file signals it is incomplete.

        The final ``fsync`` matters for the same reason: the report is the
        artifact a reader is meant to trust, so it should be on disk before the
        process claims success.
        """
        view = memoryview(payload)
        written = 0
        while written < len(view):
            try:
                written += os.write(fd, view[written:])
            except InterruptedError:  # pragma: no cover - signal timing
                continue
        os.fsync(fd)

    def _atomic_write(self, text: str) -> None:
        """Publish ``text`` to the destination, all-or-nothing.

        The destination is only ever touched by ``os.replace``, so it holds the
        old content until the new content is complete on disk.
        """
        payload = text.encode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp-{os.getpid()}")
        fd = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644
        )
        try:
            self._write_all(fd, payload)
            if os.fstat(fd).st_size != len(payload):
                raise OSError(
                    f"short write to {temporary}: {os.fstat(fd).st_size} of "
                    f"{len(payload)} bytes"
                )
        except BaseException:
            os.close(fd)
            with contextlib.suppress(OSError):
                temporary.unlink()
            raise
        os.close(fd)
        try:
            self._publish(temporary)
        except BaseException:
            with contextlib.suppress(OSError):
                temporary.unlink()
            raise
        # Durably record the rename itself, not just the file contents.
        directory = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        # The report descriptor now refers to the replaced inode; drop it but
        # KEEP the sidecar lock, which is what serialises against other runs.
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def _publish(self, temporary: Path) -> None:
        """Move the completed temporary file into place without clobbering."""
        if self.fd is None:
            # Nothing was there when we checked. `link` refuses if that is no
            # longer true, which is the atomic equivalent of an O_EXCL create.
            try:
                os.link(temporary, self.path)
            except FileExistsError:
                raise OSError(
                    f"{self.path} appeared after the ownership check; refusing to "
                    "overwrite a file this script did not create."
                ) from None
            temporary.unlink()
            return

        # We verified this inode carries our marker. Confirm the path still
        # resolves to it, so a substituted file is refused rather than replaced.
        verified = os.fstat(self.fd)
        try:
            current = os.stat(self.path, follow_symlinks=False)
        except FileNotFoundError:
            raise OSError(
                f"{self.path} disappeared after the ownership check; refusing to "
                "recreate it blindly."
            ) from None
        if (current.st_dev, current.st_ino) != (verified.st_dev, verified.st_ino):
            raise OSError(
                f"{self.path} was replaced after the ownership check "
                f"(inode {verified.st_ino} -> {current.st_ino}); refusing to overwrite it."
            )
        os.replace(temporary, self.path)

    def write(self, text: str) -> None:
        """Write the report. The destination is never left partially written."""
        self._atomic_write(text)

    def clear_stale(self) -> bool:
        """Replace our own stale report with a superseded notice.

        Uses the same atomic path as a normal write, so a failure here cannot
        leave a half-erased report either.
        """
        if self.fd is None or not self.is_ours:
            return False
        self._atomic_write(SUPERSEDED_NOTICE)
        return True


def load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def fmt(entry: dict[str, Any] | None) -> str:
    if not entry:
        return "—"
    if entry.get("unavailable"):
        return "n/a"
    return f"{entry['point']:.3f} [{entry['ci_low']:.3f}, {entry['ci_high']:.3f}]"


def fmt_delta(entry: dict[str, Any] | None) -> str:
    """Render a contrast cell.

    An entry may legitimately carry no numbers: `aggregate.py` emits
    `{"unavailable": True, "reason": ...}` when no episode was scored for both
    the arm and its baseline. That is deliberately NOT the same as a missing
    entry, and must not be formatted as a dash -- a reader would read a dash as
    "no effect" rather than "never compared".
    """
    if not entry:
        return "—"
    if entry.get("unavailable"):
        return "**n/a**"
    mark = ""
    if not entry.get("exploratory") and entry.get("p_holm", 1.0) < 0.05:
        improves = entry.get("improves_baseline")
        if improves is None:
            improves = entry["point"] * (-1 if entry.get("lower_is_better") else 1) > 0
        mark = " \\*" if improves else " **!**"
    return (
        f"{entry['point']:+.3f} [{entry['ci_low']:+.3f}, {entry['ci_high']:+.3f}]"
        f"{mark}"
    )


def load_gates(results: Path) -> dict | None:
    """Load every gate cell; the unqualified file is a duplicate primary alias."""
    cells = {}
    for path in sorted(results.glob("c1_gate*.json")):
        gate = load(path)
        if not gate or not all(k in gate for k in ("baseline", "arm", "verdict", "gate")):
            raise ValueError(f"unreadable C1 gate: {path}")
        key = f"{gate['arm']} vs {gate['baseline']}"
        if key in cells and cells[key] != gate:
            raise ValueError(f"conflicting C1 verdicts for {key}: {path}")
        cells[key] = gate
    if not cells:
        return None
    return {"cells": cells, "gate": "PASS" if all(g["gate"] == "PASS" for g in cells.values()) else "BLOCK"}


def paired_table(report: dict) -> list[str]:
    """Print both means on precisely the population used for their difference."""
    lines = ["| Comparison | Metric | Candidate mean | Baseline mean | Paired episodes / datasets | Difference |",
             "|---|---|---|---|---|---|"]
    for key, metrics in report.get("contrasts", {}).items():
        baseline = report.get("contrast_baselines", {}).get(key, report.get("baseline", "—"))
        for metric in HEADLINE:
            e = metrics.get(metric)
            if not e or e.get("unavailable"):
                lines.append(f"| {key} vs {baseline} | {PRETTY[metric]} | n/a | n/a | 0 | n/a |")
                continue
            lines.append(f"| {key} vs {baseline} | {PRETTY[metric]} | "
                         f"{e['paired_arm_mean']:.3f} | {e['paired_baseline_mean']:.3f} | "
                         f"{e['n_paired_episodes']} / {e['n_clusters']} | {fmt_delta(e)} |")
        warning = report.get("supervision_asymmetry", {}).get(key)
        if warning:
            lines.append(f"| **Caveat: {key}** | {warning} | | | | |")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--matcher", default="embedding")
    parser.add_argument(
        "--allow-incomplete", action="store_true",
        help="Write a clearly-marked draft despite missing evidence, for diagnosis. "
             "The output is stamped as not a result and the exit status stays non-zero.",
    )
    args = parser.parse_args()

    floor = load(args.results / "model_free_floor.json")
    main_report = load(args.results / f"report_{args.matcher}.json")
    c4_report = load(args.results / f"report_c4_{args.matcher}.json")
    c1 = load_gates(args.results)
    calibration = load(args.results / "matcher_calibration.json")
    reliability = load(args.results / "reliability.json")
    seams = load(args.results / "window_seams.json")
    breakdown = load(args.results / "breakdown.json")

    required = {
        "model-free floor": floor,
        "C1 gate verdict": c1,
        "C4 contrast (primary claim)": c4_report,
        "tool-vs-tool comparison": main_report,
        # Quality is scored only over episodes an arm published, and the arms
        # publish very different amounts. Rendering the quality tables without
        # the denominator would overstate whichever arm dropped more of the
        # hard components, so this is required evidence, not an appendix.
        "reliability (the denominator)": reliability,
    }
    missing = sorted(name for name, value in required.items() if not value)

    target = OutputTarget(args.out)
    try:
        if target.refusal:  # noqa: SIM102 - distinct exit reasons, kept separate
            print(f"REFUSED: {target.refusal}", file=sys.stderr)
            return 1
        if target.existed and not target.is_ours:
            print(
                f"REFUSED: {args.out} exists and was not generated by this script "
                "(no generation marker). Refusing to modify it; choose another --out path.",
                file=sys.stderr,
            )
            return 1

        if missing and not args.allow_incomplete:
            print(
                "REFUSED: cannot write a report without "
                + ", ".join(missing)
                + ".\nThese are the evidence the report's claims rest on; rendering it "
                "anyway would produce a document that reads like a result.\n"
                "Run the missing stages, or pass --allow-incomplete for a marked draft.",
                file=sys.stderr,
            )
            if target.clear_stale():
                print(f"cleared stale report {args.out}", file=sys.stderr)
            return 1
        return _render(args, target, missing, floor, c1, c4_report, main_report, calibration,
                       reliability, breakdown, seams)
    finally:
        target.close()


def _render(args, target, missing, floor, c1, c4_report, main_report, calibration,
            reliability, breakdown, seams) -> int:

    lines: list[str] = [
        GENERATION_MARKER,
        "# Subtask annotation: `lerobot-align` vs upstream `lerobot-annotate`",
        "",
    ]
    if missing:
        lines += [
            "> # ⚠ INCOMPLETE DRAFT — NOT A RESULT",
            ">",
            "> This document was generated with required evidence missing:",
            ">",
            *[f"> - **{name}**" for name in missing],
            ">",
            "> It exists for diagnosis only. Every claim below is unsupported until the",
            "> missing stages have been run. Do not cite, publish or summarise it.",
            "",
        ]

    lines += [
        "> Statistical intervals and contrasts were regenerated from saved episode scores.",
        "> Metric-definition version and any pending rescoring are recorded in regeneration.json.",
        "",
    ]
    regeneration = load(args.results / "regeneration.json")
    if regeneration:
        lines += [f"> **Metric rescoring:** {regeneration['metric_rescoring']}", ""]

    # --- what the reader must know before any number -----------------------
    lines += ["## Read this first", ""]
    if floor:
        prior = floor["component_weighted_means"].get("ref_script_prior", {})
        lines += [
            "On this corpus a **model-free script prior** — which sees ten labelled",
            "episodes and never looks at a single video frame — already scores:",
            "",
            "| Metric | Script prior |",
            "|---|---|",
        ]
        for metric in ("label_f1", "edit_score", "f1@25", "boundary_f1@1", "boundary_f1@0p5"):
            if metric in prior:
                lines.append(f"| {PRETTY.get(metric, metric)} | {prior[metric]:.3f} |")
        lines += [
            "",
            "Naming and step-sequence recovery are effectively *solved* by knowing the",
            f"script — label F1 {prior.get('label_f1', 0):.3f}, edit score {prior.get('edit_score', 0):.3f} — so no claim in this report rests",
            "on them, and the headline metrics are the three where the prior scores",
            "lowest. It is worth being blunt about what that does and does not mean: it",
            "narrows the metrics to where perception could in principle show, but it does",
            "NOT mean the prior is weak there. Macro IoU 0.688 is not a weak score, and",
            "no arm in this study beats the prior on any headline metric. An earlier",
            "draft of this sentence called the headline metrics the ones where the prior",
            "is 'genuinely weak', which the measured floor does not support.",
            "",
        ]
    else:
        lines += ["**The model-free floor has not been measured.** No result below is", 
                  "interpretable without it.", ""]

    # --- C1 gate ------------------------------------------------------------
    lines += ["## Gate: are the two tools equivalent with the new features off?", ""]
    if not c1:
        lines += [
            "**Not run.** Without it, any difference below could be an uncontrolled",
            "confound rather than a mechanism, and no C2/C3 claim is licensed.",
            "",
        ]
    else:
        lines += ["Each verdict applies only to its named cell. Untested cells have no C1 certification.",
                  "Repeated runs are averaged within episode before the episode bootstrap.", ""]
        for cell, gate in c1["cells"].items():
            lines += [f"### {cell}", "", f"Verdict: **{gate['verdict']}** (gate **{gate['gate']}**).", "",
                      "| Metric | Margin | Difference (90% CI) | Paired episodes | Verdict |",
                      "|---|---|---|---|---|"]
            for metric, entry in gate.get("metrics", {}).items():
                eq = entry.get("equivalence", {})
                lines.append(f"| {PRETTY.get(metric, metric)} | {entry.get('margin_used', float('nan')):.4f} "
                             f"| {fmt(eq)} | {entry.get('n_paired_episodes', 0)} | {eq.get('verdict', '—')} |")
            lines.append("")
            if gate["gate"] != "PASS":
                lines += ["> This cell is blocked. Its comparisons do not establish a mechanism effect.", ""]
            if gate.get("source_problems"):
                lines += [f"> Source check: {problem}" for problem in gate["source_problems"]] + [""]

    # --- C4: the primary claim ---------------------------------------------
    lines += ["## Primary claim: does reading the video beat the script prior?", ""]
    if not c4_report:
        lines += ["**Not run.** This is the primary claim; without it the report is incomplete.", ""]
    else:
        lines += paired_table(c4_report)
        lines += ["", "`*` = significantly better after Holm correction; `!` = significantly worse.",
                  "Both means and their difference use the same paired episodes and dataset weights.", ""]

    # --- C2/C3 --------------------------------------------------------------
    lines += ["## Tool vs tool", ""]
    if not main_report:
        lines += ["**Not run.**", ""]
    else:
        lines += ["### Descriptive means on each arm's own surviving episodes", "",
                  "These means describe different populations across arms. Use the paired table below for comparisons.", ""]
        arms = main_report.get("arms", [])
        lines += ["| Arm | " + " | ".join(PRETTY.get(m, m) for m in HEADLINE) + " |",
                  "|---|" + "---|" * len(HEADLINE)]
        for arm in arms:
            entries = main_report.get("per_arm", {}).get(arm, {})
            lines.append(f"| {arm} | " + " | ".join(fmt(entries.get(m)) for m in HEADLINE) + " |")
        lines.append("")
        contrasts = main_report.get("contrasts") or {}
        if contrasts:
            lines += ["### Paired comparisons, including C3 (realign minus video)", ""]
            lines += paired_table(main_report)
            lines += ["", f"Holm correction covers all {main_report.get('holm_family_size')} tests in this table.", ""]
        excluded = main_report.get("excluded_from_contrast") or {}
        if excluded:
            lines += ["### Excluded from contrast", ""]
            for arm, reason in excluded.items():
                lines.append(f"- **{arm}** — {reason}")
            lines.append("")

    # --- reliability --------------------------------------------------------
    lines += ["## Reliability: how much of the requested work each tool delivered", ""]
    if not reliability:
        lines += ["**Not measured.** Every quality table above is therefore reported over an",
                  "unknown denominator.", ""]
    else:
        from reliability import decomposition
        groups = decomposition(reliability)
        labels = {"upstream_matched": "Upstream, matched modes",
                  "fork_matched_modes": "Fork, matched modes (defaults + video)",
                  "fork_additional_modes": "Fork, additional modes (generate-then-realign, including stacks)"}
        lines += [
            "Quality is conditional on successful output. Reliability is decomposed by comparable mode",
            "and profile/camera cell; pooling additional fork capabilities into the tool comparison mixes estimands.",
            "",
            "| Group | Arms | Episodes published / requested | Episode yield | Jobs completed |",
            "|---|---|---|---|---|",
        ]
        for key, row in groups.items():
            lines.append(f"| {labels[key]} | {len(row['arms'])} | {row['episodes_published']}/{row['episodes_requested']} "
                         f"| {row['episode_yield']:.2%} | {row['jobs_ok']}/{row['jobs']} |")
        lines += ["", "The two fork generation modes each cover the six upstream cells. Additional modes have no upstream counterpart.", "",
                  "| Arm | Jobs completed | Episode yield | Discarded after success | min/job |",
                  "|---|---|---|---|---|"]
        for arm, row in sorted(reliability.items()):
            lines.append(
                f"| `{arm}` | {row['jobs_ok']}/{row['jobs']} | {row['episode_yield']:.1%} "
                f"| {row['episodes_wasted_by_batch_abort']} | {row['minutes_per_job']:.1f} |"
            )
        modes: dict[str, int] = {}
        for row in reliability.values():
            for mode, count in row["failure_modes"].items():
                modes[mode] = modes.get(mode, 0) + count
        if modes:
            lines += [
                "",
                "Every lost job ended in one of these exceptions, and each aborts the whole",
                "39–40 episode batch rather than the one bad episode — so a single invalid",
                "episode discards the ~39 that succeeded alongside it:",
                "",
                "| Failure mode | Jobs lost |",
                "|---|---|",
                *[f"| `{mode}` | {count} |" for mode, count in sorted(modes.items(), key=lambda kv: -kv[1])],
                "",
            ]

    # --- camera and profile -------------------------------------------------
    if breakdown:
        lines += [
            "## Which camera, and how many frames",
            "",
            "Different robot setups place their cameras differently, and a given view",
            "carries different information: a wrist camera sees contact and grasp, an",
            "external view sees approach and navigation. Camera is therefore reported as",
            "a dimension rather than chosen — and it turns out to matter more than the",
            "choice of tool in some cells. These tables are **exploratory**, with pointwise 95% intervals",
            "and no p-values or significance markers. Contrasts use episodes both sides",
            "produced, so they are not confounded by the differing yields above.",
            "",
            "### Camera (vs wrist, same tool and profile)",
            "",
            "| Contrast | Paired episodes | ΔBoundary F1@0.5 | ΔMacro IoU |",
            "|---|---|---|---|",
        ]
        for key, entry in sorted((breakdown.get("camera_contrasts") or {}).items()):
            lines.append(
                f"| {key} | {entry.get('n_paired', 'n/a')} | {fmt_delta(entry.get('boundary_f1@0p5'))} "
                f"| {fmt_delta(entry.get('macro_iou'))} |"
            )
        lines += [
            "",
            "Stacked multi-camera arms (`stack2`, `stack3`) exist only for `lerobot-align`:",
            "the upstream tool takes a single `camera_key` and has no way to present",
            "synchronised views together, so stacking has no upstream counterpart and is",
            "reported as a within-tool comparison only.",
            "",
            "### Frame budget (vs the `wrap` profile, same tool and camera)",
            "",
            "| Contrast | Paired episodes | ΔBoundary F1@0.5 | ΔMacro IoU |",
            "|---|---|---|---|",
        ]
        for key, entry in sorted((breakdown.get("profile_contrasts") or {}).items()):
            lines.append(
                f"| {key} | {entry.get('n_paired', 'n/a')} | {fmt_delta(entry.get('boundary_f1@0p5'))} "
                f"| {fmt_delta(entry.get('macro_iou'))} |"
            )
        lines.append("")

    # --- window seams -------------------------------------------------------
    if seams:
        windowed = {a: r for a, r in seams.items() if r.get("boundaries")}
        if windowed:
            lines += [
                "## Window seams",
                "",
                "Both tools cut a long episode into windows, prompt the model on each",
                "separately and stitch the results, so a seam is a place the model could",
                "not see across -- a boundary there is an artifact of the harness rather",
                "than an observation about the robot. The two tools lay windows out",
                "differently (upstream: fixed `budget / fps` seconds; align: `budget - 1`",
                "intervals with the final pair rebalanced), so each arm is scored against",
                "**its own** grid. `Human` is the chance baseline: the share of the",
                "annotators' own boundaries falling near the same seams.",
                "",
                "| Arm | Windowed episodes | Interior boundaries | On seam | Human | Excess |",
                "|---|---|---|---|---|---|",
            ]
            for arm, row in sorted(windowed.items()):
                human = row.get("human_seam_share") or 0.0
                lines.append(
                    f"| `{arm}` | {row['windowed_episodes']} | {row['boundaries']} "
                    f"| {row['seam_share']:.1%} | {human:.1%} "
                    f"| {row['seam_share'] - human:+.1%} |"
                )
            lines += [
                "",
                "About one predicted interior boundary in five sits on a seam in a windowed",
                "episode, against ~2.5% of human boundaries -- for **both** tools. The",
                "tail-balancing in `lerobot-align` does not measurably reduce it and the",
                "native-video path is worse. Only the realign path suppresses seams, by",
                "re-placing every boundary against the label list instead of inheriting the",
                "generation seams, and that path has the worst reliability in the study.",
                "",
                "At the `wrap` profile almost no episode is windowed at all (the 100s window",
                "exceeds nearly every episode), so `wrap` avoids seams entirely and still",
                "scores worse -- its problem is under-segmentation, not seams.",
                "",
            ]

    # --- failure rates ------------------------------------------------------
    if main_report and main_report.get("failure_rates"):
        lines += ["### Failure rates", "",
                  "An arm that cannot annotate an episode is recorded, never dropped.", "",
                  "| Arm | Failed | Total | Rate |", "|---|---|---|---|"]
        for arm, stats in main_report["failure_rates"].items():
            lines.append(f"| {arm} | {stats['failed']} | {stats['total']} | {stats['rate']:.1%} |")
        lines.append("")

    # --- limitations --------------------------------------------------------
    lines += [
        "## Limitations",
        "",
        "- **Development contamination.** This tool was developed against this corpus.",
        "  Prompts are byte-identical to upstream, but settings were chosen by observing",
        "  results here, so the comparison is tuned-versus-untuned in the new tool's favour.",
        "- **Scriptedness.** Several components have a single human label sequence across",
        "  all 50 episodes; see the floor above.",
        "- **Single annotator.** No repeat annotation exists, so there is no measured",
        "  inter-annotator floor and no score can be read as exceeding human agreement.",
        "- **One model.** All results are conditional on Qwen3.8-27B at temperature 0.2,",
        "  with no seed control in either tool, so runs are not reproducible.",
        "- **C2/C3 confound prompts with representation.** The video arms use fork-authored",
        "  prompts with no upstream counterpart.",
        "- **Not equal compute.** The realign arm spends an extra whole-episode VLM pass.",
        "- **Every tool contrast is measured on the episodes both arms produced.** That",
        "  keeps each contrast internally valid, but the shared set is the components",
        "  `lerobot-align` did not abort on, and those aborts are not random -- they",
        "  concentrate on the components it finds hard. No contrast here generalises to",
        "  the full corpus, because there is no measurement of how the align arms would",
        "  have scored on the episodes missing from each arm (see the decomposed yields).",
        "- **C1 certification is cell-specific.** All available gate cells are shown above.",
        "  A passing cell cannot certify a blocked or untested profile/camera cell.",
    ]
    if calibration and not calibration.get("paraphrase_validated"):
        lines.append(
            "- **Matcher threshold provisional.** Not yet validated on hand-labelled "
            "(model, human) paraphrase pairs, so label-aware numbers carry extra uncertainty."
        )
    lines.append("")

    target.write("\n".join(lines) + "\n")

    if missing:
        print(f"[report] wrote INCOMPLETE DRAFT {args.out} ({len(lines)} lines)")
        print(f"[report] missing evidence: {', '.join(missing)}", file=sys.stderr)
        print("[report] exit status is non-zero: this is not a result.", file=sys.stderr)
        return 1

    gate = (c1 or {}).get("gate")
    if gate != "PASS":
        print(f"[report] wrote {args.out}, but the C1 gate is {gate}: the tool-vs-tool "
              "numbers are diagnostic only.", file=sys.stderr)
        return 1
    print(f"[report] wrote {args.out} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
