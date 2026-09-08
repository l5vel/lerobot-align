#!/usr/bin/env python
"""Render the fixed-label alignment report from the aggregated results.

Assembles one markdown document from artifacts that already exist and refuses to
invent anything that does not. This is the report generator `alignment_plan.md`
section 10.2 lists as missing; `make_report.py` renders the *generation* study
and shares none of this study's structure, because none of this study's
contrasts are tool-versus-tool.

**The supervision stamp is rendered or nothing is written.**
`aggregate.py` permits a cross-level contrast only on condition that its
`supervision_asymmetry` travels with every number it produces (section 4.1). The
previous study computed exactly that stamp and never printed it, which is how a
comparison between arms at different supervision levels reached a headline and
had to be withdrawn (C4). A promise to render a caveat is worth nothing if the
renderer can quietly drop it, so this script CHECKS its own output: every
asymmetry the aggregate recorded must appear verbatim in the finished document
and be marked on the contrast's own row. If any is missing the report is not
written and the exit status is non-zero. That check is the point of this file as
much as the tables are.

**Missing evidence is a failure, not a footnote.** Following `make_report.py`:
the required artifacts are checked first, and if any is absent the report is NOT
written. `--allow-incomplete` opts into a draft for diagnosis, stamped with a
banner that survives copy-paste and still exits non-zero.

**Placed-only error is never shown alone.** Placed-only MAE improves when an arm
drops the labels it finds hard, so a table that shows it without
`placed_fraction` beside it ranks arms partly by how much work they refused to
do (section 6.1). The cell renderer refuses to emit one without the other rather
than trusting the table author to remember.

Ordering is deliberate. The metric caveats and the unmeasured inter-annotator
agreement come BEFORE any number, because a reader who meets "1.51s mean
boundary error" first will read it as accuracy, and nothing in this corpus
establishes that two humans agree to 1.51s.

Output safety mirrors `make_report.py`: one `O_NOFOLLOW` descriptor, an
ownership marker that licenses overwriting only this script's own output, a
sidecar `flock` so two runs serialise, and an atomic temp-file publish so a
failure mid-write cannot leave a document that reads as authoritative and stops
halfway. The marker is deliberately DIFFERENT from `make_report.py`'s: sharing
one would let either script overwrite the other's report.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import fcntl
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any

GENERATION_MARKER = "<!-- generated-by: evaluation/scripts/make_alignment_report.py -->"

SUPERSEDED_NOTICE = (
    f"{GENERATION_MARKER}\n"
    "# Superseded\n\n"
    "A previous alignment report was generated here, then invalidated because the\n"
    "evidence it rested on is missing or has changed. It has been cleared\n"
    "deliberately: a stale report sitting beside fresh results is easily mistaken\n"
    "for current.\n"
)

PRETTY = {
    "b_hit@1": "B@1 (boundary within 1s)",
    "b_hit@3": "B@3 (boundary within 3s)",
    "b_hit@5": "B@5 (boundary within 5s)",
    "placed_fraction": "Placed fraction",
    "macro_temporal_iou": "Macro temporal IoU",
    "boundary_mae_placed": "Placed-only boundary MAE (s)",
    "boundary_mae_norm": "Normalised boundary MAE",
    "calibration_applied": "Calibration-applied rate",
}

# Rendered together, always, in this order. Placed-only MAE is paired with the
# coverage metric it is conditional on; see the module docstring.
PRIMARY = ("b_hit@3", "placed_fraction")
CONTINUITY = ("macro_temporal_iou",)
SECONDARY = ("b_hit@1", "b_hit@5", "boundary_mae_placed", "boundary_mae_norm",
             "calibration_applied")

CLAIM_PATTERN = re.compile(r"\bA(\d+)\b")
PRIMARY_CLAIMS = ("A1", "A2", "A3")

BETTER = "\\*"          # significantly better after Holm
WORSE = "**!**"         # significantly worse after Holm
CROSS_LEVEL = "†"       # supervision levels differ; see the cross-level section
CONFOUNDED = "‡"        # more than one dimension differs, or none was declared


class OutputTarget:
    """Exclusive handle on ``--out``, safe against swaps and symlinks.

    A trimmed sibling of ``make_report.OutputTarget``; that class is not imported
    because it publishes under *its* generation marker, and two report generators
    sharing one marker may overwrite each other's documents. The rationale for
    each step is written out there at length and is not repeated here.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.fd: int | None = None
        self.lock_fd: int | None = None
        self.existed = False
        self.is_ours = False
        self.refusal: str | None = None

        lock_path = path.with_name(f".{path.name}.lock")
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            self.lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o644)
        except OSError as exc:
            self.refusal = (
                f"{lock_path} is a symbolic link; refusing to lock through it"
                if exc.errno in (errno.ELOOP, errno.EMLINK)
                else f"cannot create lock {lock_path}: {exc}"
            )
            return
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
            self.refusal = (
                f"{path} is a symbolic link. Refusing to write through it, because the "
                "target could be any file. Pass a regular path."
                if exc.errno in (errno.ELOOP, errno.EMLINK)
                else f"cannot open {path}: {exc}"
            )
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
        """Write every byte, or raise. A short write leaves a truncated report."""
        view = memoryview(payload)
        written = 0
        while written < len(view):
            try:
                written += os.write(fd, view[written:])
            except InterruptedError:  # pragma: no cover - signal timing
                continue
        os.fsync(fd)

    def _publish(self, temporary: Path) -> None:
        if self.fd is None:
            try:
                os.link(temporary, self.path)
            except FileExistsError:
                raise OSError(
                    f"{self.path} appeared after the ownership check; refusing to "
                    "overwrite a file this script did not create."
                ) from None
            temporary.unlink()
            return
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
        """Publish ``text`` all-or-nothing: readers see old or new, never half."""
        payload = text.encode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp-{os.getpid()}")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
        try:
            self._write_all(fd, payload)
            if os.fstat(fd).st_size != len(payload):
                raise OSError(f"short write to {temporary}")
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
        directory = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def clear_stale(self) -> bool:
        if self.fd is None or not self.is_ours:
            return False
        self.write(SUPERSEDED_NOTICE)
        return True


def load(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def fmt(entry: dict[str, Any] | None, digits: int = 3) -> str:
    if not entry:
        return "—"
    if entry.get("unavailable"):
        return "n/a"
    return (f"{entry['point']:.{digits}f} "
            f"[{entry['ci_low']:.{digits}f}, {entry['ci_high']:.{digits}f}]")


def fmt_delta(entry: dict[str, Any] | None, digits: int = 3, alpha: float = 0.05) -> str:
    """Render a contrast cell, with the marker its polarity and family imply.

    ``improves_baseline`` is read from the aggregate rather than recomputed from
    the sign of the point estimate: for a lower-is-better metric the improvement
    is the NEGATIVE delta, and re-deriving that here would be a second place for
    the polarity to be got wrong (D14). ``alpha`` likewise comes from the
    pre-registration by way of the aggregate, not from a constant here.
    """
    if not entry:
        return "—"
    if entry.get("unavailable"):
        return "**n/a**"
    mark = ""
    if not entry.get("exploratory") and entry.get("p_holm", 1.0) < alpha:
        mark = f" {BETTER}" if entry.get("improves_baseline") else f" {WORSE}"
    return (f"{entry['point']:+.{digits}f} "
            f"[{entry['ci_low']:+.{digits}f}, {entry['ci_high']:+.{digits}f}]{mark}")


def mae_cell(report: dict[str, Any], arm: str) -> str:
    """Placed-only MAE, never without the coverage it is conditional on.

    Section 6.1: dropping the hard labels *improves* this number, so an arm can
    buy a better MAE by placing less. Withholding it is the honest failure mode
    when the denominator is missing.
    """
    entries = report.get("per_arm", {}).get(arm, {})
    mae, placed = entries.get("boundary_mae_placed"), entries.get("placed_fraction")
    if not mae:
        return "—"
    if not placed:
        return "**withheld** (no `placed_fraction`)"
    return f"{mae['point']:.2f}s at placed {placed['point']:.3f}"


def claim_ids(text: str) -> set[str]:
    return {f"A{number}" for number in CLAIM_PATTERN.findall(text or "")}


def contrast_label(report: dict[str, Any], key: str) -> str:
    """Claim id plus the pre-registered one-line label, when the config gave one."""
    claim = report.get("contrast_claims", {}).get(key, "")
    label = (report.get("contrast_options", {}).get(key) or {}).get("label")
    return f"{claim} — {label}" if claim and label else (claim or label or "")


def contrast_rows(report: dict[str, Any], metrics: tuple[str, ...]) -> list[str]:
    """The pre-registered family, one row per (contrast, metric).

    Both means come from exactly the paired population their difference was
    computed on, so no row mixes an arm's own survivors with a paired delta.
    """
    lines = [
        "| # | Contrast | Claim | Metric | Arm mean | Baseline mean | Difference (95% CI) | "
        "Holm p | Components won | Flags |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    alpha = float(report.get("alpha", 0.05))
    for key, entries in report.get("contrasts", {}).items():
        identifier = report.get("contrast_ids", {}).get(key, "")
        claim = contrast_label(report, key)
        baseline = report.get("contrast_baselines", {}).get(key, "—")
        arm = report.get("contrast_arms", {}).get(key, key)
        scope = report.get("contrast_scope", {}).get(key) or {}
        flags = ""
        if report.get("supervision_asymmetry", {}).get(key):
            flags += CROSS_LEVEL
        if scope:
            # No scope recorded at all is not the same as "nothing differs", so the
            # confound marker is only claimed when the aggregate actually measured it.
            declared = scope.get("declared")
            if (declared is None or declared == "all"
                    or len(scope.get("differs_on") or []) > 1):
                flags += CONFOUNDED
        for metric in metrics:
            entry = entries.get(metric)
            if not entry or entry.get("unavailable"):
                lines.append(
                    f"| {identifier} | `{arm}` vs `{baseline}` | {claim} | "
                    f"{PRETTY.get(metric, metric)} | n/a | n/a | "
                    f"{fmt_delta(entry, alpha=alpha)} | — | — | {flags} |"
                )
                continue
            holm = ("exploratory" if entry.get("exploratory")
                    else f"{entry.get('p_holm', float('nan')):.4f}")
            direction = " (lower better)" if entry.get("lower_is_better") else ""
            lines.append(
                f"| {identifier} | `{arm}` vs `{baseline}` | {claim} | "
                f"{PRETTY.get(metric, metric)}{direction} | "
                f"{entry['paired_arm_mean']:.3f} | {entry['paired_baseline_mean']:.3f} | "
                f"{fmt_delta(entry, alpha=alpha)} | {holm} | "
                f"{entry['dataset_wins']}/{entry['dataset_wins'] + entry['dataset_losses']} | "
                f"{flags} |"
            )
    return lines


def cross_level_section(report: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    """Render every supervision stamp verbatim, and return what was rendered.

    The returned mapping is checked against the aggregate before the document is
    written. Rendering and verifying from the same dictionary is the whole
    mechanism: a stamp that is not rendered here cannot pass the check.
    """
    stamps: dict[str, str] = dict(report.get("supervision_asymmetry", {}))
    lines = ["## Contrasts that cross a supervision level", ""]
    if not stamps:
        lines += [
            "No contrast in this family crosses a supervision level: every pair is at the",
            "same level, so no result below is confounded with the ten annotated seed",
            "episodes a calibration fit consumes.",
            "",
        ]
        return lines, stamps
    lines += [
        f"{len(stamps)} of the pre-registered contrasts compare arms at DIFFERENT",
        "supervision levels. Each was declared `cross_level: true` in the contrast config,",
        "which is why it was computed at all — the gate refuses an undeclared one — and",
        f"each carries the marker {CROSS_LEVEL} on every row of its table above. The",
        "asymmetry is stated in full here because a marker is not an explanation:",
        "",
    ]
    for key, text in stamps.items():
        lines += [f"### {key}", "", f"> {text}", ""]
    lines += [
        "Read these as claims about *a system that has been given ten labelled examples of",
        "each component*, and price those ten examples into the effect. They are not",
        "like-for-like configuration changes, and `alignment_plan.md` section 8 registered",
        "them as such before the run rather than discovering it afterwards.",
        "",
    ]
    return lines, stamps


def equalisation_section(report: dict[str, Any]) -> list[str]:
    """What actually differs between each pair, against what was declared."""
    scopes = report.get("contrast_scope", {})
    if not scopes:
        return []
    lines = [
        "## Equalisation: what differs in each contrast",
        "",
        "Every arm shares seed, temperature, token budget and replica pool (section 7.1),",
        "so a contrast is interpretable only if the pair differs on the dimension under",
        "test and nothing else. That is a property of the PAIR, which is why the gate is",
        "scoped per contrast rather than by a per-arm `equalised` flag: the flag would",
        "exclude the stacked-camera arm from the very contrast that is about cameras.",
        "Differences are read from the arms' declared flags, not from a label.",
        "",
        "| Contrast | Differs on | Declared under test | Marked `equalised: false` |",
        "|---|---|---|---|",
    ]
    for key, scope in scopes.items():
        declared = scope.get("declared")
        declared_text = ("**all — deliberately confounded**" if declared == "all"
                         else "**none declared**" if declared is None
                         else ", ".join(f"`{d}`" for d in declared))
        differs = ", ".join(f"`{d}`" for d in scope.get("differs_on") or []) or "nothing"
        unequalised = ", ".join(f"`{a}`" for a in scope.get("unequalised_arms") or []) or "—"
        lines.append(f"| {key} | {differs} | {declared_text} | {unequalised} |")
    lines += [
        "",
        f"{CONFOUNDED} on a row above marks a contrast whose arms differ on more than one",
        "dimension, or that declared none. Those measure a margin, not a mechanism.",
        "",
    ]
    warnings = report.get("config_warnings") or {}
    if warnings:
        lines += ["Configuration warnings raised while building the family:", ""]
        for key, items in warnings.items():
            for warning in items:
                lines.append(f"- **{key}** — {warning}")
        lines.append("")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--aggregate", type=Path, default=None,
        help="Aggregated contrasts over the whole eval population "
             "(default <results>/alignment_all.json).",
    )
    parser.add_argument(
        "--modal-aggregate", type=Path, default=None,
        help="The same family restricted to episodes whose label list matches the one the "
             "calibration was fit for (default <results>/alignment_modal.json). Optional, "
             "and reported beside the unconditional numbers rather than instead of them: "
             "on a mixed-convention component the calibrated arm is calibrated on part of "
             "the population and uncalibrated on the rest (section 4.2).",
    )
    parser.add_argument(
        "--stratum", action="append", default=[], metavar="NAME=PATH",
        help="An additional aggregate over a subset of components, e.g. "
             "non_fridge_11=results/alignment_non_fridge.json. Reported BESIDE the "
             "headline population, never instead of it: the method was developed on a "
             "sibling of four components in this corpus, so the development-adjacent "
             "family is shown separately rather than dropped (alignment_plan.md §3.2).",
    )
    parser.add_argument(
        "--gates", type=Path, default=None,
        help="Preflight gate record (default <results>/alignment_gates.json).",
    )
    parser.add_argument(
        "--calibration", type=Path, default=None,
        help="Per-component calibration fit record (default "
             "<results>/alignment_calibration.json).",
    )
    parser.add_argument(
        "--allow-incomplete", action="store_true",
        help="Write a clearly-marked draft despite missing evidence, for diagnosis. "
             "The output is stamped as not a result and the exit status stays non-zero.",
    )
    args = parser.parse_args()

    aggregate = load(args.aggregate or args.results / "alignment_all.json")
    modal = load(args.modal_aggregate or args.results / "alignment_modal.json")
    gates = load(args.gates or args.results / "alignment_gates.json")
    calibration = load(args.calibration or args.results / "alignment_calibration.json")
    strata: dict[str, dict[str, Any]] = {}
    for item in args.stratum:
        name, _, location = item.partition("=")
        loaded = load(Path(location))
        if loaded is None:
            print(f"REFUSED: stratum '{name}' names {location}, which is not readable. A "
                  "stratum that silently vanishes turns a separately-reported population "
                  "into an excluded one.", file=sys.stderr)
            return 1
        strata[name] = loaded

    if aggregate is not None and aggregate.get("mode") != "alignment":
        print(
            "REFUSED: the aggregate was not produced in contrast-list mode "
            "(`mode: alignment`). Its contrasts are per-arm-baseline comparisons, not the "
            "pre-registered family, and Holm corrected a different set of tests than this "
            "report would claim. Re-run aggregate.py with --contrasts.",
            file=sys.stderr,
        )
        return 1

    computed_claims: set[str] = set()
    for key in (aggregate or {}).get("contrasts", {}):
        computed_claims |= claim_ids((aggregate or {}).get("contrast_claims", {}).get(key, ""))
    required = {
        "alignment contrast aggregate": aggregate,
        # A1-A3 are the primary claims (section 1.3). A report that renders the
        # secondary rows while the primary contrast was refused or unpaired reads as
        # a result about configuration and is not one.
        "the primary configuration claims A1, A2 and A3": (
            aggregate is not None and all(c in computed_claims for c in PRIMARY_CLAIMS)
        ),
        # A family smaller than the one registered was corrected more weakly than
        # promised, so its surviving p-values are not the ones the pre-registration
        # licensed. That is missing evidence, not a footnote.
        "the complete pre-registered Holm family": (
            aggregate is not None
            and (aggregate.get("expected_tests") is None
                 or aggregate.get("holm_family_size") == aggregate["expected_tests"])
        ),
    }
    missing = sorted(name for name, value in required.items() if not value)

    target = OutputTarget(args.out)
    try:
        if target.refusal:
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
                "REFUSED: cannot write an alignment report without "
                + ", ".join(missing)
                + ".\nThese are the evidence the claims rest on; rendering the document "
                "anyway would produce something that reads like a result.\n"
                "Run the missing stages, or pass --allow-incomplete for a marked draft.",
                file=sys.stderr,
            )
            if target.clear_stale():
                print(f"cleared stale report {args.out}", file=sys.stderr)
            return 1
        return _render(args, target, missing, aggregate, modal, strata, gates, calibration)
    finally:
        target.close()


def _render(args, target, missing, aggregate, modal, strata, gates, calibration) -> int:
    report = aggregate or {}
    # Which metrics are in the family, and in which order, is the aggregate's answer
    # rather than a third hard-coded list here: the config, the aggregator and this
    # renderer disagreeing about the family is how a test gets reported as corrected
    # when it was not. The preferred ORDER is this study's, though -- primaries first,
    # the continuity metric last -- because sorting them alphabetically buries
    # `placed_fraction` behind a metric no claim rests on.
    family = tuple(report.get("holm_family_metrics") or (PRIMARY + CONTINUITY))
    family_ordered = tuple([m for m in PRIMARY + CONTINUITY if m in family]
                           + [m for m in family if m not in PRIMARY + CONTINUITY])
    secondary = tuple(m for m in (report.get("metrics") or SECONDARY) if m not in family)
    lines: list[str] = [
        GENERATION_MARKER,
        "# Fixed-label subtask alignment: which configuration of `lerobot-align` places "
        "boundaries best",
        "",
    ]
    if missing:
        lines += [
            "> # INCOMPLETE DRAFT — NOT A RESULT",
            ">",
            "> This document was generated with required evidence missing:",
            ">",
            *[f"> - **{name}**" for name in missing],
            ">",
            "> It exists for diagnosis only. Every claim below is unsupported until the",
            "> missing stages have been run. Do not cite, publish or summarise it.",
            "",
        ]

    # --- what the reader must know before any number -----------------------
    lines += [
        "## Read this first",
        "",
        "**This is a within-tool comparison.** The caller supplies an ordered label list",
        "and the model decides only *when* each label happens. Upstream `lerobot-annotate`",
        "has no fixed-label mode at all, so there is no tool-versus-tool contrast to make",
        "here; that comparison is the other study, on generation from scratch. Nothing",
        "below says `lerobot-align` beats anything except other configurations of itself",
        "and two model-free timing templates.",
        "",
        "**Two metric caveats change how every table reads.**",
        "",
        "1. Boundary MAE is a drop counter wearing an accuracy metric's clothes. A missing",
        "   label is scored at the worst error achievable inside the episode — a median of",
        "   0.75 of the episode once normalised — so one dropped label outweighs roughly 49",
        "   accurately placed ones. Any MAE ranking is mostly a ranking on drop rate. The",
        "   primaries are therefore B@3, which is bounded and scores a miss as a miss, and",
        "   `placed_fraction`, which is the drop rate itself.",
        "2. Macro temporal IoU is endpoint-inflated: it scores the stitched segmentation",
        "   including the two forced episode endpoints, so every arm — the uniform floor",
        "   included — banks credit for boundaries no one placed. It is reported for",
        "   continuity with the prior work, not as the headline.",
        "",
        "**Inter-annotator agreement is unmeasured, and it bounds everything here.** Every",
        "number scores the model against one human labelling treated as truth. If",
        "annotators typically disagree by ~1.5s the method is already saturated and the",
        "remaining levers fit one person's idiosyncrasies; if they agree to ~0.3s there is",
        "real headroom. No second annotation exists in this corpus — the two fridge files",
        "that looked independent are byte-identical. Claim A6 stands unresolved and every",
        "improvement below roughly 1.5s of boundary error is uninterpretable.",
        "",
    ]

    # --- the family, and the correction applied to it ----------------------
    contrasts = report.get("contrasts") or {}
    lines += [
        "## The pre-registered family",
        "",
        f"- Contrast config: `{report.get('contrast_config', '—')}`"
        f" (family `{report.get('contrast_family', '—')}`)",
        f"- Contrasts computed: **{len(contrasts)}**",
        f"- Holm family: **{report.get('holm_family_size', 0)} tests** over "
        + ", ".join(f"`{m}`" for m in report.get("holm_family_metrics") or [])
        + ", corrected in ONE pass"
        + (f" (pre-registered size: {report['expected_tests']})"
           if report.get("expected_tests") else ""),
        f"- Significance threshold: {report.get('alpha', 0.05)} on the Holm-adjusted p",
        f"- Components (clusters): **{len(report.get('datasets') or [])}**; "
        f"arms scored: **{len(report.get('arms') or [])}**",
        "",
        "One pass matters. The family was pre-registered as every (contrast x primary",
        "metric) pair; computing it as several per-baseline families would correct each",
        "against a divisor several times too small and make significance strictly easier to",
        "reach. The unit of analysis is the component, not the episode.",
        "",
    ]
    if report.get("holm_family_note"):
        lines += [f"> **{report['holm_family_note']}**", ""]
    population = report.get("population") or {}
    if population.get("filter"):
        lines += [
            f"> **Restricted population.** These rows cover only score rows where "
            f"{population['filter']} is true: "
            f"{population.get('rows_after_filter')} of "
            f"{population.get('rows_before_filter')} rows.",
            "",
        ]

    # --- the stamp. Rendered before the tables it annotates. ---------------
    stamp_lines, stamps = cross_level_section(report)
    lines += stamp_lines

    # --- descriptive per-arm ------------------------------------------------
    arms = report.get("arms") or []
    lines += [
        "## Per-arm descriptive means",
        "",
        "Each arm over its own surviving episodes, with cluster-bootstrap intervals over",
        "components. These populations differ between arms; use the paired contrasts below",
        "for any comparison.",
        "",
        "| Arm | " + " | ".join(PRETTY.get(m, m) for m in family_ordered)
        + " | Placed-only MAE (with coverage) | Calibration applied |",
        "|---|" + "---|" * (len(family_ordered) + 2),
    ]
    for arm in arms:
        entries = report.get("per_arm", {}).get(arm, {})
        cells = " | ".join(fmt(entries.get(metric)) for metric in family_ordered)
        applied = entries.get("calibration_applied")
        applied_cell = f"{applied['point']:.3f}" if applied else "—"
        lines.append(f"| `{arm}` | {cells} | {mae_cell(report, arm)} | {applied_cell} |")
    lines += [
        "",
        "Placed-only MAE is shown only with the coverage it is conditional on: an arm that",
        "drops its hard labels improves that column while getting worse. The",
        "calibration-applied rate is a primary reported quantity, not a footnote — under",
        "per-episode label lists the tool's calibration only applies where the label tuple",
        "matches the one it was fit for, so a 'calibrated' arm is calibrated on part of the",
        "population and uncalibrated on the rest.",
        "",
    ]

    # --- the contrasts ------------------------------------------------------
    lines += [
        "## Pre-registered contrasts",
        "",
        "Primary metrics and the continuity metric; these are the tests Holm corrected.",
        "",
    ]
    primary_rows = contrast_rows(report, family_ordered)
    lines += primary_rows
    secondary_rows: list[str] = []
    lines += [
        "",
        f"Legend: `{BETTER}` significantly better after Holm correction; `{WORSE}` "
        f"significantly worse; `{CROSS_LEVEL}` the two arms are at different supervision "
        f"levels (see above); `{CONFOUNDED}` more than one configuration dimension differs.",
        "Both means are computed on exactly the paired episodes their difference uses.",
        "",
        "### Secondary metrics (computed, reported, outside the Holm family)",
        "",
        "No claim rests on these and none can be starred: adding them would enlarge the",
        f"correction from {report.get('holm_family_size', 0)} tests to "
        f"{len(contrasts) * (len(family) + len(secondary))} and weaken every pre-registered",
        "result to buy significance for numbers nothing depends on. Lower is better for both",
        "MAE rows, which is declared in the aggregator and cross-checked against the contrast",
        "config before anything is computed, rather than inferred from the metric name.",
        "",
    ]
    secondary_rows = contrast_rows(report, secondary)
    lines += secondary_rows
    lines.append("")

    # --- how much of the treatment actually landed --------------------------
    treated = [key for key, options in (report.get("contrast_options") or {}).items()
               if options.get("report_calibration_applied") and key in contrasts]
    if treated:
        lines += [
            "## How much of the calibration treatment actually landed",
            "",
            "A calibration is fit for one label tuple and `applies_to` demands an exact match,",
            "so on a component whose episodes follow several conventions the \"calibrated\" arm",
            "is calibrated on part of the population and uncalibrated on the rest. The effect",
            "sizes above are therefore BLENDED, and the rate below says how blended. It is a",
            "reliability rate, not an outcome — nothing is claimed from it.",
            "",
            "| Contrast | Calibrated arm | Baseline | Applied to |",
            "|---|---|---|---|",
        ]
        for key in treated:
            entry = (contrasts.get(key) or {}).get("calibration_applied")
            if not entry or entry.get("unavailable"):
                lines.append(f"| {key} | n/a | n/a | **not measured** |")
                continue
            lines.append(
                f"| {key} | {entry['paired_arm_mean']:.3f} | "
                f"{entry['paired_baseline_mean']:.3f} | "
                f"{entry['paired_arm_mean']:.1%} of paired episodes |"
            )
        lines.append("")

    # --- equalisation -------------------------------------------------------
    lines += equalisation_section(report)

    # --- modal subpopulation ------------------------------------------------
    lines += ["## The same contrasts on the uniform-treatment subpopulation", ""]
    if not modal:
        lines += [
            "**Not computed.** Under per-episode label lists a calibration fit for one label",
            "tuple no-ops on every episode whose tuple differs, so the calibrated arms carry",
            "a blended treatment and the calibration contrasts above are diluted by an",
            "unknown amount. Re-run `aggregate.py` with `--require-true` naming the",
            "modal-label-list field to bound it.",
            "",
        ]
    else:
        lines += [
            "Restricted to episodes whose label list matches the one the calibration was fit",
            f"for ({(modal.get('population') or {}).get('rows_after_filter')} of "
            f"{(modal.get('population') or {}).get('rows_before_filter')} score rows), where",
            "the treatment is uniform. Reported BESIDE the unconditional numbers, never",
            "instead of them: the prior sweep quietly reported only this population, which is",
            "precisely where calibration already works.",
            "",
            "| Contrast | Metric | All episodes | Uniform-treatment subpopulation |",
            "|---|---|---|---|",
        ]
        # Only the contrasts whose treatment is blended registered for this second
        # reading; showing it for the others would imply a caveat they do not carry.
        registered = [key for key, options in (report.get("contrast_options") or {}).items()
                      if options.get("restrict_to_modal_subpopulation")]
        for key, entries in contrasts.items():
            if registered and key not in registered:
                continue
            for metric in PRIMARY:
                lines.append(
                    f"| {key} | {PRETTY.get(metric, metric)} | "
                    f"{fmt_delta(entries.get(metric))} | "
                    f"{fmt_delta((modal.get('contrasts', {}).get(key) or {}).get(metric))} |"
                )
        lines.append("")

    # --- strata -------------------------------------------------------------
    if strata:
        lines += [
            "## The same contrasts on each pre-registered stratum",
            "",
            "The headline population is all 16 components. The alignment prompts, frame",
            "format and calibration design were developed on a sibling of the fridge-drink",
            "components, so those are reported SEPARATELY and labelled development-adjacent",
            "rather than dropped: excluding them would be a choice made after seeing the",
            "data, and reporting them silently would let development adjacency inflate the",
            "headline. Note the power cost — over 11 components the Holm threshold tolerates",
            "zero losing components on the sign test — which is why the cluster bootstrap is",
            "the primary inference and these are strata, not the headline.",
            "",
            "| Contrast | Metric | Headline (all components) | "
            + " | ".join(name for name in strata) + " |",
            "|---|---|---|" + "---|" * len(strata),
        ]
        for key, entries in contrasts.items():
            for metric in PRIMARY:
                cells = " | ".join(
                    fmt_delta((stratum.get("contrasts", {}).get(key) or {}).get(metric),
                              alpha=float(stratum.get("alpha", 0.05)))
                    for stratum in strata.values()
                )
                lines.append(
                    f"| {key} | {PRETTY.get(metric, metric)} | "
                    f"{fmt_delta(entries.get(metric), alpha=float(report.get('alpha', 0.05)))} "
                    f"| {cells} |"
                )
        lines += [
            "",
            "Each stratum was corrected within its own Holm family, so a marker there is not "
            "comparable to a marker in the headline column.",
            "",
        ]
        for name, stratum in strata.items():
            filters = stratum.get("population") or {}
            lines.append(
                f"- **{name}**: {len(stratum.get('datasets') or [])} components"
                f"{', matching ' + str(filters['datasets_matching']) if filters.get('datasets_matching') else ''}"
                f"{', excluding ' + str(filters['datasets_not_matching']) if filters.get('datasets_not_matching') else ''}"
                f"; Holm family {stratum.get('holm_family_size')}"
            )
        lines.append("")

    # --- excluded -----------------------------------------------------------
    excluded = report.get("excluded_from_contrast") or {}
    lines += ["## Excluded from contrast", ""]
    if not excluded:
        lines += ["Every pre-registered contrast was computed.", ""]
    else:
        lines += [
            "A refused contrast is a result about the design, not a gap to be filled by",
            "widening a gate. Each is listed with the reason the aggregator gave:",
            "",
        ]
        for key, reason in excluded.items():
            lines.append(f"- **{key}** — {reason}")
        lines.append("")

    # --- provenance ---------------------------------------------------------
    lines += ["## Gates and calibration provenance", ""]
    if gates:
        lines += [
            f"Preflight record: `{args.gates or args.results / 'alignment_gates.json'}`.",
            "",
            "| Gate | Verdict |",
            "|---|---|",
        ]
        for name, verdict in sorted((gates.get("checks") or gates).items()):
            lines.append(f"| {name} | {verdict if isinstance(verdict, str) else json.dumps(verdict)} |")
        lines.append("")
    else:
        lines += [
            "**No preflight record found.** The two-stage gates of section 7.4 — that every",
            "episode resolves to a non-empty supplied label list rather than falling through",
            "to generation, that seed and eval do not intersect, that every calibration file",
            "records the labels it applies to, that the fitting and scoring configurations",
            "match field by field — are what stand between this report and a set of numbers",
            "produced by a different pipeline than the one described. Their absence is not",
            "evidence that they passed.",
            "",
        ]
    if calibration:
        lines += [
            "| Component | Fit episodes | Gate verdict | Obeyed |",
            "|---|---|---|---|",
        ]
        for component, record in sorted(calibration.items()):
            if not isinstance(record, dict):
                continue
            lines.append(
                f"| `{component}` | {record.get('fit_episode_count', '—')} | "
                f"{record.get('gate', '—')} | {record.get('obeyed', 'recorded, not obeyed')} |"
            )
        lines += [
            "",
            "Every component is calibrated, including the ones the tool's own gate advises",
            "against, because letting the gate filter the corpus would condition the",
            "calibration claim on the population where calibration already works. The gate's",
            "verdict is recorded and becomes a prediction to test, not an exclusion rule.",
            "",
        ]

    # --- limitations --------------------------------------------------------
    lines += [
        "## Limitations",
        "",
        "1. **One model.** Nothing here separates the method from Qwen3.8-27B.",
        "2. **One annotation team, one collection effort** across all 16 components.",
        "   Calibration measures a convention; a different team needs a refit.",
        "3. **The calibration gate is measured on the data that produced it.** These same",
        "   components produced its thresholds, so re-measuring re-measures; it does not",
        "   independently validate.",
        "4. **Development adjacency.** The prompts, frame format and calibration design were",
        "   developed on a fridge-drink sibling of four components in this corpus.",
        "5. **Encoding and prompt are confounded** in the native-video arms: that path",
        "   changed both, so its gain cannot be attributed to video rather than wording.",
        "6. **Per-episode label lists are optimistic.** A real user rarely has them. The",
        "   realistic case — one shared list per collection — is a different question and is",
        "   out of scope here.",
        "7. **Decoding is not deterministic.** Temperature is 0.2 and no seed is sent to the",
        "   server, so any effect smaller than the measured repeat band is not an effect.",
        "8. **Even solver-placed output can drop labels**, so `placed_fraction` is reported",
        "   for every arm rather than assumed to be 1.0.",
        "",
    ]

    document = "\n".join(lines) + "\n"

    # --- the render check ---------------------------------------------------
    # The stamp is the condition on which a cross-level contrast was computed at
    # all. Verifying it in the finished text, rather than trusting the code above
    # to have emitted it, is what makes that condition enforceable.
    problems: list[str] = []
    for key, text in (report.get("supervision_asymmetry") or {}).items():
        if key not in stamps:
            problems.append(f"'{key}' has a supervision stamp that no section rendered")
        if text not in document:
            problems.append(f"the supervision stamp for '{key}' is not in the document text")
        # Identify the contrast's own rows by (arm, baseline) rather than by arm
        # alone: one arm is the numerator of several contrasts here, and a marker
        # on a sibling's row would satisfy a check keyed on the arm.
        signature = (f"`{report.get('contrast_arms', {}).get(key)}` vs "
                     f"`{report.get('contrast_baselines', {}).get(key)}`")
        rendered = [line for line in primary_rows + secondary_rows if signature in line]
        if not rendered:
            problems.append(f"no table row was rendered for '{key}'")
        elif any(CROSS_LEVEL not in line for line in rendered):
            unmarked = sum(CROSS_LEVEL not in line for line in rendered)
            problems.append(
                f"{unmarked} of {len(rendered)} rows for '{key}' do not carry the "
                f"{CROSS_LEVEL} marker"
            )
    if problems:
        print(
            "REFUSED: the report would drop a supervision-asymmetry stamp:\n  "
            + "\n  ".join(problems)
            + "\nA cross-level contrast is computed only on condition that its asymmetry is "
            "rendered with every number it produced. Nothing was written.",
            file=sys.stderr,
        )
        return 1

    target.write(document)
    print(f"wrote {args.out} "
          f"({len(contrasts)} contrasts, {report.get('holm_family_size', 0)} Holm tests, "
          f"{len(stamps)} cross-level stamp(s) rendered)")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
