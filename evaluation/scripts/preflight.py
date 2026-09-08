#!/usr/bin/env python
"""Integrity assertions that must pass before any GPU work.

Every claim in the evaluation plan about the two tools being comparable is
checked here mechanically, at run time, against the installed code. The plan
says the prompts are identical and the defaults differ in only two operational
fields; if a future version of either package changes that, the comparison
silently stops being like-for-like. This script turns those claims into a gate
rather than a footnote.

Exits non-zero on any failure. ``run_all.sh preflight`` runs it before the
first model is loaded.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses as dc
import hashlib
import io
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

# Config fields expected to differ between the two packages. Both are
# operational (which container image a remote job uses, which lerobot ref it
# installs) and neither can affect annotation output. Any OTHER difference is
# a confound and fails the gate.
ALLOWED_DEFAULT_DIFFERENCES = {"job.image", "job.lerobot_ref"}

# Prompts the new tool adds. Their absence upstream is expected; what must not
# happen is a shared prompt being modified.
NEW_TOOL_ONLY_PROMPTS = {
    "plan_subtask_align.txt",
    "plan_subtask_align_video.txt",
    "plan_subtask_align_video_onset.txt",
    "plan_subtask_describe_video.txt",
    "plan_subtasks_video.txt",
}


# Arm definitions checked by default. Both studies keep their arms in
# ``configs/``: the generation grid in ``arms.yaml`` and the fixed-label
# alignment arms in ``alignment_arms.yaml``. The set is NAMED rather than
# globbed so a superseded file (``arms_v1_upstream_defaults.yaml``) is not
# swept in by accident, and a named file that does not exist yet is skipped
# with a note -- a file passed explicitly with ``--arms-config`` must exist.
DEFAULT_ARM_CONFIGS = ("arms.yaml", "alignment_arms.yaml")

# How an arm's comparison baseline is declared, most specific first.
#
# ``pairs_with: <arm>`` on the arm names its baseline outright;
# ``pairs_with: none`` (or null) declares an arm that is a baseline itself or is
# reported on its own. A file-level ``upstream_counterparts: false`` says the
# same thing for every arm in that file, which is what the fixed-label study
# needs: upstream has no fixed-label mode (alignment_plan.md §5.3), so no
# ``baseline_upstream__*`` counterpart exists for any of its arms, and its real
# pairings are enumerated in ``configs/alignment_contrasts.yaml``.
PAIRS_WITH = "pairs_with"
UPSTREAM_COUNTERPARTS = "upstream_counterparts"
STANDALONE = {"", "none", "null"}

# Placeholders ``run_arm.py`` substitutes at launch, and the contents a
# preflight stand-in must carry to be a legitimate substitute: a label list the
# tool's reader accepts, and a calibration with one offset per internal
# boundary of that list.
PLACEHOLDER_CONTENT = {
    "{subtasks_path}": json.dumps(["pick up the cup", "put the cup down"]),
    "{calibration_path}": json.dumps(
        {
            "mode": "fraction_of_duration",
            "offsets": [0.05],
            "labels": ["pick up the cup", "put the cup down"],
            "fit_episode_count": 10,
        }
    ),
}

# A placeholder is a whole ``{name}`` token, not merely a value containing a
# brace: ``vlm.chat_template_kwargs`` is the JSON literal ``{"enable_thinking":
# false}``, and treating that as a placeholder would report every arm in the
# generation grid as broken.
PLACEHOLDER_TOKEN = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*\}")

# Sentinel for a flag whose parsed value legitimately differs from its
# command-line spelling, so only its NAME can be checked.
UNCOMPARED = object()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flatten_defaults(cls: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in dc.fields(cls):
        try:
            if field.default is not dc.MISSING:
                default = field.default
            elif field.default_factory is not dc.MISSING:  # type: ignore[misc]
                default = field.default_factory()  # type: ignore[misc]
            else:
                default = None
        except Exception:
            default = None
        if dc.is_dataclass(default):
            out.update(flatten_defaults(type(default), f"{prefix}{field.name}."))
        else:
            out[f"{prefix}{field.name}"] = default
    return out


def check_prompt_overrides(problems: list[str], notes: list[str]) -> None:
    """Refuse to run while a runtime prompt override is active.

    Both tools honour ``LEROBOT_PROMPT_OVERRIDE_<name>``, which replaces the
    packaged prompt with the environment's value. It is a supported feature,
    but it silently invalidates this study's central fairness claim: comparing
    prompt FILES proves nothing when the text actually sent comes from the
    environment, and an override present for one arm and absent for another
    would break the equalisation with every file check still passing.

    Set ``ALLOW_PROMPT_OVERRIDES=1`` to proceed deliberately -- in which case
    the override is part of the experiment and belongs in the write-up.
    """
    import os

    active = sorted(
        key[len("LEROBOT_PROMPT_OVERRIDE_"):]
        for key, value in os.environ.items()
        if key.startswith("LEROBOT_PROMPT_OVERRIDE_") and value.strip()
    )
    if not active:
        notes.append("prompt overrides: none active (packaged prompts will be used)")
        return
    if os.environ.get("ALLOW_PROMPT_OVERRIDES") == "1":
        notes.append(
            f"prompt overrides ACTIVE and explicitly allowed: {active}. "
            "These must be reported as part of the experimental condition."
        )
        return
    problems.append(
        f"runtime prompt override(s) active: {active}. The packaged prompts will NOT be "
        "sent, so the byte-identical-prompt fairness claim does not hold. Unset them, or "
        "set ALLOW_PROMPT_OVERRIDES=1 to proceed deliberately and report them."
    )


def check_prompts(problems: list[str], notes: list[str]) -> None:
    import lerobot.annotations.steerable_pipeline.prompts as upstream_prompts
    import lerobot_align.prompts as align_prompts

    upstream_dir = Path(upstream_prompts.__file__).parent
    align_dir = Path(align_prompts.__file__).parent
    upstream_files = {p.name for p in upstream_dir.glob("*.txt")}
    align_files = {p.name for p in align_dir.glob("*.txt")}

    shared = sorted(upstream_files & align_files)
    modified = [n for n in shared if digest(upstream_dir / n) != digest(align_dir / n)]
    if modified:
        problems.append(
            f"shared prompt(s) MODIFIED in lerobot-align: {modified}. "
            "The comparison is no longer prompt-controlled."
        )
    notes.append(f"prompts: {len(shared)} shared, all byte-identical" if not modified else "")

    missing_upstream = sorted(upstream_files - align_files)
    if missing_upstream:
        problems.append(f"lerobot-align is missing upstream prompt(s): {missing_upstream}")

    unexpected_new = sorted((align_files - upstream_files) - NEW_TOOL_ONLY_PROMPTS)
    if unexpected_new:
        notes.append(f"note: undeclared new prompts present (not a failure): {unexpected_new}")


def inert_default(value: Any) -> bool:
    """Type-aware sentinels: True/1 must never alias False/0 in a set."""
    return (value is None or value is False
            or (type(value) in (int, float) and value == 0)
            or (isinstance(value, str) and value in ("off", "contact_sheet", "uniform", ""))
            or (isinstance(value, tuple) and not value))


def check_defaults(problems: list[str], notes: list[str]) -> None:
    from lerobot.annotations.steerable_pipeline.config import AnnotationPipelineConfig as Upstream
    from lerobot_align.config import AnnotationPipelineConfig as Align

    upstream = flatten_defaults(Upstream)
    align = flatten_defaults(Align)

    shared = sorted(set(upstream) & set(align))
    differing = {k for k in shared if repr(upstream[k]) != repr(align[k])}
    unexpected = differing - ALLOWED_DEFAULT_DIFFERENCES
    if unexpected:
        problems.append(
            "shared config defaults differ beyond the allowed operational set: "
            + ", ".join(f"{k} ({upstream[k]!r} vs {align[k]!r})" for k in sorted(unexpected))
        )
    notes.append(
        f"defaults: {len(differing)}/{len(shared)} shared fields differ "
        f"({sorted(differing)}), {len(set(align) - set(upstream))} fields new to lerobot-align"
    )

    dropped = sorted(set(upstream) - set(align))
    if dropped:
        problems.append(f"lerobot-align dropped upstream config field(s): {dropped}")

    # Every field the new tool adds must default to something inert, or the
    # "align_defaults == baseline_upstream" falsification arm is not testing
    # what the plan says it tests.
    active = {
        k: align[k] for k in sorted(set(align) - set(upstream))
        if not inert_default(align[k])
    }
    if active:
        notes.append(f"note: new fields with non-inert defaults (verify intent): {active}")


# The pre-registered Corpus A. Asserted by name, not by counting whatever
# happens to be on disk: globbing would silently accept a corpus that had
# gained a machine-annotated component or lost a human one.
EXPECTED_COMPONENTS = (
    "base4-clean-table-01-BC-FV-v30",
    "base4-clean-table-02-BC-v30",
    "base4-clean-table-03-BC-FV-v30",
    "base4-clean-table-04-BC-FV-v30",
    "base4-mobile-door-01-BC-FV-v30",
    "base4-mobile-door-02-BC-FV-v30",
    "base4-mobile-door-04-BC-FV-v30",
    "u850-bag-place-01-FV-v30",
    "u850-bag-place-02-FV-v30",
    "u850-bag-place-03-BC-v30",
    "u850-bag-place-04-BC-v30",
    "u850-fridge-drink-01-FV-v30",
    "u850-fridge-drink-02-v30",
    "u850-fridge-drink-03-v30",
    "u850-fridge-drink-04-v30",
    "u850-fridge-drink-05-v30",
)


def check_corpus(gt_dir: Path, problems: list[str], notes: list[str]) -> None:
    present = {p.stem for p in gt_dir.glob("*.json")}
    expected = set(EXPECTED_COMPONENTS)
    missing = sorted(expected - present)
    extra = sorted(present - expected)
    if missing:
        problems.append(f"corpus is missing pre-registered component(s): {missing}")
    if extra:
        problems.append(
            f"corpus contains component(s) not in the pre-registration: {extra}. "
            "Every one of these must be checked for machine-generated labels before use."
        )
    if not missing and not extra:
        notes.append(f"corpus: exactly the {len(expected)} pre-registered components")


def check_splits(gt_dir: Path, splits_dir: Path, problems: list[str], notes: list[str]) -> None:
    gt_files = sorted(gt_dir.glob("*.json"))
    if not gt_files:
        problems.append(f"no ground truth under {gt_dir}")
        return
    total_eval = 0
    for path in gt_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        dataset = payload["dataset"]
        split_path = splits_dir / f"{dataset}.json"
        if not split_path.exists():
            problems.append(f"{dataset}: no split file")
            continue
        split = json.loads(split_path.read_text(encoding="utf-8"))
        seed, evaluation = set(split["seed"]), set(split["eval"])
        if seed & evaluation:
            problems.append(f"{dataset}: seed and eval overlap: {sorted(seed & evaluation)}")
        available = {int(k) for k in payload["episodes"]}
        missing = evaluation - available
        if missing:
            problems.append(f"{dataset}: eval episodes without ground truth: {sorted(missing)[:5]}")
        if not seed:
            problems.append(f"{dataset}: empty seed split")
        total_eval += len(evaluation)
    notes.append(f"splits: {len(gt_files)} datasets, {total_eval} evaluation episodes, seed∩eval=∅")


def load_spec(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def baseline_expectation(
    arm: dict[str, Any], infer_from_names: bool = True
) -> tuple[str, str | None]:
    """Which arm this one must be able to pair against, and how that is known.

    Returns ``(kind, baseline)`` with kind ``declared``, ``inferred``,
    ``standalone`` or ``undeclared``.

    The DECLARATION comes first, and the name template is only a fallback for
    the generation grid that predates it. Inferring the counterpart from
    ``<tool>__<profile>__<camera>`` cannot describe a fixed-label alignment arm
    at all: upstream has no fixed-label mode (alignment_plan.md §5.3), so those
    arms have no upstream counterpart to name. Left to the template they either
    demand a `baseline_upstream__*` arm that cannot exist -- failing the whole
    study on a naming convention -- or, having no ``profile``, fall through the
    old `not profile` skip and quietly check nothing at all.
    """
    if PAIRS_WITH in arm:
        declared = arm[PAIRS_WITH]
        if declared is None or str(declared).strip().lower() in STANDALONE:
            return "standalone", None
        return "declared", str(declared)
    if not infer_from_names:
        return "standalone", None
    # An upstream cell baseline IS the baseline; nothing to pair it against.
    if arm.get("base_arm") == "baseline_upstream":
        return "standalone", None
    profile, camera = arm.get("profile"), arm.get("camera")
    if not profile:
        return "undeclared", None
    expected = f"baseline_upstream__{profile}__{camera}"
    # Stacked arms have no upstream counterpart by design; they pair against
    # the single-camera realign arm instead.
    if str(camera).startswith("stack"):
        expected = f"align_video_realign__{profile}__wrist"
    return "inferred", expected


def check_grid(config_paths: list[Path], problems: list[str], notes: list[str]) -> None:
    """The arm set must be able to produce the contrasts it promises.

    Renaming every arm to `<tool>__<profile>__<camera>` silently invalidated
    every hardcoded reference to the old names, so the grid could be fully
    parseable and still unable to run or to pair anything. These checks assert
    the two properties that renaming broke: each arm's comparison baseline
    exists, and no upstream arm carries a flag upstream does not have.
    """
    from lerobot.annotations.steerable_pipeline.config import (
        AnnotationPipelineConfig as UpstreamConfig,
    )

    def field_names(cls, prefix=""):
        out = set()
        for f in dc.fields(cls):
            default = f.default if f.default is not dc.MISSING else (
                f.default_factory() if f.default_factory is not dc.MISSING else None)
            if dc.is_dataclass(default):
                out |= field_names(type(default), f"{prefix}{f.name}.")
            else:
                out.add(f"{prefix}{f.name}")
        return out

    upstream_fields = field_names(UpstreamConfig)

    for path in config_paths:
        spec = load_spec(path)
        arms = spec["arms"]
        # A file may declare that none of its arms has an upstream counterpart;
        # the name template is then not consulted at all. Without that
        # declaration the template is applied, which is what the generation grid
        # relies on.
        infer_from_names = bool(spec.get(UPSTREAM_COUNTERPARTS, True))
        names = {a["name"] for a in arms}
        missing_baselines: list[str] = []
        alien_flags: list[str] = []
        undeclared: list[str] = []
        self_paired: list[str] = []
        kinds: dict[str, int] = {}
        for arm in arms:
            if arm["tool"] == "reference":
                continue
            if arm["tool"] == "upstream":
                for key in (arm.get("flags") or {}):
                    if key not in upstream_fields:
                        alien_flags.append(f"{arm['name']}:{key}")
            kind, baseline = baseline_expectation(arm, infer_from_names)
            kinds[kind] = kinds.get(kind, 0) + 1
            if kind == "undeclared":
                undeclared.append(arm["name"])
            elif baseline == arm["name"]:
                self_paired.append(arm["name"])
                problems.append(f"arm '{arm['name']}' declares itself as its own baseline")
            elif baseline is not None and baseline not in names:
                missing_baselines.append(f"{arm['name']} -> {baseline} ({kind})")

        if alien_flags:
            problems.append(
                f"[{path.name}] upstream arm(s) carry flags upstream does not define "
                f"(upstream has no such field, so the run would abort): {alien_flags}"
            )
        if undeclared:
            problems.append(
                f"[{path.name}] arm(s) with neither a `{PAIRS_WITH}` declaration nor a "
                f"(profile, camera) to infer one from, so nothing checks that their "
                f"contrast can be run: {undeclared}. Declare `{PAIRS_WITH}: <arm>`, or "
                f"`{PAIRS_WITH}: none` for an arm that is a baseline itself or is "
                "reported on its own."
            )
        if missing_baselines:
            problems.append(
                f"[{path.name}] arm(s) whose comparison baseline does not exist: "
                f"{missing_baselines}. An `inferred` baseline comes from the generation "
                f"grid's `<tool>__<profile>__<camera>` template; an arm that does not "
                f"follow it must declare `{PAIRS_WITH}: <arm>` (or `{PAIRS_WITH}: none`), "
                f"or the whole file must declare `{UPSTREAM_COUNTERPARTS}: false`."
            )
        if not (alien_flags or undeclared or missing_baselines or self_paired):
            vlm = [a for a in arms if a["tool"] != "reference"]
            summary = ", ".join(f"{n} {k}" for k, n in sorted(kinds.items()))
            cells = sorted({(a.get("profile"), a.get("camera")) for a in vlm
                            if a.get("profile")})
            cell_note = f" across {len(cells)} (profile, camera) cells" if cells else ""
            if infer_from_names:
                verdict = "every comparison baseline exists"
            else:
                # Said plainly: this file opts out of the name template, so the
                # only pairing check left is whatever `pairs_with` an arm
                # declares. The study's real contrast family lives in its
                # contrasts config and is validated there, not here.
                verdict = (
                    f"{UPSTREAM_COUNTERPARTS}: false -- no baseline is inferred from arm "
                    "names; the contrast family is this study's contrasts config"
                )
            notes.append(
                f"grid [{path.name}]: {len(vlm)} VLM arms{cell_note} ({summary}); {verdict}"
            )


def render_flags(
    arm: dict[str, Any], substitutes: dict[str, Path], problems: list[str]
) -> list[tuple[str, Any, Any]]:
    """One arm's flags as (name, command-line value, expected parsed value).

    A flag value carrying a ``{placeholder}`` gets a REAL temporary file, the
    same substitution ``run_arm.py`` performs at launch. Skipping such flags --
    which is what "contains a brace" did -- meant the label-file flag was never
    parsed, so a misnamed one (`plan.subtask_path` for `plan.subtasks_path`)
    passed preflight and the arm ran in GENERATION mode, inventing the labels
    the study exists to supply.

    ``expected`` is ``UNCOMPARED`` where the parsed value legitimately differs
    from its command-line spelling: ``vlm.chat_template_kwargs`` is a JSON
    object literal that parses back as a dict. Those flags still go on the
    command line, so the field NAME is validated even when the value cannot be
    compared.
    """
    rendered: list[tuple[str, Any, Any]] = []
    for key, value in sorted((arm.get("flags") or {}).items()):
        if value is None:
            continue
        if not isinstance(value, str):
            rendered.append((key, value, value))
            continue
        substituted = value
        for token in dict.fromkeys(PLACEHOLDER_TOKEN.findall(value)):
            replacement = substitutes.get(token)
            if replacement is None:
                problems.append(
                    f"arm '{arm['name']}' flag {key}: unknown placeholder {token}. "
                    f"run_arm.py substitutes only {sorted(substitutes)}, so this would "
                    "reach the tool as literal text."
                )
                substituted = None
                break
            substituted = substituted.replace(token, str(replacement))
        if substituted is None:
            continue
        if substituted != value:
            rendered.append((key, substituted, substituted))
        elif "{" in value:
            rendered.append((key, value, UNCOMPARED))
        else:
            rendered.append((key, value, value))
    return rendered


def check_arm_flags(config_paths: list[Path], problems: list[str], notes: list[str]) -> None:
    """Every arm's flags must PARSE and round-trip to the intended values.

    Parsing is not enough. `plan.subtask_align_camera_keys` is a
    `tuple[str, ...]`, and a bare comma-separated string parses without error
    into a tuple of individual characters -- so the multiview arm ran as a
    silent single-camera duplicate of another arm while looking correct in
    every log. This check renders each arm's real command line, parses it with
    the tool's own config class, and compares the resulting VALUES against what
    the YAML asked for.
    """
    import draccus

    from lerobot.annotations.steerable_pipeline.config import (
        AnnotationPipelineConfig as UpstreamConfig,
    )
    from lerobot_align.config import AnnotationPipelineConfig as AlignConfig

    with tempfile.TemporaryDirectory(prefix="preflight-arms-") as scratch:
        root = Path(scratch)
        substitutes = {}
        for token, content in PLACEHOLDER_CONTENT.items():
            stand_in = root / f"{token.strip('{}')}.json"
            stand_in.write_text(content, encoding="utf-8")
            substitutes[token] = stand_in

        for path in config_paths:
            checked = uncompared = substituted = total = 0
            for arm in load_spec(path)["arms"]:
                if arm["tool"] == "reference":
                    continue
                total += 1
                config_cls = UpstreamConfig if arm["tool"] == "upstream" else AlignConfig
                flags = render_flags(arm, substitutes, problems)
                argv = [f"--root={root / 'dataset'}"]
                for key, value, _ in flags:
                    rendered = "true" if value is True else "false" if value is False else value
                    argv.append(f"--{key}={rendered}")
                # draccus parses with argparse, which does NOT raise on an
                # unknown field: it prints usage and calls sys.exit. Catching
                # only Exception let that SystemExit escape and kill preflight
                # where it stood, taking every later check with it and burying
                # the reason under a page of usage text. Catch it, and keep the
                # one line that names the offending flag.
                stderr = io.StringIO()
                try:
                    with contextlib.redirect_stderr(stderr):
                        parsed = draccus.parse(config_cls, args=argv)
                except (Exception, SystemExit) as exc:  # noqa: BLE001
                    detail = next(
                        (line for line in reversed(stderr.getvalue().splitlines()) if line.strip()),
                        f"{type(exc).__name__}: {exc}",
                    )
                    problems.append(f"arm '{arm['name']}' does not parse: {detail}")
                    continue
                for key, _, expected in flags:
                    if expected is UNCOMPARED:
                        uncompared += 1
                        continue
                    # A flag whose expected value is one of the stand-in files
                    # carried a placeholder and was parsed against a real file.
                    if isinstance(expected, str) and str(root) in expected:
                        substituted += 1
                    target = parsed
                    for part in key.split("."):
                        target = getattr(target, part, None)
                    if isinstance(expected, str) and expected.startswith("[") \
                            and expected.endswith("]"):
                        expected = tuple(
                            v.strip().strip("'\"") for v in expected[1:-1].split(",") if v.strip()
                        )
                    if isinstance(target, tuple) and isinstance(expected, tuple):
                        if target != expected:
                            problems.append(
                                f"arm '{arm['name']}' flag {key}: asked for {expected}, "
                                f"got {target!r} -- the value did not round-trip"
                            )
                    elif isinstance(expected, (int, float)) and isinstance(target, (int, float)):
                        if abs(float(target) - float(expected)) > 1e-9:
                            problems.append(
                                f"arm '{arm['name']}' flag {key}: asked for {expected}, "
                                f"got {target}"
                            )
                    elif str(target) != str(expected):
                        problems.append(
                            f"arm '{arm['name']}' flag {key}: asked for {expected!r}, "
                            f"got {target!r}"
                        )
                checked += 1
            # Only an unbroken file earns a note. Reporting "0 VLM arms parse
            # and every flag value round-trips" as an `ok` line, which is what
            # counting successes alone produced, reads as a pass.
            if checked == total:
                notes.append(
                    f"arms [{path.name}]: {checked} VLM arms parse and every flag value "
                    f"round-trips ({substituted} supplied-file placeholder(s) parsed against "
                    f"a real temporary file, {uncompared} JSON-valued flag(s) name-checked "
                    "only)"
                )


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--gt-dir", type=Path, default=None)
    parser.add_argument("--splits-dir", type=Path, default=None)
    parser.add_argument(
        "--arms-config", type=Path, action="append", default=None,
        help="Arm definitions to check; repeatable. Defaults to whichever of "
             + ", ".join(DEFAULT_ARM_CONFIGS) + " exist under configs/.",
    )
    args = parser.parse_args()

    problems: list[str] = []
    notes: list[str] = []

    configs_dir = Path(__file__).resolve().parent.parent / "configs"
    if args.arms_config:
        arm_configs = [p for p in args.arms_config if p.exists()]
        for path in args.arms_config:
            if not path.exists():
                problems.append(f"--arms-config {path} does not exist")
    else:
        arm_configs = [configs_dir / n for n in DEFAULT_ARM_CONFIGS if (configs_dir / n).exists()]
        absent = [n for n in DEFAULT_ARM_CONFIGS if not (configs_dir / n).exists()]
        if absent:
            # Named, not globbed: a study whose arms file has not been written
            # yet is reported here rather than being silently unchecked.
            notes.append(f"arm configs: {absent} absent, nothing checked for them")
    if not arm_configs:
        problems.append("no arm config to check; pass --arms-config")

    check_prompt_overrides(problems, notes)
    check_prompts(problems, notes)
    check_defaults(problems, notes)
    check_arm_flags(arm_configs, problems, notes)
    check_grid(arm_configs, problems, notes)
    if args.gt_dir and args.splits_dir:
        check_corpus(args.gt_dir, problems, notes)
        check_splits(args.gt_dir, args.splits_dir, problems, notes)
    else:
        notes.append("splits: SKIPPED (no --gt-dir/--splits-dir given)")

    print("=" * 72)
    print("PREFLIGHT")
    print("=" * 72)
    for note in notes:
        if note:
            print(f"  ok   {note}")
    if problems:
        print()
        for problem in problems:
            print(f"  FAIL {problem}", file=sys.stderr)
        print(f"\n{len(problems)} integrity check(s) failed; refusing to start.", file=sys.stderr)
        return 1
    print("\nAll integrity checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
