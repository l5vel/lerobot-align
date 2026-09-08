#!/usr/bin/env python
"""Write the supplied-label JSON files that drive fixed-label alignment.

Study 2 (`alignment_plan.md`) hands the tool an ordered label list and asks it
only *when* each label happens. The labels travel to the tool in a JSON file
named by `--plan.subtasks_path`, so these files are the study's independent
variable: two arms can differ by nothing but which file they were given.

**The failure this script exists to prevent.** `_load_subtasks_file`
(`plan_subtasks_memory.py`) accepts either a flat list -- applied to every
episode -- or an object keyed by episode index with an optional `"default"`.
`_GivenSubtasks.for_episode` is `by_episode.get(index, self.default)`, and
`default` is `[]` when the file omits it. An episode absent from an
object-keyed file with no default therefore resolves to an EMPTY label list,
and `_subtask_spans` then falls through to open-ended **generation**: the run
produces plausible output, costs GPU time, and silently answers a different
question than the one being asked. Nothing downstream would notice, because a
generated span list is shaped exactly like an aligned one. That is plan §7.4.2
/ defect D3, and the tool is not modified to fix it (the standing rule), so the
guarantee has to live here: every file emitted below carries an **explicit
entry for every episode it covers**, and `--verify` re-reads the files through
the tool's own loader and asserts that every eval episode resolves to a
non-empty list *from its own entry rather than from a default*.

Two label sources, per plan §4.2:

``oracle``  each episode gets its own ground-truth label list. This is the
            study -- the direct reading of "a dataset that keeps its subtasks
            and loses their timestamps". Boundaries are never supplied.
``modal``   one list per component, derived from the **seed episodes only**.
            Deriving it from all 50 would read held-out ground truth, which is
            plan gate §7.4.3. Episodes whose true list differs still receive
            the modal list; they cannot be scored and are reported as coverage
            loss (§2.1b) rather than dropped.

**Tie-break.** "The modal list of the seed episodes" is undefined when two
lists tie, which happens on `u850-fridge-drink-03-v30`: its ten seed episodes
split 5/5 between a 6-label convention and the same list without a leading
"move closer to the fridge". The rule is **the longer list wins**, and the
choice, the losing candidates and the rule that decided it are all recorded in
the manifest. Any rule would do; leaving it undefined would not.

Ties are resolved over an insertion-ordered `Counter`, never over a `set`.
That is deliberate: `fit_align_calibration.py:229-234` picks its fit set with
`max()` over a set and so resolves ties by **hash order**, which gives that
fitter a different answer on different runs (defect D7). A label file that
changed identity between runs would be the same bug one layer up.

The two files differ in one further respect, and it is intentional:

* the ``modal`` file carries a ``"default"`` as well as per-episode entries.
  Both routes yield the same single list, so the fall-through cannot change
  the answer -- it only removes a way to fail.
* the ``oracle`` file carries **no** default. There is no per-episode list that
  could serve as one, and a default would make the file claim to cover
  episodes whose labels it does not know, which plan gate §7.4.1 checks. The
  explicit entries plus `--verify` are the guarantee instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Span = dict[str, Any]
LabelList = tuple[str, ...]

ORACLE = "oracle"
MODAL = "modal"
SOURCES = (ORACLE, MODAL)

# Cap on printed failures; the count is always reported in full.
MAX_REPORTED_PROBLEMS = 20


def digest(path: Path) -> str:
    """SHA-256 of a written file.

    Recorded because the run cache keys on the label file's *path* while
    hashing the calibration file's contents (defect D1). Until that is fixed,
    an edited-in-place label file serves predictions made against the old
    labels; the manifest at least makes the substitution detectable after the
    fact.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label_list(spans: list[Span]) -> LabelList:
    """The episode's ordered labels, verbatim.

    Verbatim matters twice over: `_coerce_label_list` strips each string on the
    way in, and `evaluate_alignment` pairs predictions to ground truth by exact
    string equality. A label that does not survive `.strip()` unchanged would
    be supplied as one string and scored against another, and every span of
    that episode would read as a miss. `check_labels` refuses such a label
    rather than emitting it.
    """
    return tuple(str(span["text"]) for span in spans)


@dataclass(frozen=True)
class Component:
    """One dataset's ground truth and its frozen seed/eval split."""

    dataset: str
    truth: dict[int, LabelList]
    seed: list[int]
    evaluation: list[int]

    @property
    def covered_annotated(self) -> list[int]:
        return sorted(self.truth)


@dataclass(frozen=True)
class ModalChoice:
    """The seed-derived modal list, plus everything needed to audit it."""

    labels: LabelList
    seed_count: int
    n_seed_considered: int
    tie_broken: bool
    rule: str
    candidates: list[dict[str, Any]]


def read_component(gt_path: Path, split_path: Path) -> Component:
    truth_payload = json.loads(gt_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    dataset = str(truth_payload["dataset"])
    if str(split.get("dataset", dataset)) != dataset:
        raise SystemExit(
            f"{split_path}: split is for {split['dataset']!r}, ground truth for {dataset!r}"
        )
    truth = {
        int(episode): label_list(spans)
        for episode, spans in truth_payload["episodes"].items()
        if spans
    }
    return Component(
        dataset=dataset,
        truth=truth,
        seed=sorted({int(e) for e in split["seed"]}),
        evaluation=sorted({int(e) for e in split["eval"]}),
    )


def choose_modal(component: Component) -> ModalChoice:
    """Modal seed label list, with the tie broken explicitly.

    Takes the whole component but reads ONLY `component.seed`. Passing the
    component rather than a pre-filtered list keeps that restriction in one
    place where it can be read, and plan gate §7.4.3 -- "modal_list files are
    derived from seed episodes only" -- is then a property of this function
    rather than of every caller.
    """
    counts: Counter[LabelList] = Counter()
    first_seen: dict[LabelList, int] = {}
    for episode in component.seed:  # sorted, so insertion order is reproducible
        labels = component.truth.get(episode)
        if labels is None:
            continue
        counts[labels] += 1
        first_seen.setdefault(labels, episode)
    if not counts:
        raise SystemExit(f"{component.dataset}: no seed episode carries ground truth")

    top = max(counts.values())
    # Counter preserves insertion order, so `tied` is in ascending seed-episode
    # order. Iterating a set here is what makes D7's fit-set selector unstable.
    tied = [labels for labels, count in counts.items() if count == top]
    longest = max(len(labels) for labels in tied)
    finalists = [labels for labels in tied if len(labels) == longest]
    chosen = finalists[0]

    if len(tied) == 1:
        rule = "unique modal list among the seed episodes"
    elif len(finalists) == 1:
        rule = f"{len(tied)}-way tie at {top} seed episodes; longest list wins (plan §4.2)"
    else:
        # Not reachable on Corpus A, but the rule must still be total: two
        # equally frequent lists of equal length would otherwise reintroduce
        # exactly the undefined choice §4.2 was written to close.
        rule = (
            f"{len(tied)}-way tie at {top} seed episodes, {len(finalists)} of equal "
            f"length; lowest-numbered seed episode wins"
        )
    return ModalChoice(
        labels=chosen,
        seed_count=top,
        n_seed_considered=sum(counts.values()),
        tie_broken=len(tied) > 1,
        rule=rule,
        # Every distinct seed list, not only the tied ones: the audit question
        # is "how close was this call?", which the runner-up answers and a
        # bare winner does not.
        candidates=[
            {
                "labels": list(labels),
                "n_labels": len(labels),
                "seed_episodes": counts[labels],
                "first_seed_episode": first_seen[labels],
                "tied_for_top": labels in tied,
                "chosen": labels == chosen,
            }
            for labels in sorted(counts, key=lambda t: (-counts[t], -len(t), first_seen[t]))
        ],
    )


def label_count_distribution(component: Component, episodes: list[int]) -> dict[str, int]:
    counts = Counter(len(component.truth[e]) for e in episodes if e in component.truth)
    return {str(k): counts[k] for k in sorted(counts)}


def build_payload(
    component: Component, source: str, episodes: list[int], modal: ModalChoice
) -> dict[str, Any]:
    """The JSON object handed to `--plan.subtasks_path`.

    One entry per covered episode in both cases: that, not the default, is what
    makes the §7.4.2 fall-through unreachable.
    """
    payload: dict[str, Any] = {}
    if source == MODAL:
        # Written first purely so a human opening the file sees the list once
        # before 50 repetitions of it. The loader ignores key order.
        payload["default"] = list(modal.labels)
    for episode in episodes:
        payload[str(episode)] = (
            list(component.truth[episode]) if source == ORACLE else list(modal.labels)
        )
    return payload


def check_labels(component: Component, episodes: list[int], problems: list[str]) -> None:
    """Refuse labels the tool would silently alter or reject."""
    for episode in episodes:
        labels = component.truth.get(episode)
        if not labels:
            problems.append(f"{component.dataset} ep {episode}: no ground-truth labels to supply")
            continue
        for position, label in enumerate(labels):
            if not label.strip():
                problems.append(
                    f"{component.dataset} ep {episode} label {position}: empty; "
                    "_coerce_label_list would reject the whole file"
                )
            elif label != label.strip():
                problems.append(
                    f"{component.dataset} ep {episode} label {position}: {label!r} does not "
                    "survive .strip(); supplied and scored text would differ"
                )


_LOADER: Any = None


def tool_loader() -> Any:
    """The tool's own `_load_subtasks_file`.

    Verification reads the files back through the real parser rather than a
    re-implementation of it. A private import is the point: the guarantee being
    made is about what *this* loader does with these files, and a local copy of
    its rules would be free to drift away from the code under test.
    """
    global _LOADER
    if _LOADER is None:
        # Importing the tool pulls in torch. Nothing here needs a device, and
        # the standing constraint is that GPU 4 is never touched, so the safest
        # thing is to make the whole script structurally CPU-only. `setdefault`
        # leaves an explicit caller setting alone.
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

        from lerobot_align.modules.plan_subtasks_memory import _load_subtasks_file

        _LOADER = _load_subtasks_file
    return _LOADER


def verify_file(
    path: Path, component: Component, source: str, modal: ModalChoice, problems: list[str]
) -> None:
    """Assert a written file cannot fall through to generation, and is correct."""
    if not path.exists():
        problems.append(f"{path}: missing")
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        problems.append(f"{path}: expected an episode-keyed object, got {type(payload).__name__}")
        return

    try:
        given = tool_loader()(path)
    except Exception as exc:  # noqa: BLE001 -- any failure here is a hard gate failure
        problems.append(f"{path}: the tool's own loader rejected this file: {exc}")
        return

    for episode in component.evaluation:
        expected = list(component.truth[episode]) if source == ORACLE else list(modal.labels)
        # The trap, checked directly: presence in `by_episode`, not merely a
        # non-empty result, which a default would also produce.
        if episode not in given.by_episode:
            problems.append(
                f"{path}: eval episode {episode} has no entry of its own; "
                "_load_subtasks_file would return the default (or [] -> GENERATION)"
            )
            continue
        resolved = given.for_episode(episode)
        if not resolved:
            problems.append(f"{path}: eval episode {episode} resolves to an empty list")
        elif resolved != expected:
            problems.append(f"{path}: eval episode {episode} labels differ from {source} source")

    if source == ORACLE:
        if given.default:
            problems.append(
                f"{path}: oracle files must not carry a 'default' -- it would claim "
                "coverage of episodes whose labels are unknown (plan §7.4.1)"
            )
        for episode, labels in given.by_episode.items():
            truth = component.truth.get(episode)
            if truth is None:
                problems.append(f"{path}: episode {episode} has no ground truth to match")
            elif tuple(labels) != truth:
                problems.append(f"{path}: episode {episode} does not match its ground-truth list")
            elif len(labels) != len(truth):
                # evaluate_alignment raises unless these are equal; catching it
                # here fails the study before the GPU work, not after.
                problems.append(f"{path}: episode {episode} label count differs from ground truth")
    else:
        if tuple(given.default) != modal.labels:
            problems.append(f"{path}: 'default' is not the recorded modal list")
        for episode, labels in given.by_episode.items():
            if tuple(labels) != modal.labels:
                problems.append(f"{path}: episode {episode} is not the modal list")


def process(
    component: Component,
    out_dir: Path,
    sources: list[str],
    cover: str,
    write: bool,
    problems: list[str],
) -> dict[str, Any]:
    """Write (or verify) one component's label files and return its manifest."""
    overlap = sorted(set(component.seed) & set(component.evaluation))
    if overlap:  # plan gate §7.4.4
        problems.append(f"{component.dataset}: seed and eval overlap on {overlap}")

    episodes = (
        component.covered_annotated if cover == "annotated" else sorted(component.evaluation)
    )
    uncovered = sorted(set(component.evaluation) - set(episodes))
    if uncovered:
        problems.append(f"{component.dataset}: eval episodes {uncovered} have no ground truth")
        episodes = sorted(set(episodes) | (set(component.evaluation) & set(component.truth)))
    check_labels(component, episodes, problems)

    modal = choose_modal(component)
    eval_matches = sum(1 for e in component.evaluation if component.truth.get(e) == modal.labels)
    seed_matches = sum(1 for e in component.seed if component.truth.get(e) == modal.labels)

    files: dict[str, Any] = {}
    for source in sources:
        path = out_dir / f"{component.dataset}__{source}.json"
        if write:
            payload = build_payload(component, source, episodes, modal)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        # Verified on the write path too: a file is only ever reported as
        # written after it has been read back through the tool's loader.
        verify_file(path, component, source, modal, problems)
        files[source] = {
            "path": path.name,
            "sha256": digest(path) if path.exists() else None,
            "n_entries": len(episodes),
            "has_default": source == MODAL,
        }

    if not write:
        # Verification also re-checks the files against the manifest written
        # beside them. This is the one check that catches an edit made AFTER
        # the files were generated -- which matters here more than it normally
        # would, because the run cache keys on the label file's path while
        # hashing the calibration file's contents (defect D1), so an in-place
        # edit otherwise serves predictions made against the previous labels.
        manifest_path = out_dir / f"{component.dataset}__labels_manifest.json"
        if not manifest_path.exists():
            problems.append(f"{manifest_path}: missing")
        else:
            recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
            for source, info in files.items():
                was = (recorded.get("files") or {}).get(source, {}).get("sha256")
                if was and was != info["sha256"]:
                    problems.append(
                        f"{component.dataset} {source}: contents changed since the manifest "
                        f"was written ({was[:12]} -> {(info['sha256'] or '')[:12]}); "
                        "regenerate rather than editing a label file in place"
                    )
            if (recorded.get("modal") or {}).get("labels") != list(modal.labels):
                problems.append(
                    f"{component.dataset}: manifest records a different modal list than "
                    "the seed episodes now yield"
                )

    return {
        "dataset": component.dataset,
        "n_episodes_with_ground_truth": len(component.truth),
        "coverage": {
            "policy": cover,
            "n_covered": len(episodes),
            "covered_episodes": episodes,
            "seed_episodes": component.seed,
            "eval_episodes": component.evaluation,
            "n_eval": len(component.evaluation),
            "eval_fully_covered": not set(component.evaluation) - set(episodes),
        },
        "files": files,
        "modal": {
            "labels": list(modal.labels),
            "n_labels": len(modal.labels),
            "derived_from": "seed episodes only (plan §7.4.3)",
            "seed_episodes_matching": modal.seed_count,
            "n_seed_considered": modal.n_seed_considered,
            "tie_broken": modal.tie_broken,
            "tie_break_rule": modal.rule,
            "candidates": modal.candidates,
            "eval_episodes_matching": eval_matches,
            "eval_match_fraction": round(eval_matches / len(component.evaluation), 4)
            if component.evaluation
            else None,
            "eval_coverage_loss": len(component.evaluation) - eval_matches,
            "seed_episodes_matching_check": seed_matches,
        },
        "label_count_distribution": {
            "seed": label_count_distribution(component, component.seed),
            "eval": label_count_distribution(component, component.evaluation),
            "covered": label_count_distribution(component, episodes),
        },
        "distinct_label_lists": {
            "seed": len({component.truth[e] for e in component.seed if e in component.truth}),
            "eval": len({component.truth[e] for e in component.evaluation if e in component.truth}),
            "all": len(set(component.truth.values())),
        },
        "absent_episode_behaviour": (
            "modal: 'default' applies"
            if MODAL in sources and ORACLE not in sources
            else "oracle: NO default -- an absent episode would fall through to generation, "
            "which is why every covered episode has an explicit entry (plan §7.4.2)"
        ),
    }


def prior_sweep_population(component: Component) -> int:
    """The prior sweep's held-out count for this component, for reconciliation.

    Plan §2.1b reconciles the prior sweep's "507 held-out episodes" as the
    modal-label-list episodes minus the ten seed episodes. Reproduced here
    because the number it predicts is NOT what this script's rule produces:
    that arithmetic takes the modal list over ALL fifty episodes (which reads
    held-out ground truth) and then assumes all ten seed episodes match it. On
    six of sixteen components they do not, and on `u850-fridge-drink-01-FV-v30`
    the all-episode modal is a different list from the seed-derived one. The
    seed-only rule this study mandates therefore covers MORE eval episodes than
    507, and the summary reports both so the gap is visible rather than read as
    a bug in either.

    Diagnostic only: nothing here reaches a label file.
    """
    all_episode_modal = Counter(component.truth.values()).most_common(1)[0][1]
    return all_episode_modal - len(component.seed)


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--gt-dir", type=Path, required=True, help="ground_truth/ directory")
    parser.add_argument("--splits-dir", type=Path, required=True, help="splits/ directory")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--datasets", nargs="*", default=None, help="component names; default is every ground truth file"
    )
    parser.add_argument("--sources", nargs="*", default=list(SOURCES), choices=list(SOURCES))
    parser.add_argument(
        "--cover",
        choices=("annotated", "eval"),
        default="annotated",
        help="which episodes get an entry. 'annotated' (default) also covers the seed "
             "episodes, so the same file serves the calibration fit; 'eval' is the minimum "
             "the scored run needs. Eval coverage is asserted either way.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="do not write; re-read existing files through the tool's own loader and "
             "assert every eval episode resolves to its own non-empty label list",
    )
    parser.add_argument("--summary", type=Path, default=None, help="corpus summary JSON (default: <out-dir>/labels_summary.json)")
    args = parser.parse_args()

    gt_paths = sorted(args.gt_dir.glob("*.json"))
    if args.datasets:
        wanted = set(args.datasets)
        gt_paths = [p for p in gt_paths if p.stem in wanted]
        missing = wanted - {p.stem for p in gt_paths}
        if missing:
            raise SystemExit(f"no ground truth for {sorted(missing)}")
    if not gt_paths:
        raise SystemExit(f"no ground truth files under {args.gt_dir}")

    problems: list[str] = []
    manifests: list[dict[str, Any]] = []
    totals = Counter()
    for gt_path in gt_paths:
        split_path = args.splits_dir / gt_path.name
        if not split_path.exists():
            raise SystemExit(f"missing split for {gt_path.stem}: {split_path}")
        component = read_component(gt_path, split_path)
        manifest = process(
            component, args.out_dir, args.sources, args.cover, not args.verify, problems
        )
        manifest["prior_sweep_held_out"] = prior_sweep_population(component)
        manifests.append(manifest)

        totals["components"] += 1
        totals["covered"] += manifest["coverage"]["n_covered"]
        totals["eval"] += manifest["coverage"]["n_eval"]
        totals["modal_eval_matches"] += manifest["modal"]["eval_episodes_matching"]
        totals["prior_sweep_held_out"] += manifest["prior_sweep_held_out"]
        totals["ties"] += int(manifest["modal"]["tie_broken"])

        if not args.verify:
            manifest_path = args.out_dir / f"{component.dataset}__labels_manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        verb = "verified" if args.verify else "wrote"
        print(
            f"[labels] {component.dataset}: {verb} {len(args.sources)} file(s), "
            f"{manifest['coverage']['n_covered']} episodes covered "
            f"({manifest['coverage']['n_eval']} eval); modal |L|="
            f"{manifest['modal']['n_labels']} matches "
            f"{manifest['modal']['eval_episodes_matching']}/{manifest['coverage']['n_eval']} eval"
            + ("  [TIE BROKEN]" if manifest["modal"]["tie_broken"] else "")
        )

    summary = {
        "mode": "verify" if args.verify else "write",
        "cover": args.cover,
        "sources": args.sources,
        "components": totals["components"],
        "episodes_covered": totals["covered"],
        "eval_episodes": totals["eval"],
        "modal_eval_matches": totals["modal_eval_matches"],
        "modal_eval_coverage_loss": totals["eval"] - totals["modal_eval_matches"],
        "components_with_tie": totals["ties"],
        # Both populations, per §2.1b. They differ, and the difference is a
        # property of the plan's two definitions, not of this script.
        "prior_sweep_held_out_reconstruction": totals["prior_sweep_held_out"],
        "manifests": manifests,
    }
    summary_path = args.summary or (args.out_dir / "labels_summary.json")
    if not args.verify:
        summary_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")

    print("=" * 72)
    print(
        f"[labels] {totals['components']} components, {totals['covered']} episodes covered, "
        f"{totals['eval']} eval episodes"
    )
    print(
        f"[labels] modal list matches {totals['modal_eval_matches']}/{totals['eval']} eval "
        f"episodes ({totals['eval'] - totals['modal_eval_matches']} coverage loss); "
        f"prior-sweep reconstruction {totals['prior_sweep_held_out']}"
    )
    if problems:
        # One bad file yields one problem per episode it covers, so an
        # uncapped list buries the distinct failures under fifty copies of one.
        for problem in problems[:MAX_REPORTED_PROBLEMS]:
            print(f"  FAIL {problem}")
        if len(problems) > MAX_REPORTED_PROBLEMS:
            print(f"  ... and {len(problems) - MAX_REPORTED_PROBLEMS} more")
        print(f"\n{len(problems)} label-file check(s) failed.")
        return 1
    print("[labels] every eval episode resolves to its own non-empty label list.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
