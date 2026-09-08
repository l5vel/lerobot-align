#!/usr/bin/env python
"""Fingerprint of everything a C1 verdict depends on.

A C1 PASS is a statement about a specific pair of tools, a specific set of arm
flags, a specific model and a specific corpus. Change any of those and the
verdict stops being about the run you are now attempting -- but the JSON file
still says "PASS", and a gate that only reads the verdict will wave it through.

That is not a hypothetical: editing one flag in ``arms.yaml``, checking out a
different commit of either tool, or pointing at a different model would all
leave a stale PASS on disk that authorises a sweep it never covered.

So the gate records this fingerprint when it certifies, and re-computes it
before every sweep. If they differ, the verdict is stale and the sweep is
refused with a diff of what moved.

Deliberately included:
  * the canonical definitions of the certified baseline and arm -- every flag
    of the two arms the verdict is actually about. The whole arms file is
    recorded alongside but not gated on, so that adding an unrelated arm
    elsewhere in the file does not falsely invalidate a live verdict
  * the annotation source of BOTH tools, hashed from disk rather than from git,
    so an uncommitted edit invalidates the verdict too
  * the model id, which decides what was actually measured
  * the matcher and threshold, which decide how it was scored
  * the arm and baseline names the verdict was about

Deliberately excluded: the episode split and the scores file. E1 runs on a
subset of components by design, so requiring the split to match would make the
gate unusable for the sweep it exists to authorise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL_DIR = HERE.parent
REPO_DIR = EVAL_DIR.parent


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else "absent"


def _hash_tree(root: Path, suffixes: tuple[str, ...] = (".py", ".txt")) -> str:
    """Content hash of a source tree, ignoring caches and file order.

    Hashed from disk, not from a git revision: a verdict certified against an
    uncommitted local edit must not survive that edit being changed again.
    """
    digest = hashlib.sha256()
    if not root.exists():
        return "absent"
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        if "__pycache__" in path.parts:
            continue
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


# Entry points whose import graphs define "the code under test". Everything
# either tool loads to annotate is reachable from one of these.
ENTRY_POINTS = ("lerobot.scripts.lerobot_annotate", "lerobot_align.cli")

# The provenance a prediction must carry to be certifiable, declared once so the
# producer (run_arm), the carrier (score_runs) and the verifier (c1_gate) cannot
# drift apart. They already had: `effective_prompts` was recorded by the
# producer, silently dropped by the carrier, and skipped by the verifier because
# an absent key left an empty observed set that the mismatch branch never
# examined -- so prompt-override provenance was collected and then discarded.
PROVENANCE_KEYS = (
    "align_source",
    "upstream_source",
    "effective_prompts",
    "prompt_overrides_active",
)

# Modules that MUST appear in the discovered graph. This is a sanity assertion,
# not the definition: if discovery silently returned a thin set, a fingerprint
# computed from it would look valid while covering almost nothing.
REQUIRED_IN_GRAPH = (
    "scripts/lerobot_annotate.py",       # the baseline's entry point
    "annotations/steerable_pipeline/",   # the annotation pipeline itself
    "datasets/",                         # dataset IO, executed by both tools
)

# Non-code files that are read at runtime and shape the output. The import
# graph cannot see these: a prompt is loaded as data, so no module entry ever
# appears for it, and hashing only modules left every prompt unfingerprinted --
# including `plan_subtasks.txt`, the prompt that produces the segmentation
# being measured. Editing it would have changed every prediction while leaving
# the cache key and the C1 verdict untouched.
RESOURCE_SUFFIXES = (".txt", ".json", ".yaml", ".yml", ".jinja", ".j2", ".md")

# Resources that MUST be found, for the same reason REQUIRED_IN_GRAPH exists.
REQUIRED_RESOURCES = (
    "annotations/steerable_pipeline/prompts/plan_subtasks.txt",
    "annotations/steerable_pipeline/prompts/plan_subtask_describe.txt",
)


_DISCOVERY_CACHE: list[Path] | None = None


def _discover_executed_modules() -> list[Path]:
    """Every file under the installed ``lerobot`` package that the tools import.

    Enumerating directories by hand does not work, and failing at it twice is
    what motivated this: the first attempt hashed only
    ``annotations/steerable_pipeline`` and missed the baseline's own entry
    point; the second added five directories by inspection and still missed 39
    imported modules, among them ``lerobot/__init__.py``, ``lerobot_types.py``
    and the whole ``processor`` and ``transforms`` packages.

    The import graph is the authoritative answer, and asking Python for it is
    both exact and self-maintaining: a dependency that appears in a future
    version is picked up without anyone remembering to add it.

    Discovery runs in a subprocess so that importing the entry points cannot
    perturb the calling interpreter, and so that an import failure is a clean
    error rather than a half-initialised module table.
    """
    global _DISCOVERY_CACHE
    if _DISCOVERY_CACHE is not None:
        return _DISCOVERY_CACHE

    import subprocess

    program = (
        "import importlib, json, sys\n"
        f"for name in {list(ENTRY_POINTS)!r}:\n"
        "    importlib.import_module(name)\n"
        "import lerobot\n"
        "from pathlib import Path\n"
        "root = Path(lerobot.__file__).parent\n"
        "out = set()\n"
        "for name, module in sys.modules.items():\n"
        "    if name != 'lerobot' and not name.startswith('lerobot.'):\n"
        "        continue\n"
        "    path = getattr(module, '__file__', None)\n"
        "    if not path:\n"
        "        continue\n"
        "    resolved = Path(path).resolve()\n"
        "    if resolved.is_relative_to(root):\n"
        "        out.add(resolved.relative_to(root).as_posix())\n"
        "print(json.dumps(sorted(out)))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(
            "cannot enumerate the upstream import graph; refusing to fingerprint a "
            f"guessed subset.\n{result.stderr.strip()[-2000:]}"
        )
    import lerobot

    root = Path(lerobot.__file__).parent
    relatives = json.loads(result.stdout)
    missing = [
        required for required in REQUIRED_IN_GRAPH
        if not any(rel == required or rel.startswith(required) for rel in relatives)
    ]
    if missing:
        raise RuntimeError(
            f"upstream import graph is missing {missing}; discovery returned "
            f"{len(relatives)} module(s) and cannot be trusted."
        )
    _DISCOVERY_CACHE = [root / rel for rel in relatives]
    return _DISCOVERY_CACHE


def effective_prompt_digest() -> tuple[str, list[str]]:
    """Hash the prompts as the tools will ACTUALLY see them.

    Both packages load prompts through a helper that checks
    ``LEROBOT_PROMPT_OVERRIDE_<name>`` in the environment first and only falls
    back to the packaged ``.txt``. The override is a supported feature (it lets
    prompt search inject candidates without rebuilding), and it makes a file
    hash insufficient on its own: with an override set, the file that was
    hashed is not the text that was sent.

    Two failure modes this closes. A verdict certified with an override active
    would be reused for a run without it, and vice versa -- same fingerprint,
    different prompts, different predictions. And an override applied while one
    arm ran but not another would break the byte-identical-prompt equalisation
    that the whole fairness argument rests on, invisibly.

    Returns the digest and the names of any prompts currently overridden, so
    callers can report them rather than merely account for them.
    """
    import os

    import lerobot

    upstream_dir = (
        Path(lerobot.__file__).parent / "annotations" / "steerable_pipeline" / "prompts"
    )
    import lerobot_align

    align_dir = Path(lerobot_align.__file__).resolve().parent / "prompts"
    names = sorted(
        {p.stem for d in (upstream_dir, align_dir) if d.is_dir() for p in d.glob("*.txt")}
    )
    digest = hashlib.sha256()
    overridden: list[str] = []
    for name in names:
        override = os.environ.get(f"LEROBOT_PROMPT_OVERRIDE_{name}")
        digest.update(name.encode("utf-8"))
        if override and override.strip():
            overridden.append(name)
            digest.update(b"override:")
            digest.update(hashlib.sha256(override.encode("utf-8")).hexdigest().encode("utf-8"))
        else:
            digest.update(b"packaged:")
            for directory in (upstream_dir, align_dir):
                candidate = directory / f"{name}.txt"
                digest.update(_hash_file(candidate).encode("utf-8"))
    return digest.hexdigest()[:16], overridden


def _discover_runtime_resources() -> list[Path]:
    """Non-code files read at runtime by the packages the tools import.

    Derived the same way as the module set rather than hand-listed: every
    package directory that contains an imported module is scanned for resource
    files. That keeps the rule "fingerprint what the tools actually use"
    without needing anyone to remember that prompts live in a particular
    subdirectory.
    """
    import lerobot

    root = Path(lerobot.__file__).parent
    directories = {path.parent for path in _discover_executed_modules()}
    found: set[Path] = set()
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if (
                path.is_file()
                and path.suffix in RESOURCE_SUFFIXES
                and "__pycache__" not in path.parts
            ):
                found.add(path)
    relatives = {p.relative_to(root).as_posix() for p in found}
    missing = [r for r in REQUIRED_RESOURCES if r not in relatives]
    if missing:
        raise RuntimeError(
            f"runtime resource discovery missed {missing}; refusing to fingerprint a "
            "subset that excludes the prompts the tools actually send."
        )
    return sorted(found)


def source_hashes() -> dict[str, str]:
    """Content hashes of everything either tool executes to annotate.

    Used both to fingerprint a C1 verdict and to key the prediction cache, so
    the two cannot disagree: if the code changes, cached predictions are
    invalidated AND any verdict resting on them goes stale.

    ``align_source`` hashes the whole package tree including its prompts (a
    superset of what is imported, which is the conservative direction).
    ``upstream_source`` hashes the modules the tools load, discovered from the
    import graph, **plus** the runtime resources in those packages -- the
    prompts above all, which no import graph can reveal because they are read
    as data.
    """
    import lerobot

    root = Path(lerobot.__file__).parent
    digest = hashlib.sha256()
    modules = _discover_executed_modules()
    resources = _discover_runtime_resources()
    for path in sorted(modules) + sorted(resources):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(_hash_file(path).encode("utf-8"))
    prompt_digest, overridden = effective_prompt_digest()
    return {
        "align_source": _hash_tree(
            Path(__import__("lerobot_align").__file__).resolve().parent, (".py", *RESOURCE_SUFFIXES)
        ),
        "upstream_source": digest.hexdigest()[:16],
        "upstream_module_count": str(len(modules)),
        "upstream_resource_count": str(len(resources)),
        # The prompts as they will actually be sent, override-aware. A packaged
        # file hash is not sufficient on its own; see effective_prompt_digest.
        "effective_prompts": prompt_digest,
        "prompt_overrides_active": ",".join(overridden) if overridden else "none",
    }


# Keys excluded from staleness comparison: recorded for the audit trail, but a
# change to them does not mean the verdict describes a different configuration.
INFORMATIONAL_KEYS = frozenset({"arms_yaml_file"})


def _hash_arms(*names: str) -> str:
    """Hash the canonical definitions of the named arms, in the order given.

    Raises if an arm is missing: a verdict certified for an arm that no longer
    exists is not merely stale, it is unreadable, and silently hashing an empty
    definition would make it look current.
    """
    import yaml

    spec = yaml.safe_load((EVAL_DIR / "configs" / "arms.yaml").read_text(encoding="utf-8"))
    by_name = {a["name"]: a for a in spec.get("arms", [])}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise SystemExit(f"arms.yaml no longer defines: {', '.join(missing)}")
    payload = json.dumps([by_name[n] for n in names], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def compute(
    *, model_id: str, matcher: str, threshold: float | None,
    baseline: str, arm: str, metrics: list[str],
) -> dict[str, str]:
    import lerobot

    return {
        # The definitions of the two arms this verdict is ABOUT, canonicalised.
        # This used to hash the whole arms.yaml, which invalidated a live verdict
        # whenever any unrelated arm was added -- adding a secondary experiment
        # elsewhere in the file does not change what the certified pair does, so
        # that was a false staleness, and the only way past it was to spend GPU
        # re-certifying an unchanged configuration.
        #
        # The narrowing is deliberate and strictly toward precision: editing any
        # flag of either certified arm, renaming one, or deleting one still
        # invalidates. What no longer invalidates is a change to some arm the
        # verdict never covered -- which it never covered anyway (see
        # analysis/deviations.md D48 on the gate's cell scope).
        "arms_certified": _hash_arms(baseline, arm),
        # Informational: recorded so a file change is visible in the audit trail,
        # but NOT gated on, for the reason above. `diff` skips it.
        "arms_yaml_file": _hash_file(EVAL_DIR / "configs" / "arms.yaml"),
        **source_hashes(),
        "lerobot_version": getattr(lerobot, "__version__", "unknown"),
        "model_id": model_id,
        "matcher": matcher,
        "threshold": f"{threshold:.4f}" if threshold is not None else "none",
        "baseline": baseline,
        "arm": arm,
        "metrics": ",".join(sorted(metrics)),
    }


def diff(recorded: dict[str, str], current: dict[str, str]) -> list[str]:
    recorded = dict(recorded)
    if "arms_yaml" in recorded and "arms_certified" not in recorded:
        legacy = recorded["arms_yaml"]
        if legacy == current.get("arms_yaml_file"):
            certified = current.get("arms_certified")
        else:
            certified = legacy_certified_hash(
                legacy, recorded.get("baseline", ""), recorded.get("arm", ""))
        if certified is not None:
            recorded.pop("arms_yaml")
            recorded["arms_certified"] = certified
    out = []
    for key in sorted(set(recorded) | set(current)):
        if key in INFORMATIONAL_KEYS:
            continue
        was, now = recorded.get(key, "<missing>"), current.get(key, "<missing>")
        if was != now:
            out.append(f"{key}: certified with {was!r}, now {now!r}")
    return out


def legacy_certified_hash(file_hash: str, baseline: str, arm: str) -> str | None:
    """Resolve an old whole-file fingerprint through verifiable git history.

    Never alias the two hash formats or ignore a missing certification. Only a
    historical file with the recorded content hash can establish what was run.
    Unknown snapshots remain stale, and edits to either certified arm still fail.
    """
    import subprocess
    import yaml

    path = "evaluation/configs/arms.yaml"
    history = subprocess.run(
        ["git", "-C", str(REPO_DIR), "log", "--format=%H", "--", path],
        capture_output=True, text=True, check=False,
    )
    if history.returncode:
        return None
    for commit in history.stdout.splitlines():
        snapshot = subprocess.run(
            ["git", "-C", str(REPO_DIR), "show", f"{commit}:{path}"],
            capture_output=True, check=False,
        )
        if snapshot.returncode or hashlib.sha256(snapshot.stdout).hexdigest()[:16] != file_hash:
            continue
        spec = yaml.safe_load(snapshot.stdout)
        by_name = {a["name"]: a for a in spec["arms"]}
        if baseline not in by_name or arm not in by_name:
            return None
        payload = json.dumps([by_name[baseline], by_name[arm]], sort_keys=True,
                             separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--matcher", default="embedding")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--baseline", default="baseline_upstream")
    parser.add_argument("--arm", default="align_defaults")
    parser.add_argument("--metrics", nargs="*",
                        default=["boundary_f1@0p5", "boundary_f1@1", "macro_iou"])
    parser.add_argument("--check", type=Path, default=None,
                        help="A c1_gate.json to validate against the current state.")
    args = parser.parse_args()

    current = compute(
        model_id=args.model_id, matcher=args.matcher, threshold=args.threshold,
        baseline=args.baseline, arm=args.arm, metrics=args.metrics,
    )
    if args.check is None:
        print(json.dumps(current, indent=1))
        return 0

    if not args.check.exists():
        print(f"STALE: {args.check} does not exist")
        return 1
    payload = json.loads(args.check.read_text(encoding="utf-8"))
    recorded = payload.get("environment_fingerprint")
    if not recorded:
        print("STALE: verdict carries no environment fingerprint; it predates this check "
              "and cannot be shown to describe the current configuration")
        return 1
    problems = diff(recorded, current)
    if problems:
        print("STALE: the C1 verdict was certified against a different configuration:")
        for line in problems:
            print(f"  - {line}")
        return 1
    print("FRESH: C1 verdict matches the current configuration")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
