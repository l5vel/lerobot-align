#!/usr/bin/env python
"""LLM-judge matcher backend, plus the cache pre-warm that makes it affordable.

Section 6.2 of the pre-registration promises every label-aware number under
three matchers -- ``exact``, ``embedding`` and ``judge`` -- on the grounds that
a finding surviving only one of them is a finding about the matcher rather than
about the tools. Only the first two existed, so that promise was
unachievable (recorded as an open item in ``analysis/deviations.md``). This
module supplies the third.

It provides exactly one thing to the metric layer: a callable
``(left, right, context) -> bool`` that ``metrics.semantic.JudgeMatcher``
invokes on a cache miss. Disk caching, blinding-by-sorting, the cache key and
the contrast guard all live in ``JudgeMatcher`` and are deliberately not
reimplemented here -- two implementations of a cache key is how a study ends up
with two different answers to the same question.

Why the prompt must be symmetric
--------------------------------
``JudgeMatcher`` hands ``ask`` the two labels already normalised and sorted, so
which label appears first is a function of the *text* and never of the arm that
produced it. That property is only worth having if the prompt preserves it.
An asymmetric prompt -- "reference:" / "candidate:", "expected:" / "predicted:",
or any wording that invites the judge to treat one side as authoritative --
would convert the judge's position and role bias into a systematic effect on
one tool. It would not even be a constant bias: the human label sorts first for
some pairs and second for others, so the bias would land unevenly across
components and appear as a real, clustered difference between arms. The two
phrases are therefore presented as an unordered pair of independent
descriptions, with no claim that either is correct, and the system prompt says
so explicitly.

For the same reason the task context describes the *task* and never the label
inventory. Listing the human vocabulary would let the judge identify which
phrase is the human one by lookup, which is exactly the blinding the sorted
pair exists to protect.

Why an unparseable reply is a NON-match, and is counted
------------------------------------------------------
The tempting failure handling -- retry until something parses, or map anything
non-negative to "match" -- silently inflates every label-aware metric for every
arm, and does so unevenly: the arm producing vaguer labels gains most, which is
the precise direction of error this evaluation exists to exclude. An
unparseable reply is therefore False, and the count and a sample of the raw
replies are reported, so a mis-templated model shows up as a number rather than
as a suspiciously generous score.

That leniency applies only to *scoring*. During the pre-warm the client runs
``strict=True``: an unparseable reply raises instead of resolving, so it is
never written to the cache. A False written to an append-only cache is
permanent and invisible, and the pre-warm is the one moment at which a broken
serving configuration can still be fixed cheaply.

A transport failure is not a verdict either. Timeouts and 5xx are retried and
then raised; they are never turned into False, because a fabricated "these two
labels differ" cached forever is far worse than a job that stops.

Determinism
-----------
``temperature=0``, ``top_p=1``, a fixed ``seed`` and an on-disk cache. The plan
already accepts that neither *tool* is reproducible (section 2.2a.4); a
non-reproducible *ruler* on top of that would make even the same predictions
score differently on re-analysis, and no equivalence margin could absorb it.

Two answers are settled without spending a call, because they cannot depend on
a model's opinion and every backend must agree on them: labels identical after
normalisation match (that is what ``exact`` means), and a pair with an empty
label on one side only does not. Asking a judge whether "pick up the cup"
denotes the same action as "pick up the cup" risks getting "no".

Why pre-warm
------------
``metrics.score`` calls the matcher through the ``Matcher`` protocol, which has
no context parameter, from inside a serial per-episode loop. A cold cache would
therefore issue tens of thousands of sequential requests *and* ask every one of
them without task context. The CLI below enumerates exactly the pairs the
scorer will need -- the union over scored episodes of (predicted labels x
reference labels), which is what the matching DPs actually consult -- asks them
concurrently in batches, and commits each batch through ``JudgeMatcher`` so the
keys are byte-identical to the ones scoring will look up. Scoring then hits the
cache for every pair and the endpoint for none.

The HTTP call is stdlib rather than the ``openai`` SDK so that ``--dry-run``
can print the literal request body that would be sent. The body is the artifact
a reviewer needs to check; a printed SDK call is a description of one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from metrics.semantic import (  # noqa: E402
    JudgeMatcher,
    contrast_conflict,
    judge_cache_key,
    normalise_label,
)

DEFAULT_ENDPOINT = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "Qwen/Qwen3.8-27B"
DEFAULT_TIMEOUT = 120.0

# Kept as module constants so a run can record which wording produced its
# verdicts: the judge prompt is part of the measuring instrument, and the report
# carries its SHA-256 for the same reason every other input to this study does.
SYSTEM_PROMPT = """\
You decide whether two short phrases describe the same physical action performed \
by a robot in a demonstration video.

The two phrases were written independently to describe a step of the same task. \
Neither is authoritative and neither is a correction of the other. Their order \
carries no meaning: it is alphabetical, not a ranking, and tells you nothing \
about where either phrase came from.

Answer YES when the same physical motion on the same object would satisfy both \
phrases -- including when one is worded differently, is less specific, or names \
the object or the robot differently.

Answer NO when they describe different steps of the task: a different object, a \
different destination, an opposite motion (open vs close, pick up vs put down, \
left vs right, push vs pull), or a different stage of the same manipulation.

Reply with exactly one word, YES or NO. Do not explain."""

USER_TEMPLATE = """\
{context_line}Phrase A: {a}
Phrase B: {b}

Do Phrase A and Phrase B describe the same physical robot action? Answer YES or NO."""


class JudgeError(RuntimeError):
    """The judge produced no usable verdict for a pair.

    Raised for transport failures always, and for unparseable replies only in
    strict mode. Never converted into a verdict by this module: the caller
    decides whether to abandon the pair (pre-warm) or to record a counted
    non-match (scoring).
    """


# --------------------------------------------------------------------------
# reply parsing
# --------------------------------------------------------------------------

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_WORD = re.compile(r"[a-z]+")
# Keys a model wrapping its answer in JSON plausibly uses. Only an object with
# recognised keys is accepted; a bare `true` is not a YES, because the prompt
# asked for a word and accepting near-misses is how a parser starts guessing.
_VERDICT_KEYS = ("same_action", "same", "equivalent", "match", "verdict", "answer", "result")


def strip_reasoning(reply: str) -> str:
    """Remove a ``<think>...</think>`` block from a reply.

    Qwen3-family chat templates emit one unless thinking is disabled, and its
    contents necessarily discuss both possibilities -- so leaving it in makes
    every reply ambiguous and every pair a counted non-match. Removing it is
    not a licence to parse the reasoning: an unterminated block means the model
    never reached an answer and the reply stays unparseable.
    """
    text = _THINK_BLOCK.sub(" ", reply)
    if "</think>" in text:  # opening tag suppressed by the template, or nested
        text = text.rsplit("</think>", 1)[1]
    lowered = text.casefold()
    if "<think>" in lowered:
        # An unclosed block: the completion hit its token limit mid-deliberation.
        # Everything from the tag on is reasoning, and reading a verdict out of
        # reasoning is precisely what this function must not do.
        text = text[: lowered.index("<think>")]
    return text.strip()


def _coerce_word(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        word = value.strip().casefold().strip(".!\"' ")
        if word in {"yes", "true"}:
            return True
        if word in {"no", "false"}:
            return False
    return None


def _json_verdict(text: str) -> bool | None:
    candidates = list(_FENCE.findall(text))
    candidates.append(text)
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    for blob in candidates:
        try:
            payload = json.loads(blob)
        except (ValueError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        present = [payload[key] for key in _VERDICT_KEYS if key in payload]
        if not present:
            continue
        coerced = {_coerce_word(value) for value in present}
        if len(coerced) == 1 and None not in coerced:
            return coerced.pop()
    return None


def parse_verdict(reply: str) -> bool | None:
    """Strict YES/NO from a raw completion; ``None`` when there is no answer.

    ``None`` means *unparseable*, and the caller must treat it as a non-match
    rather than as anything else. The rule is deliberately unable to guess: a
    reply mentioning both polarities ("yes and no"), mentioning neither
    ("maybe", a refusal) or empty yields None. Only a JSON object carrying a
    recognised verdict key, or a reply containing exactly one of the words
    "yes" and "no", resolves.
    """
    text = strip_reasoning(reply or "")
    if not text:
        return None
    verdict = _json_verdict(text)
    if verdict is not None:
        return verdict
    words = set(_WORD.findall(text.casefold()))
    has_yes, has_no = "yes" in words, "no" in words
    if has_yes != has_no:
        return has_yes
    return None


# --------------------------------------------------------------------------
# the client
# --------------------------------------------------------------------------

@dataclass
class JudgeStats:
    """Call accounting, shared across worker threads."""

    calls: int = 0
    yes: int = 0
    no: int = 0
    unparseable: int = 0
    retries: int = 0
    unparseable_samples: list[dict[str, Any]] = field(default_factory=list)
    max_samples: int = 20
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, verdict: bool | None, sample: dict[str, Any]) -> None:
        with self._lock:
            self.calls += 1
            if verdict is None:
                self.unparseable += 1
                if len(self.unparseable_samples) < self.max_samples:
                    self.unparseable_samples.append(sample)
            elif verdict:
                self.yes += 1
            else:
                self.no += 1

    def record_retry(self) -> None:
        with self._lock:
            self.retries += 1

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "calls": self.calls,
                "yes": self.yes,
                "no": self.no,
                "unparseable": self.unparseable,
                "unparseable_rate": (self.unparseable / self.calls) if self.calls else 0.0,
                "retries": self.retries,
                "unparseable_samples": list(self.unparseable_samples),
            }


@dataclass
class JudgeClient:
    """An ``ask`` callable for ``metrics.semantic.JudgeMatcher``.

    Instances are callable as ``client(left, right, context) -> bool`` and are
    thread-safe: no per-call state is kept outside ``stats``, which locks.

    ``strict`` decides what an unparseable reply means. False (the scoring
    default) returns a counted non-match, so a single odd reply cannot abort a
    scoring run. True (the pre-warm default) raises, so the pair is left out of
    the cache and can be re-asked after the serving configuration is fixed --
    once a False is in the append-only cache it is permanent.
    """

    endpoint: str = DEFAULT_ENDPOINT
    model: str = DEFAULT_MODEL
    timeout: float = DEFAULT_TIMEOUT
    api_key: str = "EMPTY"
    max_tokens: int = 256
    seed: int = 0
    max_retries: int = 3
    retry_backoff: float = 2.0
    default_context: str = ""
    # vLLM honours this for Qwen3-family templates; a template that does not
    # take the key ignores it. Set to {} for an endpoint that rejects unknown
    # request fields. Thinking is off because the judge is a one-word
    # classification and a truncated reasoning block answers nothing.
    chat_template_kwargs: dict[str, Any] | None = field(
        default_factory=lambda: {"enable_thinking": False}
    )
    strict: bool = False
    stats: JudgeStats = field(default_factory=JudgeStats)

    def messages(self, left: str, right: str, context: str = "") -> list[dict[str, str]]:
        """Render the blinded prompt for one pair.

        ``left`` and ``right`` are used exactly as given: ``JudgeMatcher`` has
        already normalised and sorted them, and re-ordering or re-labelling
        them here would undo the blinding it provides.
        """
        text = context or self.default_context
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    context_line=f"Task context: {text}\n\n" if text else "",
                    a=left,
                    b=right,
                ),
            },
        ]

    def request_body(self, left: str, right: str, context: str = "") -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": self.messages(left, right, context),
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": self.max_tokens,
            "n": 1,
            "seed": self.seed,
            "stream": False,
        }
        if self.chat_template_kwargs:
            body["chat_template_kwargs"] = dict(self.chat_template_kwargs)
        return body

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        url = self.endpoint.rstrip("/") + "/chat/completions"
        payload = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(  # noqa: S310 - operator-supplied local endpoint
            url,
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        last: str = "no attempt made"
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:400]
                # A 4xx is a malformed request -- an unknown field, a wrong
                # model id -- and repeating it changes nothing except the log.
                if exc.code < 500 and exc.code != 429:
                    raise JudgeError(f"HTTP {exc.code} from {url}: {detail}") from exc
                last = f"HTTP {exc.code}: {detail}"
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                last = f"{type(exc).__name__}: {exc}"
            if attempt + 1 < self.max_retries:
                self.stats.record_retry()
                time.sleep(self.retry_backoff * (attempt + 1))
        raise JudgeError(f"{self.max_retries} attempts failed against {url}: {last}")

    def __call__(self, left: str, right: str, context: str = "") -> bool:
        body = self.request_body(left, right, context)
        response = self._post(body)
        try:
            choice = response["choices"][0]
            message = choice.get("message") or {}
        except (KeyError, IndexError, TypeError) as exc:
            raise JudgeError(f"malformed response: {json.dumps(response)[:400]}") from exc
        content = message.get("content") or ""
        verdict = parse_verdict(content)
        self.stats.record(
            verdict,
            {
                "a": left,
                "b": right,
                "reply": content[:400],
                "finish_reason": choice.get("finish_reason"),
                # Present when the server runs a reasoning parser, in which case
                # the answer never reached `content` and thinking must be turned
                # off rather than the parser loosened.
                "had_reasoning_field": bool(message.get("reasoning_content")),
            },
        )
        if verdict is None:
            if self.strict:
                raise JudgeError(f"unparseable reply {content[:200]!r} for {left!r} ~ {right!r}")
            return False
        return verdict


# --------------------------------------------------------------------------
# pair enumeration
# --------------------------------------------------------------------------

# Camera/version suffixes and component numbers carry no task information, so
# they are dropped from the context. Nothing else is: the context must describe
# the task and must never enumerate the labels (see the module docstring).
_NAME_NOISE = {"fv", "bc", "v30", "v21", "v20"}


def task_context(dataset: str) -> str:
    """Mechanical task description derived from the dataset name.

    ``base4-clean-table-01-BC-FV-v30`` -> ``base4 clean table``. Mechanical so
    that no per-dataset wording choice can favour an arm, and overridable with
    ``--context-map`` when a corpus needs a real description.
    """
    tokens = [
        token
        for token in dataset.split("-")
        if token and not token.isdigit() and token.casefold() not in _NAME_NOISE
    ]
    return " ".join(tokens) if tokens else dataset


def render_context(names: Iterable[str]) -> str:
    """Compose the context line for a pair that occurs under several tasks.

    ``JudgeMatcher``'s cache key covers the two labels and the model but not
    the context, so a pair recurring across components is asked once and
    whichever context arrived first would win. Making the context a
    deterministic function of the whole set removes that ordering dependence:
    re-running the pre-warm on the same corpus asks the identical question.
    """
    unique = sorted({name for name in names if name})
    if not unique:
        return ""
    if len(unique) == 1:
        return f'a robot performing the task "{unique[0]}"'
    listed = "; ".join(f'"{name}"' for name in unique)
    return f"a robot performing one of these tasks: {listed}"


@dataclass(frozen=True)
class PairRequest:
    """One question for the judge, keyed exactly as ``JudgeMatcher`` will."""

    a: str
    b: str
    key: str
    context: str


def _labels(spans: Sequence[dict[str, Any]]) -> set[str]:
    return {normalise_label(str(span.get("text", ""))) for span in spans}


def enumerate_pairs(
    *,
    gt_dir: Path,
    predictions_dir: Path,
    splits_dir: Path | None,
    model: str,
    context_overrides: dict[str, str] | None = None,
) -> list[PairRequest]:
    """The exact pair set scoring will consult, and nothing beyond it.

    Every matcher call in ``metrics.joint`` and ``metrics.semantic`` compares a
    *predicted* label against a *reference* label from the SAME episode, so the
    set is the union over scored episodes of (predicted x reference). Warming
    the dataset-level cross product instead would ask for pairs no metric ever
    forms -- on this corpus roughly an order of magnitude more calls, all of
    them wasted.
    """
    overrides = context_overrides or {}
    truth: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for path in sorted(gt_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        truth[payload["dataset"]] = {
            int(key): value for key, value in (payload.get("episodes") or {}).items()
        }
    if not truth:
        raise SystemExit(f"no ground truth under {gt_dir}")

    splits: dict[str, set[int]] = {}
    if splits_dir is not None:
        for path in sorted(splits_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            splits[payload["dataset"]] = {int(e) for e in payload["eval"]}

    contexts: dict[tuple[str, str], set[str]] = {}
    for path in sorted(predictions_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        dataset = payload.get("dataset")
        if dataset not in truth:
            print(f"[judge] SKIP {path.name}: no ground truth for {dataset}", file=sys.stderr)
            continue
        context = overrides.get(dataset, task_context(dataset))
        reference_all = truth[dataset]
        scored = splits.get(dataset, set(reference_all))
        predictions = {int(k): v for k, v in (payload.get("episodes") or {}).items()}
        for episode in sorted(scored):
            reference = reference_all.get(episode)
            predicted = predictions.get(episode)
            if not reference or not predicted:
                continue
            for left in _labels(predicted):
                for right in _labels(reference):
                    contexts.setdefault(tuple(sorted((left, right))), set()).add(context)

    pairs = [
        PairRequest(a=a, b=b, key=judge_cache_key(a, b, model), context=render_context(names))
        for (a, b), names in contexts.items()
    ]
    pairs.sort(key=lambda pair: (pair.a, pair.b))
    return pairs


def pairs_from_json(path: Path, model: str) -> list[PairRequest]:
    """Explicit pairs: ``[[a, b], ...]`` or ``[[a, b, context], ...]``.

    Used to warm hand-labelled (model, human) pairs from the pilot, the same
    material ``calibrate_matcher.py --extra-pairs`` consumes, so the judge and
    the embedding threshold can be validated against one another on identical
    inputs.
    """
    rows = json.loads(path.read_text(encoding="utf-8"))
    contexts: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        left, right = normalise_label(str(row[0])), normalise_label(str(row[1]))
        context = str(row[2]) if len(row) > 2 else ""
        contexts.setdefault(tuple(sorted((left, right))), set()).add(context)
    pairs = [
        PairRequest(a=a, b=b, key=judge_cache_key(a, b, model), context="; ".join(sorted(n - {""})))
        for (a, b), n in contexts.items()
    ]
    pairs.sort(key=lambda pair: (pair.a, pair.b))
    return pairs


def cached_keys(cache_path: Path) -> set[str]:
    """Keys already on disk, so the pre-warm can size its work before asking.

    ``JudgeMatcher`` loads the same file, but only to answer lookups one at a
    time; the batch planner needs the miss set up front. The key derivation
    itself is imported, never re-derived -- a second implementation of the key
    would eventually disagree with the first.
    """
    if not cache_path.exists():
        return set()
    keys: set[str] = set()
    for number, line in enumerate(cache_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            keys.add(json.loads(line)["key"])
        except (ValueError, KeyError) as exc:
            raise SystemExit(f"{cache_path}:{number}: unreadable cache line ({exc})") from exc
    return keys


def chunked(items: Sequence[PairRequest], size: int) -> Iterator[list[PairRequest]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


# --------------------------------------------------------------------------
# pre-warm
# --------------------------------------------------------------------------

def judge_batch(
    client: JudgeClient, batch: Sequence[PairRequest], workers: int
) -> tuple[dict[tuple[str, str], bool], list[tuple[PairRequest, str]]]:
    """Ask one batch concurrently; return resolved verdicts and failures apart.

    Concurrency is per batch rather than over the whole corpus so that an
    interrupt loses at most one batch: each batch is committed to the cache
    before the next is asked, and the cache is append-only, so a resumed
    pre-warm re-asks only what it never got.
    """
    resolved: dict[tuple[str, str], bool] = {}
    failures: list[tuple[PairRequest, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(client, pair.a, pair.b, pair.context): pair for pair in batch}
        for future in as_completed(futures):
            pair = futures[future]
            try:
                resolved[(pair.a, pair.b)] = future.result()
            except JudgeError as exc:
                failures.append((pair, str(exc)))
    return resolved, failures


def commit(
    matcher: JudgeMatcher,
    batch: Sequence[PairRequest],
    resolved: dict[tuple[str, str], bool],
) -> int:
    """Write resolved verdicts through ``JudgeMatcher`` so the keys match.

    The verdicts are already known; ``JudgeMatcher`` is driven with a lookup
    that cannot call anything, purely so that the cache file is written by the
    same code that will later read it. Anything not in ``resolved`` -- a
    transport failure, an unparseable reply under ``strict`` -- is skipped and
    stays absent from the cache rather than being recorded as a non-match.
    """

    def ask(left: str, right: str, context: str) -> bool:  # noqa: ARG001 - protocol shape
        try:
            return resolved[(left, right)]
        except KeyError:  # pragma: no cover - guarded by the caller's filter
            raise JudgeError(f"no verdict resolved for {left!r} ~ {right!r}") from None

    previous, matcher.ask = matcher.ask, ask
    try:
        written = 0
        for pair in batch:
            if (pair.a, pair.b) not in resolved:
                continue
            matcher.equivalent(pair.a, pair.b, pair.context)
            written += 1
        return written
    finally:
        matcher.ask = previous


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, required=True, help="JudgeMatcher cache JSONL")
    parser.add_argument("--gt-dir", type=Path, default=None)
    parser.add_argument("--predictions-dir", type=Path, default=None)
    parser.add_argument(
        "--splits-dir", type=Path, default=None,
        help="restrict to the evaluation episodes, i.e. exactly what score_runs.py scores",
    )
    parser.add_argument("--pairs-json", type=Path, default=None, help="explicit [[a, b], ...] pairs")
    parser.add_argument("--context-map", type=Path, default=None, help="JSON {dataset: task text}")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="served model id AND the cache-key model; score_runs.py must pass the same string")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8, help="concurrent requests per batch")
    parser.add_argument("--batch-size", type=int, default=256, help="pairs committed to the cache at once")
    parser.add_argument("--limit", type=int, default=None, help="ask at most N pairs this invocation")
    parser.add_argument("--chat-template-kwargs", default='{"enable_thinking": false}',
                        help="JSON forwarded to vLLM; pass {} for an endpoint that rejects it")
    parser.add_argument("--allow-unparseable", action="store_true",
                        help="exit 0 even if some pairs produced no usable verdict (they stay uncached)")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the exact request that would be sent for every pair and exit")
    args = parser.parse_args()

    overrides = (
        json.loads(args.context_map.read_text(encoding="utf-8")) if args.context_map else {}
    )
    if args.pairs_json:
        pairs = pairs_from_json(args.pairs_json, args.model)
    elif args.gt_dir and args.predictions_dir:
        pairs = enumerate_pairs(
            gt_dir=args.gt_dir,
            predictions_dir=args.predictions_dir,
            splits_dir=args.splits_dir,
            model=args.model,
            context_overrides=overrides,
        )
    else:
        raise SystemExit("need --pairs-json, or both --gt-dir and --predictions-dir")
    if not pairs:
        raise SystemExit(
            "no label pairs found: run the arms first, then pre-warm against their predictions"
        )

    known = cached_keys(args.cache)
    # Mirrors JudgeMatcher's own precedence exactly. A guarded pair is answered
    # before the cache is consulted, so asking about it would spend tokens on a
    # verdict that is discarded, and caching one would create an entry nothing
    # ever reads.
    guarded = [p for p in pairs if contrast_conflict(p.a, p.b)]
    remaining = [p for p in pairs if not contrast_conflict(p.a, p.b)]
    cached = [p for p in remaining if p.key in known]
    remaining = [p for p in remaining if p.key not in known]
    # Settled without a call: see the module docstring.
    free = {(p.a, p.b): (p.a == p.b) for p in remaining if p.a == p.b or not (p.a and p.b)}
    to_ask = [p for p in remaining if (p.a, p.b) not in free]
    free_pairs = [p for p in remaining if (p.a, p.b) in free]
    if args.limit is not None:
        to_ask = to_ask[: args.limit]

    print(
        f"[judge] {len(pairs)} distinct pairs: {len(cached)} cached, {len(guarded)} blocked by the "
        f"contrast guard, {len(free_pairs)} settled without a call, {len(to_ask)} to ask"
    )

    client = JudgeClient(
        endpoint=args.endpoint,
        model=args.model,
        timeout=args.timeout,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        seed=args.seed,
        chat_template_kwargs=json.loads(args.chat_template_kwargs),
        # The pre-warm refuses to cache a verdict it could not read; scoring is
        # the lenient path. See the module docstring.
        strict=True,
    )

    if args.dry_run:
        for index, pair in enumerate(to_ask, 1):
            print(f"\n--- pair {index}/{len(to_ask)}  key={pair.key[:16]}")
            print(json.dumps(client.request_body(pair.a, pair.b, pair.context), indent=1))
        print(
            f"\n[judge] dry run: {len(to_ask)} requests would go to "
            f"{args.endpoint.rstrip('/')}/chat/completions as model {args.model}; none were sent"
        )
        return 0

    matcher = JudgeMatcher(ask=client, cache_path=args.cache, model=args.model)
    started = time.time()
    written = commit(matcher, free_pairs, free)
    failures: list[tuple[PairRequest, str]] = []
    for number, batch in enumerate(chunked(to_ask, args.batch_size), 1):
        resolved, batch_failures = judge_batch(client, batch, args.workers)
        failures.extend(batch_failures)
        written += commit(matcher, batch, resolved)
        stats = client.stats.summary()
        print(
            f"[judge] batch {number}: {stats['calls']}/{len(to_ask)} asked, "
            f"{stats['yes']} yes, {stats['no']} no, {stats['unparseable']} unparseable, "
            f"{len(failures)} failed, {time.time() - started:.0f}s"
        )

    stats = client.stats.summary()
    report = {
        "endpoint": args.endpoint,
        "model": args.model,
        "cache": str(args.cache),
        "system_prompt": SYSTEM_PROMPT,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "user_template_sha256": hashlib.sha256(USER_TEMPLATE.encode("utf-8")).hexdigest(),
        "temperature": 0.0,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "n_pairs": len(pairs),
        "n_cached_before": len(cached),
        "n_contrast_guarded": len(guarded),
        "n_settled_without_call": len(free_pairs),
        "n_asked": len(to_ask),
        "n_written": written,
        "n_failed": len(failures),
        "failures": [{"a": p.a, "b": p.b, "error": message} for p, message in failures[:20]],
        "elapsed_seconds": time.time() - started,
        **stats,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(
        f"[judge] {written} verdicts cached ({stats['yes']} yes / {stats['no']} no from "
        f"{stats['calls']} calls), {len(failures)} pairs unresolved -> {args.cache}"
    )
    if stats["unparseable_samples"]:
        print("[judge] unparseable replies (these pairs were NOT cached):", file=sys.stderr)
        for sample in stats["unparseable_samples"][:5]:
            print(
                f"    {sample['a']!r} ~ {sample['b']!r}  finish={sample['finish_reason']}"
                f"  reasoning_field={sample['had_reasoning_field']}  reply={sample['reply']!r}",
                file=sys.stderr,
            )
        print(
            "[judge] a high rate here usually means the chat template is still emitting "
            "reasoning; the answer never reaches `content`.",
            file=sys.stderr,
        )
    if failures and not args.allow_unparseable:
        print(
            f"[judge] FAIL: {len(failures)} pairs have no verdict and are absent from the cache. "
            f"Scoring would ask them one at a time with no task context. Fix the endpoint and "
            f"re-run (the cache is append-only, so completed work is kept), or pass "
            f"--allow-unparseable to accept it deliberately.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
