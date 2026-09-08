"""Label-matching metrics for free-form subtask descriptions.

The two tools invent their own wording, so no metric that string-compares
labels can be fair. This module provides the matching layer that everything
label-aware is built on, with three interchangeable backends of increasing
cost and fidelity:

``exact``      normalised string equality -- cheap, only useful as a floor
``embedding``  cosine similarity from a sentence encoder
``judge``      an LLM asked whether two phrases denote the same robot action

The backend is a parameter, never a hard-coded choice, because the headline
comparison must be shown to be robust to it. A result that only holds under
one matcher is a result about the matcher.

Blinding
--------
``judge`` is the only backend that could plausibly favour one tool over the
other. ``JudgeMatcher`` therefore never sees which arm produced a label: it is
handed an unordered pair and returns a symmetric verdict, and the caller is
responsible for caching on a canonicalised, arm-independent key. See
``evaluation/scripts/run_judge.py``.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

_ARTICLES = {"the", "a", "an"}
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def normalise_label(text: str) -> str:
    """Casefold, strip punctuation/articles, collapse whitespace.

    Applied before every backend so that trivial formatting differences
    ("Pick up the red can." vs "pick up red can") never count as a semantic
    difference. This is intentionally aggressive: it can only make the two
    tools look MORE similar, so it cannot manufacture a win for either.
    """
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    text = _PUNCT.sub(" ", text)
    tokens = [t for t in _WS.split(text) if t and t not in _ARTICLES]
    return " ".join(tokens)


class Matcher(Protocol):
    """Returns a similarity in [0, 1] for a pair of labels."""

    name: str

    def similarity(self, left: str, right: str) -> float: ...

    def equivalent(self, left: str, right: str) -> bool: ...


@dataclass
class ExactMatcher:
    """Normalised string equality. The conservative floor."""

    name: str = "exact"

    def similarity(self, left: str, right: str) -> float:
        return 1.0 if normalise_label(left) == normalise_label(right) else 0.0

    def equivalent(self, left: str, right: str) -> bool:
        return self.similarity(left, right) >= 1.0


@dataclass
class TokenF1Matcher:
    """Bag-of-words F1 over normalised tokens.

    A lexical middle ground that needs no model. DIAGNOSTIC ONLY -- it must
    never be the primary matcher, because bag-of-words similarity is blind to
    exactly the tokens that carry the meaning here: "open the fridge door" and
    "close the fridge door" share three of four content tokens and score 0.67,
    while denoting opposite actions. The default threshold is set above that
    value so the known antonym pairs in this corpus fall on the correct side,
    but the failure mode is structural and the threshold only papers over it.
    Use it to sanity-check the embedding backend, not to produce headline
    numbers.
    """

    threshold: float = 0.8
    name: str = "token_f1"

    def similarity(self, left: str, right: str) -> float:
        a = set(normalise_label(left).split())
        b = set(normalise_label(right).split())
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        overlap = len(a & b)
        if overlap == 0:
            return 0.0
        precision = overlap / len(a)
        recall = overlap / len(b)
        return 2 * precision * recall / (precision + recall)

    def equivalent(self, left: str, right: str) -> bool:
        return self.similarity(left, right) >= self.threshold


# Token groups whose members flip the meaning of an otherwise identical phrase.
# Sentence encoders are measurably blind to several of these: calibration on
# this corpus scored "move the bag to the left" against "move the bag to the
# right" at 0.983 cosine, above any threshold that still admits real
# paraphrases. No choice of threshold can separate them, because the
# information is not in the embedding. The guard below restores it by refusing
# a match when the two labels take different sides of one of these contrasts.
# Each group maps surface forms to the SIDE of the contrast they denote, so
# that inflections and synonyms of the same side do not read as a conflict.
# The earlier flat-set version treated "open" vs "opening" and "in" vs "into"
# as opposite sides and hard-zeroed ordinary paraphrases of the corpus's most
# common labels.
CONTRAST_GROUPS: tuple[dict[str, str], ...] = (
    {"left": "L", "leftmost": "L", "right": "R", "rightmost": "R"},
    {
        "open": "O", "opens": "O", "opening": "O", "opened": "O",
        "close": "C", "closes": "C", "closing": "C", "closed": "C", "shut": "C",
    },
    {"up": "U", "upward": "U", "upwards": "U", "down": "D", "downward": "D", "downwards": "D"},
    {"in": "I", "into": "I", "inside": "I", "on": "N", "onto": "N", "atop": "N"},
    {"push": "P", "pushes": "P", "pushing": "P", "pull": "L", "pulls": "L", "pulling": "L"},
    {"first": "1", "second": "2", "third": "3", "fourth": "4"},
    {"top": "T", "bottom": "B", "upper": "T", "lower": "B"},
    {"front": "F", "back": "K", "rear": "K"},
)


def contrast_conflict(left: str, right: str) -> bool:
    """True when the two labels differ on a meaning-flipping token contrast.

    Only fires when BOTH sides mention the contrast and they disagree, so it
    never blocks a match merely because one label is less specific than the
    other ("put the cup down" vs "put the cup on the left shelf" is not a
    conflict; "left shelf" vs "right shelf" is).
    """
    a = set(normalise_label(left).split())
    b = set(normalise_label(right).split())
    for group in CONTRAST_GROUPS:
        sides_a = {group[t] for t in a if t in group}
        sides_b = {group[t] for t in b if t in group}
        # Only a genuine disagreement blocks: both sides must name the contrast,
        # each must name exactly one side of it, and those sides must differ.
        if len(sides_a) == 1 and len(sides_b) == 1 and sides_a != sides_b:
            return True
    return False


@dataclass
class EmbeddingMatcher:
    """Cosine similarity from a sentence-transformers encoder, with a guard.

    The model is loaded once and every distinct label in the corpus is encoded
    once, so the cost is negligible next to the annotation runs themselves.

    ``threshold`` MUST come from ``evaluation/scripts/calibrate_matcher.py``
    rather than intuition. The fallback is 0.93; the saved calibration selects
    0.91, which the sweep passes explicitly. A
    plausible-looking 0.6 admits 70 false matches out of 273 hard negatives,
    which would have inflated every label-aware metric for every arm. The
    default is conservative and must be re-derived per corpus.

    ``guard_contrasts`` additionally blocks pairs that disagree on a
    meaning-flipping token (left/right, open/close, ...), which no threshold
    can catch because the encoder does not represent the distinction.
    """

    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    threshold: float = 0.93
    guard_contrasts: bool = True
    name: str = "embedding"
    _model: Any = field(default=None, repr=False)
    _cache: dict[str, Any] = field(default_factory=dict, repr=False)

    def _encode(self, text: str):
        key = normalise_label(text)
        if key not in self._cache:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
            self._cache[key] = self._model.encode(key, normalize_embeddings=True)
        return self._cache[key]

    def warm(self, labels: Iterable[str]) -> None:
        """Pre-encode a whole corpus in one batched call."""
        keys = sorted({normalise_label(t) for t in labels} - set(self._cache))
        if not keys:
            return
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        vectors = self._model.encode(keys, normalize_embeddings=True, batch_size=64)
        self._cache.update(dict(zip(keys, vectors, strict=True)))

    def similarity(self, left: str, right: str) -> float:
        if self.guard_contrasts and contrast_conflict(left, right):
            return 0.0
        a, b = self._encode(left), self._encode(right)
        return float(max(0.0, min(1.0, float((a * b).sum()))))

    def equivalent(self, left: str, right: str) -> bool:
        return self.similarity(left, right) >= self.threshold


def judge_cache_key(left: str, right: str, model: str) -> str:
    """Arm-independent, order-independent cache key.

    Sorting the two normalised labels before hashing guarantees that the judge
    is asked the identical question regardless of which tool produced which
    side, so a cached verdict can never encode arm identity.
    """
    a, b = sorted((normalise_label(left), normalise_label(right)))
    payload = json.dumps({"a": a, "b": b, "model": model}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class JudgeMatcher:
    """LLM equivalence judgement, blinded and cached on disk.

    ``ask`` receives the two labels in sorted order and the task context, and
    must return True when they denote the same physical robot action. The
    caller supplies it so this module stays free of any client dependency.
    """

    ask: Callable[[str, str, str], bool]
    cache_path: Path
    model: str = "unspecified"
    name: str = "judge"
    _cache: dict[str, bool] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.cache_path = Path(self.cache_path)
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                self._cache[row["key"]] = bool(row["equivalent"])

    def _lookup(self, left: str, right: str, context: str) -> bool:
        # The guard is applied before the judge too, so all backends agree on
        # the cases where the answer is known a priori and no tokens are spent
        # asking. An LLM judge normally gets these right, but "normally" is not
        # a property a metric should depend on.
        if contrast_conflict(left, right):
            return False
        key = judge_cache_key(left, right, self.model)
        if key in self._cache:
            return self._cache[key]
        a, b = sorted((normalise_label(left), normalise_label(right)))
        verdict = bool(self.ask(a, b, context))
        self._cache[key] = verdict
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps({"key": key, "a": a, "b": b, "model": self.model, "equivalent": verdict})
                + "\n"
            )
        return verdict

    def similarity(self, left: str, right: str, context: str = "") -> float:
        return 1.0 if self._lookup(left, right, context) else 0.0

    def equivalent(self, left: str, right: str, context: str = "") -> bool:
        return self._lookup(left, right, context)


# --------------------------------------------------------------------------
# vocabulary projection
# --------------------------------------------------------------------------

def project_to_vocabulary(
    labels: Sequence[str], vocabulary: Sequence[str], matcher: Matcher
) -> list[str | None]:
    """Map each free-form label onto the dataset's closed human vocabulary.

    Every L5vel dataset uses a small closed set of human labels (5-9 per
    dataset), so projecting predictions onto that set turns an open-ended
    captioning problem into a classification problem we can score with
    per-class precision/recall and a confusion matrix. A label that matches
    nothing maps to ``None`` and is counted as a hallucination rather than
    being silently dropped.
    """
    out: list[str | None] = []
    for label in labels:
        best, best_score = None, 0.0
        for candidate in vocabulary:
            score = matcher.similarity(label, candidate)
            if score > best_score:
                best, best_score = candidate, score
        out.append(best if (best is not None and matcher.equivalent(label, best)) else None)
    return out


def label_agreement(
    predicted: Sequence[str], reference: Sequence[str], matcher: Matcher
) -> dict[str, float]:
    """Set-level precision/recall of the predicted label multiset.

    Deliberately ignores time: it answers "did the tool name the right steps",
    leaving "did it put them in the right place" to the segmentation metrics.
    Multiset rather than set, because several datasets legitimately repeat a
    label within one episode and collapsing duplicates would hide a real
    under-generation failure.
    """
    # Maximum-cardinality bipartite matching. Equivalence from an embedding
    # threshold need not be transitive: greedy consumption is order-dependent.
    edges = [[j for j, candidate in enumerate(reference)
              if matcher.equivalent(label, candidate)] for label in predicted]
    owners: dict[int, int] = {}

    def augment(i: int, visited: set[int]) -> bool:
        for j in edges[i]:
            if j in visited:
                continue
            visited.add(j)
            if j not in owners or augment(owners[j], visited):
                owners[j] = i
                return True
        return False

    hits = sum(augment(i, set()) for i in range(len(predicted)))
    precision = hits / len(predicted) if predicted else 0.0
    recall = hits / len(reference) if reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "label_precision": precision,
        "label_recall": recall,
        "label_f1": f1,
        "n_pred_labels": float(len(predicted)),
        "n_true_labels": float(len(reference)),
        "hallucinated": float(len(predicted) - hits),
        "missed": float(len(reference) - hits),
    }
