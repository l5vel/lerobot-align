"""Direction-blind counterexamples for the September 2026 harness audit."""

import itertools

import pytest

from .joint import dense_caption_score, edit_score, f1_at_iou, mof
from .score import score_episode
from .semantic import ExactMatcher, label_agreement


def spans(*items):
    return [{"start": s, "end": e, "text": t} for s, e, t in items]


def test_edit_collapses_consecutive_equal_labels():
    ref = spans((0, 2, "pick"), (2, 4, "place"))
    pred = spans((0, 1, "pick"), (1, 2, "pick"), (2, 4, "place"))
    assert edit_score(pred, ref, ExactMatcher()) == 1


def test_segmental_f1_does_not_fall_back_from_claimed_best_reference():
    # Official temporal traversal: the last 'wait' chooses the already-hit
    # 21..33 reference (IoU .25), rather than falling back to 37..40 (also .25).
    # The old global greedy matcher returned .8; MS-TCN returns .6.
    pred = spans((0, 6, "place"), (6, 22, "pick"), (22, 27, "wait"),
                 (27, 28, "place"), (28, 40, "wait"))
    ref = spans((0, 10, "place"), (10, 21, "pick"), (21, 33, "wait"),
                (33, 37, "place"), (37, 40, "wait"))
    assert f1_at_iou(pred, ref, ExactMatcher(), threshold=.25)["f1@25"] == .6



class NonTransitiveMatcher:
    name = "synthetic"
    edges = {frozenset(p) for p in [("flexible", "ref1"), ("flexible", "ref2"),
                                   ("specific", "ref1")]}

    def equivalent(self, a, b):
        return a == b or frozenset((a, b)) in self.edges

    def similarity(self, a, b):
        return float(self.equivalent(a, b))


def test_label_multiset_is_invariant_to_both_orders():
    for p in itertools.permutations(["flexible", "specific"]):
        for r in itertools.permutations(["ref1", "ref2"]):
            assert label_agreement(p, r, NonTransitiveMatcher())["label_f1"] == 1


def test_dense_caption_missing_reference_retains_denominator():
    ref = spans((0, 1, "pick"), (1, 2, "place"))
    wrong = spans((0, 1, "pick"), (1, 2, "wait"))
    full = dense_caption_score(wrong, ref, ExactMatcher())["dvc_mean"]
    partial = dense_caption_score(wrong[:1], ref, ExactMatcher())["dvc_mean"]
    assert full == partial == .5
    assert dense_caption_score(ref + ref, ref, ExactMatcher())["dvc_mean"] == 1


def test_mof_exact_substep_episode():
    ref = spans((1, 1.01, "pick"))
    assert mof(ref, ref, ExactMatcher())["mof"] == 1


@pytest.mark.parametrize("bad", [[{"start": 0, "text": "pick"}],
                                 [{"start": None, "end": 2}], [None]])
def test_malformed_prediction_is_failed_row(bad):
    result = score_episode(dataset="d", episode=1, arm="a", predicted=bad,
                           reference=spans((0, 2, "pick")), matcher=ExactMatcher())
    assert not result.ok
    assert "bad prediction" in result.error
