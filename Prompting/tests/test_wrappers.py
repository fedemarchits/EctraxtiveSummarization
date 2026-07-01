"""Wrapper logic — no models. Fake backend returns canned JSON."""
from prompts.base import Cap, Shot, Variant

from engine.wrappers import dynamic_cap, majority_vote, select_document


class FakeBackend:
    """Returns a fixed JSON per generate_batch call; cycles through `scripts`."""
    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.calls = 0

    def generate_batch(self, prompts, system=None):
        out = self.scripts[min(self.calls, len(self.scripts) - 1)]
        self.calls += 1
        return out  # list aligned with prompts


def test_majority_vote_threshold():
    samples = [[1, 2], [1, 3], [1, 2]]  # 3 samples: 1 x3, 2 x2, 3 x1
    assert majority_vote(samples, 3) == [1, 2]      # >= 2 kept
    assert majority_vote(samples, 3, threshold=3) == [1]  # unanimous only


def test_dynamic_cap_never_adds_and_falls_back():
    sents = ["s1", "s2", "s3", "s4"]
    # model prunes [1,2,3] -> keeps [2], plus a hallucinated 4 (dropped: not in selected)
    be = FakeBackend([['{"selected_sentences": [2, 4]}']])
    assert dynamic_cap(be, sents, "challenge", [1, 2, 3], None) == [2]
    # empty selection -> empty, no call needed
    assert dynamic_cap(be, sents, "challenge", [], None) == []
    # garbage prune -> fall back to original selection
    be2 = FakeBackend([["no json"]])
    assert dynamic_cap(be2, sents, "challenge", [1, 3], None) == [1, 3]


def test_select_document_self_consistency_majority():
    aspects = ["challenge"]
    sents = ["a", "b", "c"]
    variant = Variant("vanilla", Shot.ZERO, Cap.UNCAPPED)
    # 3 sampled passes for the single aspect: [1,2], [1], [1,2] -> majority [1]... 2 appears 2/3
    be = FakeBackend([
        ['{"selected_sentences": [1, 2]}'],
        ['{"selected_sentences": [1]}'],
        ['{"selected_sentences": [1, 2]}'],
    ])
    sc = {"enabled": True, "n_samples": 3}
    preds, raws = select_document(be, ["p"], aspects, sents, {}, variant, sc, None, None)
    assert preds["challenge"] == [1, 2]  # 1 x3, 2 x2, both >= 2
    assert isinstance(raws["challenge"], list) and len(raws["challenge"]) == 3  # N raw strings


def test_select_document_fixed_cap_applies_when_no_wrappers():
    aspects = ["challenge"]
    sents = ["a", "b", "c", "d"]
    variant = Variant("vanilla", Shot.ZERO, Cap.CAPPED)
    be = FakeBackend([['{"selected_sentences": [1, 2, 3, 4]}']])
    preds, raws = select_document(be, ["p"], aspects, sents, {"challenge": 2}, variant, None, None, None)
    assert preds["challenge"] == [1, 2]  # capped to K=2
    assert isinstance(raws["challenge"], str)  # single raw string when no self-consistency
