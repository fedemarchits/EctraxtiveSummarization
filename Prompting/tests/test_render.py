"""Every technique must render in all its declared variants without error,
and honor cap / one-shot injection."""
from prompts.base import Cap, Exemplar, RenderCtx, Shot, expand
from prompts.registry import all_techniques

EX = Exemplar(
    sentences=["The model is slow.", "We propose a fast solver.", "It wins on all metrics."],
    gold_indices=[2],
)
DOC = ["We study X.", "Prior work fails on Y.", "Our method fixes Y.", "Results improve by 10%."]


def _ctx(v):
    return RenderCtx(
        shot=v.shot,
        cap=v.cap,
        k=3 if v.cap is Cap.CAPPED else None,
        exemplar=EX if v.shot is Shot.ONE else None,
    )


def test_all_variants_render():
    for tech in all_techniques().values():
        for v in expand(tech):
            out = tech.build(DOC, "challenge", _ctx(v))
            assert isinstance(out, str) and len(out) > 50
            assert "selected_sentences" in out
            if v.cap is Cap.CAPPED:
                assert "AT MOST 3" in out
            if v.shot is Shot.ONE:
                assert "Example (one-shot" in out
                assert "New task:" in out


def test_negative_aware_is_one_shot_only():
    tech = all_techniques()["negative_aware"]
    shots = {v.shot for v in expand(tech)}
    assert shots == {Shot.ONE}
