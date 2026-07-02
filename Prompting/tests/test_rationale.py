"""Cached reasoning traces: reasoning techniques render the real trace when
present, and fall back to answer-only when absent. Uses a temp cache dir so no
real trace files are needed."""
import json
from pathlib import Path

from prompts import rationale
from prompts.base import Cap, Exemplar, RenderCtx, Shot
from prompts.registry import all_techniques

DOC = ["We study X.", "Prior work fails on Y.", "Our method fixes Y.", "Results improve 10%."]
EX = Exemplar(["A is slow.", "We propose B.", "B wins."], [3])


def _one_shot_ctx():
    return RenderCtx(shot=Shot.ONE, cap=Cap.UNCAPPED, exemplar=EX)


def _write_trace(base: Path, technique: str, aspect: str) -> None:
    base.mkdir(parents=True, exist_ok=True)
    (base / f"{technique}__{aspect}.json").write_text(json.dumps({
        "technique": technique,
        "aspect": aspect,
        "source_model": "qwen35_397b",
        "exemplar_sentences": ["A is slow.", "We propose B.", "B wins."],
        "shown_indices": [3],
        "gold_indices": [3],
        "f1": 1.0,
        "rationale": "Sentence 1: a problem -> No\nSentence 2: a method -> No\n"
                     "Sentence 3: a result -> Yes",
    }))


def test_reasoning_technique_uses_cached_trace(tmp_path, monkeypatch):
    monkeypatch.setattr(rationale, "DEFAULT_RATIONALE_DIR", tmp_path)
    rationale._cache.clear()
    _write_trace(tmp_path, "chain_of_thought", "outcome")

    # the *_trace technique consumes the trace; the base one is answer-only
    out = all_techniques()["chain_of_thought_trace"].build(DOC, "outcome", _one_shot_ctx())
    assert "Reasoning:" in out
    assert "reference model" in out
    assert "-> Yes" in out
    assert "Selected indices: [3]" in out

    base = all_techniques()["chain_of_thought"].build(DOC, "outcome", _one_shot_ctx())
    assert "Reasoning:" not in base  # base stays answer-only even when a trace exists


def test_falls_back_to_answer_only_when_no_trace(tmp_path, monkeypatch):
    monkeypatch.setattr(rationale, "DEFAULT_RATIONALE_DIR", tmp_path)
    rationale._cache.clear()
    # no trace written -> even the _trace technique falls back to answer-only
    out = all_techniques()["self_ask_trace"].build(DOC, "challenge", _one_shot_ctx())
    assert "Reasoning:" not in out
    assert "Example (one-shot; exemplar from the training split):" in out
    assert "Selected indices: [3]" in out
