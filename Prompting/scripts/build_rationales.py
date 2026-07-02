"""Generate one-shot reasoning traces with the strong reference model, once.

For each reasoning technique x aspect:
  1. pick a fixed exemplar from the TRAIN split,
  2. ask the reference model to solve it AND reveal its reasoning,
  3. parse the trailing "Selected indices: [...]" and verify it equals gold,
  4. on match, cache the (exemplar, reasoning, gold) via prompts.rationale.

Runs once; all grid models then reuse the cached traces. Requires the engine
(engine.data + engine.backends) to be ported first — until then it raises a
clear NotImplementedError from those stubs.

Usage:
    python -m scripts.build_rationales --config configs/experiment.yaml
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Sequence

import yaml

from prompts.rationale import REASONING_TECHNIQUES, RationaleShot, save_rationale
from prompts.shared import header, numbered

SYSTEM = "You are an expert in extractive summarization."

# How to ask each technique to REVEAL its working (opposite of the run-time
# "do not include reasoning" instruction). End marker is parsed for the answer.
_ELICIT = {
    "chain_of_thought":
        "For each sentence, reason step-by-step: identify its key claim, map it "
        "to challenge/approach/outcome, and conclude Yes/No for '{aspect}'. Show "
        "all reasoning.",
    "self_ask":
        "For each sentence, write the question 'Does this primarily express "
        "\"{aspect}\"?' and answer Yes/No with a short justification. Show all Q/A.",
    "scoring_based":
        "For each sentence, give a 1-5 relevance score for '{aspect}' with a one-"
        "clause reason. Show every score.",
    "salience_inference":
        "State the salient '{aspect}' theme in one sentence, then score each "
        "sentence 0-2 for it with a brief reason. Show the theme and all scores.",
}

_END = "\nEnd with exactly one line: Selected indices: [comma-separated 1-based indices]"
_ANSWER_RE = re.compile(r"Selected indices:\s*\[([0-9,\s]*)\]")


def _elicitation_prompt(technique: str, aspect: str, sentences: Sequence[str]) -> str:
    return (
        f"{header(aspect)}\n\nInput:\n{numbered(sentences)}\n\n"
        f"Instructions:\n{_ELICIT[technique].format(aspect=aspect)}{_END}"
    )


def _parse_indices(text: str) -> List[int]:
    m = None
    for m in _ANSWER_RE.finditer(text):
        pass  # keep the last match
    if not m:
        return []
    body = m.group(1).strip()
    return [int(x) for x in body.split(",") if x.strip()]


def _strip_answer_line(text: str) -> str:
    return _ANSWER_RE.split(text)[0].rstrip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--models", default="configs/models.yaml")
    ap.add_argument("--force", action="store_true",
                    help="regenerate cells that already have a trace file")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    aspects = cfg["dataset"]["aspects"]
    ref_alias = cfg["rationale"]["source_model"]
    max_tries = int(cfg["rationale"].get("max_exemplar_tries", 10))
    # Best-overlap acceptance: a model won't reproduce human gold exactly, so we
    # keep the exemplar whose picks best overlap gold. Early-accept once overlap
    # is strong; otherwise take the best seen if it clears the floor.
    accept_f1 = float(cfg["rationale"].get("accept_f1", 0.7))
    min_f1 = float(cfg["rationale"].get("min_f1", 0.4))

    # Ported engine pieces (stubs until then -> clear error here).
    from engine.backends import GenConfig, get_backend
    from engine.data import load_split
    from engine.metrics import prf
    from prompts.fewshot import iter_exemplars
    from prompts.rationale import load_rationale

    # Traces need room to reason — the run-time default (128) would truncate.
    ref_gen = GenConfig(
        max_new_tokens=int(cfg["rationale"].get("max_new_tokens", 1024)),
        temperature=0.0,
    )
    backend = get_backend(ref_alias, args.models, ref_gen)
    train = load_split("train")
    seed = int(cfg["fewshot"]["seed"])

    for technique in sorted(REASONING_TECHNIQUES):
        for aspect in aspects:
            # Resume-safe: skip cells already cached unless --force. Lets a run
            # pick up where a quota/crash left off without re-spending.
            if not args.force and load_rationale(technique, aspect) is not None:
                print(f"[have] {technique}/{aspect} (cached; use --force to redo)")
                continue
            best = None  # (f1, exemplar, model_pred, trace)
            for ex in iter_exemplars(train, aspect, seed=seed, limit=max_tries):
                prompt = _elicitation_prompt(technique, aspect, ex.sentences)
                try:
                    out = backend.generate_batch([prompt], system=SYSTEM)[0]
                except Exception as e:  # timeout / rate limit / transient API error
                    print(f"  [warn] {technique}/{aspect}: call failed ({type(e).__name__}); "
                          "skipping this exemplar")
                    continue
                pred = sorted(_parse_indices(out))
                gold = sorted(int(i) for i in ex.gold_indices)
                f1 = prf(gold, pred)["f1"]
                if best is None or f1 > best[0]:
                    best = (f1, ex, pred, _strip_answer_line(out))
                if f1 >= accept_f1:
                    break  # strong enough, stop early
            if best is not None and best[0] >= min_f1:
                f1, ex, pred, trace = best
                shot = RationaleShot(
                    technique=technique,
                    aspect=aspect,
                    source_model=ref_alias,
                    exemplar_sentences=list(ex.sentences),
                    shown_indices=pred,
                    gold_indices=list(ex.gold_indices),
                    f1=f1,
                    rationale=trace,
                )
                path = save_rationale(shot)
                print(f"[ok]   {technique}/{aspect} -> {path.name} (F1={f1:.2f})")
            else:
                bf1 = best[0] if best else 0.0
                print(f"[skip] {technique}/{aspect}: best overlap F1={bf1:.2f} < {min_f1}; "
                      "one-shot falls back to answer-only.")


if __name__ == "__main__":
    main()
