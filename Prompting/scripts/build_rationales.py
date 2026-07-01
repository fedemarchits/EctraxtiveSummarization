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
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    aspects = cfg["dataset"]["aspects"]
    ref_alias = cfg["rationale"]["source_model"]
    max_tries = int(cfg["rationale"].get("max_exemplar_tries", 10))

    # Ported engine pieces (stubs until then -> clear error here).
    from engine.backends import get_backend
    from engine.data import load_split
    from prompts.fewshot import iter_exemplars

    backend = get_backend(ref_alias, args.models)
    train = load_split("train")
    seed = int(cfg["fewshot"]["seed"])

    for technique in sorted(REASONING_TECHNIQUES):
        for aspect in aspects:
            saved = False
            for ex in iter_exemplars(train, aspect, seed=seed, limit=max_tries):
                prompt = _elicitation_prompt(technique, aspect, ex.sentences)
                out = backend.generate_batch([prompt], system=SYSTEM)[0]
                pred = sorted(_parse_indices(out))
                if pred == sorted(int(i) for i in ex.gold_indices):
                    shot = RationaleShot(
                        technique=technique,
                        aspect=aspect,
                        source_model=ref_alias,
                        exemplar_sentences=list(ex.sentences),
                        gold_indices=list(ex.gold_indices),
                        rationale=_strip_answer_line(out),
                    )
                    path = save_rationale(shot)
                    print(f"[ok]   {technique}/{aspect} -> {path.name}")
                    saved = True
                    break
            if not saved:
                print(f"[skip] {technique}/{aspect}: no exemplar matched gold; "
                      "one-shot falls back to answer-only.")


if __name__ == "__main__":
    main()
