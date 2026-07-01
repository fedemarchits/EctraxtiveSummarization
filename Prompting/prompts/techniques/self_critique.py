"""self_critique — draft candidates, critique, output pruned final list.

zero-shot: draft then self-critique in one pass.
one-shot : the "self_refinement" form — the example shows a draft list being
           pruned down to the final gold list, teaching the pruning operation.
Reduces over-selection and keyword-only matches.
"""
from __future__ import annotations

from typing import Sequence

from ..base import RenderCtx, Shot, Technique
from ..registry import register
from ..shared import numbered, render

_INSTRUCTIONS = (
    "1. Draft a candidate list of sentences that seem to express the "
    '"{aspect}" aspect.\n'
    "2. Critique the draft: remove any sentence that only shares surface "
    'keywords or does not PRIMARILY express "{aspect}".\n'
    "3. Output the pruned final selection.\n"
    "Do not include the draft or the critique in the final output."
)


def _refinement_example(aspect: str, ex) -> str:
    gold = list(ex.gold_indices)
    n = len(ex.sentences)
    spurious = next((i for i in range(1, n + 1) if i not in gold), None)
    draft = sorted(set(gold + ([spurious] if spurious else [])))
    return (
        "Example (one-shot; exemplar from the training split):\n"
        "Input:\n" + numbered(ex.sentences) + "\n\n"
        f"Draft candidates: {draft}\n"
        "Critique: remove sentences that only share keywords or do not "
        f'primarily express "{aspect}".\n'
        f"Final selection: {gold}\n\n"
        "---\n\nNew task:\n"
    )


def build(sentences: Sequence[str], aspect: str, ctx: RenderCtx) -> str:
    override = ""
    if ctx.shot is Shot.ONE and ctx.exemplar is not None:
        override = _refinement_example(aspect, ctx.exemplar)
    return render(
        aspect, sentences, ctx,
        instructions=_INSTRUCTIONS.format(aspect=aspect),
        example_override=override,
    )


register(Technique(name="self_critique", build=build))
