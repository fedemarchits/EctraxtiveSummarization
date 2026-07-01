"""negative_aware — one-shot only. The example highlights near-miss negatives:
sentences that look relevant but must NOT be selected. Reduces false positives
and sharpens aspect boundaries.

One-shot by nature: the whole point is showing the negatives, so there is no
zero-shot variant (restricted via shots=).
"""
from __future__ import annotations

from typing import Sequence

from ..base import Cap, RenderCtx, Shot, Technique
from ..registry import register
from ..shared import numbered, render


def _negative_example(aspect: str, ex) -> str:
    gold = list(ex.gold_indices)
    n = len(ex.sentences)
    # Near-misses = non-selected sentences, shown as tempting-but-excluded.
    near_miss = [i for i in range(1, n + 1) if i not in gold]
    nm = ", ".join(map(str, near_miss)) if near_miss else "none"
    return (
        "Example (one-shot; exemplar from the training split):\n"
        "Input:\n" + numbered(ex.sentences) + "\n\n"
        f"Selected indices: {gold}\n"
        f"Near-miss sentences NOT selected (mention related terms but do not "
        f'primarily express "{aspect}"): {nm}\n\n'
        "---\n\nNew task:\n"
    )


def build(sentences: Sequence[str], aspect: str, ctx: RenderCtx) -> str:
    override = ""
    if ctx.shot is Shot.ONE and ctx.exemplar is not None:
        override = _negative_example(aspect, ctx.exemplar)
    return render(
        aspect, sentences, ctx,
        instructions=(
            f'Select ONLY sentences that primarily express the "{aspect}" aspect. '
            "Exclude sentences that merely mention related terms but primarily "
            "express a different aspect (near-misses), as shown in the example."
        ),
        example_override=override,
    )


# One-shot only; both cap settings.
register(Technique(
    name="negative_aware",
    build=build,
    shots=(Shot.ONE,),
    caps=(Cap.UNCAPPED, Cap.CAPPED),
    note="one-shot only; example shows near-miss negatives",
))
