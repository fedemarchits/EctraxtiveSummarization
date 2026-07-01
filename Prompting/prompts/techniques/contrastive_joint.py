"""contrastive_joint — classify each sentence as challenge/approach/outcome/none,
then keep those labeled with the target aspect.

zero-shot: contrastive classification in one pass.
one-shot : the "joint_self_ask" form. NOTE: ACLSum gold gives only the target
           aspect's labels per call, not a full 4-way labeling, so the example
           shows the target-vs-not-target boundary (real gold) rather than a
           fabricated full 4-way trace. Reduces cross-aspect false positives.
"""
from __future__ import annotations

from typing import Sequence

from ..base import RenderCtx, Technique
from ..registry import register
from ..shared import render


def build(sentences: Sequence[str], aspect: str, ctx: RenderCtx) -> str:
    return render(
        aspect, sentences, ctx,
        instructions=(
            "1. For each sentence, assign exactly one label: challenge, approach, "
            "outcome, or none.\n"
            "2. A sentence gets a label only if it PRIMARILY expresses that aspect; "
            "otherwise none.\n"
            f'3. Select the sentences labeled "{aspect}".\n'
            "Do not include the per-sentence labels in the final output."
        ),
    )


register(Technique(name="contrastive_joint", build=build))
