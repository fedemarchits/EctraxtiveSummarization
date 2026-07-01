"""salience_inference — infer the salient aspect theme, score 0-2, keep the strongest."""
from __future__ import annotations

from typing import Sequence

from ..base import RenderCtx, Technique
from ..registry import register
from ..shared import reasoning_example_block, render


def build(sentences: Sequence[str], aspect: str, ctx: RenderCtx) -> str:
    return render(
        aspect, sentences, ctx,
        example_override=reasoning_example_block("salience_inference", aspect, ctx),
        instructions=(
            f'1. First, infer the most salient "{aspect}"-related theme of the document '
            "in one sentence (internally).\n"
            "2. Score each sentence 0-2 for how strongly it expresses that theme:\n"
            f'   - 0: not related to "{aspect}"\n'
            "   - 1: weakly or partially related\n"
            f'   - 2: directly and centrally expresses "{aspect}"\n'
            "3. Select all sentences scoring 2.\n"
            "4. If none score 2, select those scoring 1.\n"
            "Do not include the scores or the inferred theme in the final output."
        ),
    )


register(Technique(name="salience_inference", build=build))
