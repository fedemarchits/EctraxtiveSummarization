"""scoring_based — score each sentence 1-5, keep 4-5."""
from __future__ import annotations

from typing import Sequence

from ..base import RenderCtx, Technique
from ..registry import register
from ..shared import reasoning_example_block, render


def build(sentences: Sequence[str], aspect: str, ctx: RenderCtx) -> str:
    return render(
        aspect, sentences, ctx,
        example_override=reasoning_example_block("scoring_based", aspect, ctx),
        instructions=(
            "1. For each sentence, assign a score from 1 (low relevance) to 5 (high "
            f'relevance) for how well it expresses the "{aspect}" aspect.\n'
            "2. Select only sentences with a score of 4 or 5.\n"
            "Do not include the scores in the final output."
        ),
    )


register(Technique(name="scoring_based", build=build))
