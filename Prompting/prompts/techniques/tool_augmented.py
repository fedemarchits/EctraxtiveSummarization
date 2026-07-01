"""tool_augmented — real TF-IDF centrality scores annotated on each sentence."""
from __future__ import annotations

from typing import Sequence

from ..base import RenderCtx, Technique
from ..centrality import numbered_with_centrality
from ..registry import register
from ..shared import render

_PREAMBLE = (
    "Each sentence is annotated with a centrality score. Higher scores indicate "
    "greater importance/relevance in the document. Use these scores together with "
    "the sentence content to guide selection."
)


def build(sentences: Sequence[str], aspect: str, ctx: RenderCtx) -> str:
    return render(
        aspect, sentences, ctx,
        preamble=_PREAMBLE,
        render_fn=numbered_with_centrality,
        instructions=(
            "1. Consider both the semantic content and the centrality score of each sentence.\n"
            f'2. Prioritize higher-scored sentences that also match the "{aspect}" aspect.\n'
            f'3. Select sentences that primarily describe the "{aspect}" aspect.'
        ),
    )


register(Technique(name="tool_augmented", build=build))
