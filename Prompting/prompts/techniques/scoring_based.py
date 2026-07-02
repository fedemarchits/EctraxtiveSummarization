"""scoring_based — score each sentence 1-5, keep 4-5.

scoring_based  -> one-shot answer-only ; scoring_based_trace -> one-shot 397B trace.
"""
from __future__ import annotations

from typing import Sequence

from ..base import RenderCtx, Shot, Technique
from ..registry import register
from ..shared import reasoning_example_block, render

_NAME = "scoring_based"


def _instructions(aspect: str) -> str:
    return (
        "1. For each sentence, assign a score from 1 (low relevance) to 5 (high "
        f'relevance) for how well it expresses the "{aspect}" aspect.\n'
        "2. Select only sentences with a score of 4 or 5.\n"
        "Do not include the scores in the final output."
    )


def _build(sentences: Sequence[str], aspect: str, ctx: RenderCtx, use_trace: bool) -> str:
    override = reasoning_example_block(_NAME, aspect, ctx) if use_trace else ""
    return render(aspect, sentences, ctx,
                  instructions=_instructions(aspect), example_override=override)


register(Technique(name=_NAME, build=lambda s, a, c: _build(s, a, c, False)))
register(Technique(name=_NAME + "_trace", shots=(Shot.ONE,),
                   build=lambda s, a, c: _build(s, a, c, True)))
