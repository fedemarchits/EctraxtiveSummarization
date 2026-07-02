"""salience_inference — infer the salient aspect theme, score 0-2, keep the strongest.

salience_inference -> one-shot answer-only ; salience_inference_trace -> 397B trace.
"""
from __future__ import annotations

from typing import Sequence

from ..base import RenderCtx, Shot, Technique
from ..registry import register
from ..shared import reasoning_example_block, render

_NAME = "salience_inference"


def _instructions(aspect: str) -> str:
    return (
        f'1. First, infer the most salient "{aspect}"-related theme of the document '
        "in one sentence (internally).\n"
        "2. Score each sentence 0-2 for how strongly it expresses that theme:\n"
        f'   - 0: not related to "{aspect}"\n'
        "   - 1: weakly or partially related\n"
        f'   - 2: directly and centrally expresses "{aspect}"\n'
        "3. Select all sentences scoring 2.\n"
        "4. If none score 2, select those scoring 1.\n"
        "Do not include the scores or the inferred theme in the final output."
    )


def _build(sentences: Sequence[str], aspect: str, ctx: RenderCtx, use_trace: bool) -> str:
    override = reasoning_example_block(_NAME, aspect, ctx) if use_trace else ""
    return render(aspect, sentences, ctx,
                  instructions=_instructions(aspect), example_override=override)


register(Technique(name=_NAME, build=lambda s, a, c: _build(s, a, c, False)))
register(Technique(name=_NAME + "_trace", shots=(Shot.ONE,),
                   build=lambda s, a, c: _build(s, a, c, True)))
