"""self_ask — internal Yes/No per sentence, return the Yes indices.

self_ask       -> one-shot answer-only ; self_ask_trace -> one-shot 397B trace.
"""
from __future__ import annotations

from typing import Sequence

from ..base import RenderCtx, Shot, Technique
from ..registry import register
from ..shared import reasoning_example_block, render

_NAME = "self_ask"


def _instructions(aspect: str) -> str:
    return (
        '1. For each sentence, ask internally: "Does this sentence primarily '
        f'express the \'{aspect}\' aspect?"\n'
        '2. Answer "Yes" or "No" internally.\n'
        '3. Select all sentences answered "Yes".\n'
        "Do not include the questions or answers in the final output."
    )


def _build(sentences: Sequence[str], aspect: str, ctx: RenderCtx, use_trace: bool) -> str:
    override = reasoning_example_block(_NAME, aspect, ctx) if use_trace else ""
    return render(aspect, sentences, ctx,
                  instructions=_instructions(aspect), example_override=override)


register(Technique(name=_NAME, build=lambda s, a, c: _build(s, a, c, False)))
register(Technique(name=_NAME + "_trace", shots=(Shot.ONE,),
                   build=lambda s, a, c: _build(s, a, c, True)))
