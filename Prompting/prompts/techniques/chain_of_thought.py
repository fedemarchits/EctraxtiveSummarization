"""chain_of_thought — step-by-step map each sentence to an aspect, then select.

Registered twice for the trace ablation:
  chain_of_thought        -> one-shot example is answer-only
  chain_of_thought_trace  -> one-shot example is the 397B reasoning trace
Only one-shot differs, so _trace is one-shot only.
"""
from __future__ import annotations

from typing import Sequence

from ..base import RenderCtx, Shot, Technique
from ..registry import register
from ..shared import reasoning_example_block, render

_NAME = "chain_of_thought"


def _instructions(aspect: str) -> str:
    return (
        "1. For each sentence, think step-by-step (internally):\n"
        "   a. Identify the key claim or information the sentence conveys.\n"
        "   b. Map it to one of the three aspects (challenge / approach / outcome).\n"
        f'   c. Conclude whether it matches "{aspect}": Yes or No.\n'
        "2. Select the sentences concluded Yes.\n"
        "Do not include the reasoning in the final output."
    )


def _build(sentences: Sequence[str], aspect: str, ctx: RenderCtx, use_trace: bool) -> str:
    override = reasoning_example_block(_NAME, aspect, ctx) if use_trace else ""
    return render(aspect, sentences, ctx,
                  instructions=_instructions(aspect), example_override=override)


register(Technique(name=_NAME, build=lambda s, a, c: _build(s, a, c, False)))
register(Technique(name=_NAME + "_trace", shots=(Shot.ONE,),
                   build=lambda s, a, c: _build(s, a, c, True)))
