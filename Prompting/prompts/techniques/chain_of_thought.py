"""chain_of_thought — step-by-step map each sentence to an aspect, then select."""
from __future__ import annotations

from typing import Sequence

from ..base import RenderCtx, Technique
from ..registry import register
from ..shared import reasoning_example_block, render


def build(sentences: Sequence[str], aspect: str, ctx: RenderCtx) -> str:
    return render(
        aspect, sentences, ctx,
        example_override=reasoning_example_block("chain_of_thought", aspect, ctx),
        instructions=(
            "1. For each sentence, think step-by-step (internally):\n"
            "   a. Identify the key claim or information the sentence conveys.\n"
            "   b. Map it to one of the three aspects (challenge / approach / outcome).\n"
            f'   c. Conclude whether it matches "{aspect}": Yes or No.\n'
            "2. Select the sentences concluded Yes.\n"
            "Do not include the reasoning in the final output."
        ),
    )


register(Technique(name="chain_of_thought", build=build))