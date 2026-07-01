"""self_ask — internal Yes/No per sentence, return the Yes indices."""
from __future__ import annotations

from typing import Sequence

from ..base import RenderCtx, Technique
from ..registry import register
from ..shared import reasoning_example_block, render


def build(sentences: Sequence[str], aspect: str, ctx: RenderCtx) -> str:
    return render(
        aspect, sentences, ctx,
        example_override=reasoning_example_block("self_ask", aspect, ctx),
        instructions=(
            '1. For each sentence, ask internally: "Does this sentence primarily '
            f'express the \'{aspect}\' aspect?"\n'
            '2. Answer "Yes" or "No" internally.\n'
            '3. Select all sentences answered "Yes".\n'
            "Do not include the questions or answers in the final output."
        ),
    )


register(Technique(name="self_ask", build=build))
