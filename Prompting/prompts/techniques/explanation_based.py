"""explanation_based — select only sentences justifiable as belonging to the aspect."""
from __future__ import annotations

from typing import Sequence

from ..base import RenderCtx, Technique
from ..registry import register
from ..shared import render


def build(sentences: Sequence[str], aspect: str, ctx: RenderCtx) -> str:
    return render(
        aspect, sentences, ctx,
        instructions=(
            "1. For each sentence you would select, form a brief internal justification "
            f'(one sentence) of why it expresses the "{aspect}" aspect.\n'
            "2. Keep only sentences with a clear justification; reject the rest.\n"
            "3. Select the indices of the justified sentences.\n"
            "Do not include the justifications in the final output."
        ),
    )


register(Technique(name="explanation_based", build=build))
