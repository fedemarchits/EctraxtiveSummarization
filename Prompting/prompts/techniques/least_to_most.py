"""least_to_most — reason about document purpose first, then select."""
from __future__ import annotations

from typing import Sequence

from ..base import RenderCtx, Technique
from ..registry import register
from ..shared import render


def build(sentences: Sequence[str], aspect: str, ctx: RenderCtx) -> str:
    return render(
        aspect, sentences, ctx,
        instructions=(
            "1. First, consider the overall purpose of the document and how each "
            "sentence contributes to it.\n"
            f'2. Then, select ONLY sentences that primarily express the "{aspect}" aspect.'
        ),
    )


register(Technique(name="least_to_most", build=build))
