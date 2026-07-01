"""vanilla — direct baseline. Template for every plain technique."""
from __future__ import annotations

from typing import Sequence

from ..base import RenderCtx, Technique
from ..registry import register
from ..shared import render


def build(sentences: Sequence[str], aspect: str, ctx: RenderCtx) -> str:
    return render(
        aspect, sentences, ctx,
        instructions=f'Select ONLY sentences that primarily express the "{aspect}" aspect.',
    )


register(Technique(name="vanilla", build=build))
