"""simulated_tool_augmented — imagined check_aspect(sentence, aspect) pseudo-tool."""
from __future__ import annotations

from typing import Sequence

from ..base import RenderCtx, Technique
from ..registry import register
from ..shared import render


def build(sentences: Sequence[str], aspect: str, ctx: RenderCtx) -> str:
    return render(
        aspect, sentences, ctx,
        instructions=(
            "1. For each sentence, use the internal `check_aspect(sentence, aspect)` tool.\n"
            f'2. The tool returns "match" if the sentence primarily expresses the "{aspect}" '
            'aspect, otherwise "no_match".\n'
            '3. Select the sentences whose tool result is "match".'
        ),
    )


register(Technique(name="simulated_tool_augmented", build=build))
