"""Shared rendering helpers used by every technique.

One place for the header, aspect definitions, sentence numbering, cap line,
one-shot example block, and the JSON return contract — so no technique
re-implements variant plumbing and all four variants stay consistent.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

from .base import Cap, RenderCtx, Shot
from .rationale import load_rationale

ASPECT_DEFS = """- challenge: problem, gap, limitation, unmet need, difficulty/motivation.
- approach: method, model, system, algorithm, dataset design, procedure.
- outcome: results, findings, improvements, metrics, performance, impact."""

RETURN_FORMAT = '{"selected_sentences": [list_of_sentence_numbers]}'

SHARED_RULES = """- If no sentences match, return an empty list.
- Indices are 1-based.
- Return ONLY valid JSON."""

Renderer = Callable[[Sequence[str]], str]


def numbered(sentences: Sequence[str]) -> str:
    return "\n".join(f"Sentence {i+1}: {s}" for i, s in enumerate(sentences))


def header(aspect: str) -> str:
    return (
        "You are an expert in extractive summarization. Your task is to select "
        f'sentences that express the "{aspect}" aspect of the document.\n\n'
        f"Aspect definitions:\n{ASPECT_DEFS}"
    )


def cap_bullet(ctx: RenderCtx, aspect: str) -> str:
    """Cap constraint, integrated into the rules (not a tail suffix)."""
    if ctx.cap is Cap.CAPPED and ctx.k is not None:
        return f'- Select AT MOST {ctx.k} sentences for the "{aspect}" aspect.\n'
    return ""


def example_block(ctx: RenderCtx, render_fn: Optional[Renderer] = None) -> str:
    """One-shot example: real input + real gold indices, no fabricated reasoning.

    render_fn lets a technique format the exemplar input exactly like its own
    input (e.g. centrality-annotated) so the example matches the task format.
    """
    if ctx.shot is not Shot.ONE or ctx.exemplar is None:
        return ""
    ex = ctx.exemplar
    rf = render_fn or numbered
    return (
        "Example (one-shot; exemplar from the training split):\n"
        "Input:\n" + rf(ex.sentences) + "\n\n"
        f"Selected indices: {list(ex.gold_indices)}\n\n"
        "---\n\nNew task:\n"
    )


def reasoning_example_block(
    technique: str, aspect: str, ctx: RenderCtx, render_fn: Optional[Renderer] = None
) -> str:
    """One-shot block that shows a real, gold-verified reasoning trace.

    Returns "" when zero-shot or when no cached rationale exists yet — the
    caller's render() then falls back to the answer-only example from
    ctx.exemplar. The shown answer always matches the shown reasoning because
    both come from the same cached RationaleShot.
    """
    if ctx.shot is not Shot.ONE:
        return ""
    shot = load_rationale(technique, aspect)
    if shot is None:
        return ""  # graceful fallback to answer-only
    rf = render_fn or numbered
    return (
        f"Example (one-shot; exemplar and reasoning from the {shot.source_model} "
        "reference model, verified against gold):\n"
        "Input:\n" + rf(shot.exemplar_sentences) + "\n\n"
        "Reasoning:\n" + shot.rationale.strip() + "\n\n"
        f"Selected indices: {list(shot.gold_indices)}\n\n"
        "---\n\nNew task:\n"
    )


def render(
    aspect: str,
    sentences: Sequence[str],
    ctx: RenderCtx,
    instructions: str,
    preamble: str = "",
    render_fn: Optional[Renderer] = None,
    example_override: str = "",
) -> str:
    """Standard assembly shared by all techniques.

    example_override: a technique-specific one-shot block (e.g. a prune demo).
    Used in place of the generic example when non-empty.
    """
    rf = render_fn or numbered
    parts = [header(aspect), "\n\n"]
    if preamble:
        parts += [preamble.strip(), "\n\n"]
    parts += [example_override or example_block(ctx, render_fn)]
    parts += ["Input:\n", rf(sentences), "\n\n"]
    parts += ["Instructions:\n", instructions.strip(), "\n\n"]
    parts += ["Rules:\n", cap_bullet(ctx, aspect) + SHARED_RULES, "\n\n"]
    parts += ["Return format:\n", RETURN_FORMAT]
    return "".join(parts)
