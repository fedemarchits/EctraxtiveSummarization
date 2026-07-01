"""Small constants shared across the engine. Experiment-level knobs live in
configs/*.yaml and are loaded by the runner; only the truly fixed aspect set
lives here."""
from __future__ import annotations

ASPECTS = ("challenge", "approach", "outcome")
