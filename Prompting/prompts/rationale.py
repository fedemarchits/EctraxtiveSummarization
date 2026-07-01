"""Cached one-shot reasoning traces.

For the 4 reasoning techniques the one-shot example must SHOW reasoning, not
just the answer. To keep it honest we do not fabricate the trace: a single
strong reference model generates it once, we verify its answer matches gold,
and cache it. All models then reuse the same exemplar+trace (comparable across
the grid). Techniques without a reasoning demo use answer-only examples.

File layout: data/shots/rationales/<technique>__<aspect>.json
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Techniques whose one-shot value is the reasoning itself.
REASONING_TECHNIQUES = {
    "chain_of_thought",
    "self_ask",
    "scoring_based",
    "salience_inference",
}

_PKG_ROOT = Path(__file__).resolve().parents[1]  # Prompting/
DEFAULT_RATIONALE_DIR = _PKG_ROOT / "data" / "shots" / "rationales"


@dataclass
class RationaleShot:
    technique: str
    aspect: str
    source_model: str          # the reference model that produced the trace
    exemplar_sentences: List[str]
    gold_indices: List[int]
    rationale: str             # the reference model's real, gold-verified trace


_cache: Dict[Tuple[str, str, str], Optional[RationaleShot]] = {}


def _path(technique: str, aspect: str, base_dir: Path) -> Path:
    return base_dir / f"{technique}__{aspect}.json"


def load_rationale(
    technique: str, aspect: str, base_dir: Optional[Path] = None
) -> Optional[RationaleShot]:
    base = Path(base_dir) if base_dir is not None else DEFAULT_RATIONALE_DIR
    key = (str(base), technique, aspect)
    if key in _cache:
        return _cache[key]
    p = _path(technique, aspect, base)
    shot = RationaleShot(**json.loads(p.read_text())) if p.exists() else None
    _cache[key] = shot
    return shot


def save_rationale(shot: RationaleShot, base_dir: Optional[Path] = None) -> Path:
    base = Path(base_dir) if base_dir is not None else DEFAULT_RATIONALE_DIR
    base.mkdir(parents=True, exist_ok=True)
    p = _path(shot.technique, shot.aspect, base)
    p.write_text(json.dumps(asdict(shot), ensure_ascii=False, indent=2))
    _cache.pop((str(base), shot.technique, shot.aspect), None)
    return p
