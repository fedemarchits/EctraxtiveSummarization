"""ACLSum loading, gold labels, text reconstruction, and TRAIN-derived caps.

Splits: eval on `test`; exemplars and the cap K come from `train` (never test).
A doc exposes: doc["id"], doc["source_sentences"], doc[f"{aspect}_labels"].
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

_EXTRACTIVE: Dict[str, object] = {}
_ABS_BY_ID: Optional[Dict[str, Dict[str, str]]] = None


def load_split(name: str):
    """Return the ACLSum extractive split ('train' | 'validation' | 'test').

    HF split name is 'validation'; we accept 'val' as an alias.
    Cached so repeated calls in a run don't reload.
    """
    hf_name = "validation" if name in ("val", "validation") else name
    if hf_name not in _EXTRACTIVE:
        from datasets import load_dataset
        _EXTRACTIVE[hf_name] = load_dataset("sobamchan/aclsum", "extractive")[hf_name]
    return _EXTRACTIVE[hf_name]


def load_abstractive_refs() -> Dict[str, Dict[str, str]]:
    """Abstractive references for the test split, keyed by doc id -> {aspect: text}."""
    global _ABS_BY_ID
    if _ABS_BY_ID is None:
        from datasets import load_dataset
        from .config import ASPECTS
        abstractive = load_dataset("sobamchan/aclsum", "abstractive")["test"]
        _ABS_BY_ID = {ex["id"]: {a: ex[a] for a in ASPECTS} for ex in abstractive}
    return _ABS_BY_ID


def labels_to_indices(labels: Sequence) -> List[int]:
    return [i + 1 for i, v in enumerate(labels) if int(v) == 1]


def gold_for_doc(doc, aspects: Sequence[str]) -> Dict[str, List[int]]:
    return {a: labels_to_indices(doc[f"{a}_labels"]) for a in aspects}


def indices_to_text(sentences: Sequence[str], idxs_1b: Sequence[int]) -> str:
    """Reconstruct text from 1-based sentence indices (out-of-range skipped)."""
    return " ".join(sentences[i - 1] for i in idxs_1b if 1 <= i <= len(sentences))


def aspect_caps_from_gold(
    docs, aspects: Sequence[str], min_cap: int = 1
) -> Dict[str, int]:
    """K per aspect = median gold count over `docs` (the TRAIN split). No grid search."""
    import numpy as np
    counts: Dict[str, List[int]] = {a: [] for a in aspects}
    for doc in docs:
        for a in aspects:
            counts[a].append(len(labels_to_indices(doc[f"{a}_labels"])))
    return {a: max(min_cap, int(np.median(counts[a]) or 0)) for a in aspects}
