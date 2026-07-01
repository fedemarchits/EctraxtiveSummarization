"""Build one-shot Exemplars from TRAIN docs.

Pure and deterministic: given train docs it selects the exemplar(s) for an
aspect. No data loading or model calls here — engine.data supplies the docs.

A "doc" is any mapping with:
    doc["source_sentences"]      -> list[str]
    doc[f"{aspect}_labels"]      -> list[int]  (1 = gold selected, 0 = not)

Used by:
  - the runner, for answer-only one-shot techniques (select one exemplar),
  - scripts/build_rationales, to try exemplars until one's gold is reproduced.
"""
from __future__ import annotations

import random
from typing import Iterable, Iterator, List, Mapping, Optional, Sequence

from .base import Exemplar


def gold_indices_1based(doc: Mapping, aspect: str) -> List[int]:
    labels = doc[f"{aspect}_labels"]
    return [i + 1 for i, v in enumerate(labels) if int(v) == 1]


def has_gold(doc: Mapping, aspect: str) -> bool:
    return any(int(v) == 1 for v in doc[f"{aspect}_labels"])


def to_exemplar(doc: Mapping, aspect: str) -> Exemplar:
    return Exemplar(
        sentences=list(doc["source_sentences"]),
        gold_indices=gold_indices_1based(doc, aspect),
    )


def _shuffled(docs: Sequence[Mapping], seed: int) -> List[Mapping]:
    order = list(range(len(docs)))
    random.Random(seed).shuffle(order)
    return [docs[i] for i in order]


def select_exemplar(
    docs: Sequence[Mapping], aspect: str, seed: int = 42
) -> Optional[Exemplar]:
    """Deterministic single exemplar: first shuffled doc with gold for aspect;
    falls back to the first doc; None if docs is empty."""
    if not docs:
        return None
    for doc in _shuffled(docs, seed):
        if has_gold(doc, aspect):
            return to_exemplar(doc, aspect)
    return to_exemplar(docs[0], aspect)


def iter_exemplars(
    docs: Sequence[Mapping], aspect: str, seed: int = 42, limit: Optional[int] = None
) -> Iterator[Exemplar]:
    """Yield exemplars with gold for aspect, deterministic order, up to limit.
    Lets the rationale builder try several until the model reproduces gold."""
    n = 0
    for doc in _shuffled(docs, seed):
        if not has_gold(doc, aspect):
            continue
        yield to_exemplar(doc, aspect)
        n += 1
        if limit is not None and n >= limit:
            return
