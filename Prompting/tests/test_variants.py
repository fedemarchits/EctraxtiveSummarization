"""Seed test: variant expansion + registry integrity.

Guards the core invariant of this work — every technique yields exactly its
declared shot x cap grid, and nothing double-registers.
"""
from prompts.base import Cap, Shot
from prompts.registry import all_techniques, all_variants


def test_full_grid_technique_has_four_variants():
    from prompts import expand
    techs = all_techniques()
    v = expand(techs["vanilla"])
    assert len(v) == 4
    combos = {(x.shot, x.cap) for x in v}
    assert combos == {
        (Shot.ZERO, Cap.UNCAPPED), (Shot.ZERO, Cap.CAPPED),
        (Shot.ONE, Cap.UNCAPPED), (Shot.ONE, Cap.CAPPED),
    }


def test_variant_slugs_unique():
    slugs = [v.slug for v in all_variants()]
    assert len(slugs) == len(set(slugs))
