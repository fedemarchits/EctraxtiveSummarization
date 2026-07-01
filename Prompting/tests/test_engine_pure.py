"""Pure-logic engine tests — no models, no dataset, no network.
Covers JSON parsing, index cleaning/capping, grid resolution, exemplar picking.
"""
from prompts.base import Cap, Shot

from engine.postprocess import clean_indices, cap_indices_in_order, parse_selection, safe_extract_json
from engine.grid import resolve_variants
from prompts.fewshot import select_exemplar, iter_exemplars, gold_indices_1based


# ---- postprocess ---------------------------------------------------------

def test_json_direct_and_fenced():
    assert safe_extract_json('{"selected_sentences": [1,2]}')["selected_sentences"] == [1, 2]
    fenced = "```json\n{\"selected_sentences\": [3]}\n```"
    assert safe_extract_json(fenced)["selected_sentences"] == [3]


def test_json_last_object_and_bare_array():
    txt = "noise {\"x\":1} then {\"selected_sentences\": [2, 4]}"
    assert safe_extract_json(txt)["selected_sentences"] == [2, 4]
    assert safe_extract_json("here:\n[1, 5]\n")["selected_sentences"] == [1, 5]


def test_clean_and_cap():
    assert clean_indices([2, 2, 1, 9, 0, "x"], n_sentences=5) == [1, 2]
    assert cap_indices_in_order([1, 2, 3, 4], 2) == [1, 2]
    assert cap_indices_in_order([1, 2, 3], None) == [1, 2, 3]


def test_parse_selection_end_to_end():
    out = 'The answer is {"selected_sentences": [1, 3, 7, 3]}'
    assert parse_selection(out, n_sentences=5, cap=None) == [1, 3]
    assert parse_selection(out, n_sentences=5, cap=1) == [1]


def test_parse_selection_garbage_is_empty():
    assert parse_selection("no json here", n_sentences=4) == []


# ---- grid ----------------------------------------------------------------

def test_grid_resolves_46_variants(tmp_path):
    # empty grid file -> every technique defaults to its full valid grid
    g = tmp_path / "grid.yaml"
    g.write_text("techniques: {}\n")
    variants = resolve_variants(str(g))
    assert len(variants) == 46
    # negative_aware must stay one-shot only even under default expansion
    na = [v for v in variants if v.technique == "negative_aware"]
    assert na and all(v.shot is Shot.ONE for v in na)


def test_grid_restriction_is_intersected(tmp_path):
    # ask for a zero-shot negative_aware cell that isn't valid -> dropped
    g = tmp_path / "grid.yaml"
    g.write_text(
        "techniques:\n"
        "  negative_aware:\n"
        "    - {shot: zero_shot, cap: uncapped}\n"
        "    - {shot: one_shot, cap: capped}\n"
    )
    na = [v for v in resolve_variants(str(g)) if v.technique == "negative_aware"]
    assert {(v.shot, v.cap) for v in na} == {(Shot.ONE, Cap.CAPPED)}


# ---- fewshot -------------------------------------------------------------

_DOCS = [
    {"source_sentences": ["a", "b", "c"], "challenge_labels": [0, 0, 0]},
    {"source_sentences": ["d", "e"],      "challenge_labels": [1, 0]},
]


def test_select_exemplar_prefers_doc_with_gold():
    ex = select_exemplar(_DOCS, "challenge", seed=42)
    assert list(ex.gold_indices) == [1]  # from the 2nd doc
    assert list(ex.sentences) == ["d", "e"]


def test_iter_exemplars_only_gold_and_deterministic():
    a = [list(e.gold_indices) for e in iter_exemplars(_DOCS, "challenge", seed=1)]
    b = [list(e.gold_indices) for e in iter_exemplars(_DOCS, "challenge", seed=1)]
    assert a == b and all(g for g in a)  # deterministic; every yielded ex has gold


def test_gold_indices_1based():
    assert gold_indices_1based({"challenge_labels": [0, 1, 1]}, "challenge") == [2, 3]
