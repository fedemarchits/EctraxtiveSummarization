"""Drive the grid: model x variant x doc x aspect -> per-doc JSONL + summary.

Resume-safe: a variant whose JSONL already exists is skipped. One JSON line per
(doc, aspect) plus a per-doc 'union' line, mirroring the old pipeline's rows.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from prompts.base import Cap, RenderCtx, Shot
from prompts.fewshot import select_exemplar
from prompts.registry import all_techniques

from . import data as D
from . import metrics as M
from .backends import GenConfig, get_backend
from .grid import resolve_variants
from .wrappers import select_document

SYSTEM = "You are an expert in extractive summarization."


def _ctx_for(variant, aspect, caps, train, seed) -> RenderCtx:
    k = caps.get(aspect) if variant.cap is Cap.CAPPED else None
    exemplar = select_exemplar(train, aspect, seed=seed) if variant.shot is Shot.ONE else None
    return RenderCtx(shot=variant.shot, cap=variant.cap, k=k, exemplar=exemplar)


def _row(doc_id, doc_idx, aspect, n, pred, gold, m, rm, ro, oracle_idx, raw=None) -> Dict:
    gap = (ro["rougeL"] - rm["rougeL"]) if ro else 0.0
    return {
        "doc_id": doc_id, "doc_idx": doc_idx, "aspect": aspect, "num_sentences": n,
        "pred_indices": pred, "gold_indices": sorted(gold),
        "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
        "rouge1_model": rm.get("rouge1", 0.0), "rouge2_model": rm.get("rouge2", 0.0),
        "rougeL_model": rm.get("rougeL", 0.0),
        "rouge1_oracle": ro.get("rouge1", 0.0) if ro else 0.0,
        "rouge2_oracle": ro.get("rouge2", 0.0) if ro else 0.0,
        "rougeL_oracle": ro.get("rougeL", 0.0) if ro else 0.0,
        "oracle_gap_rougeL": gap, "oracle_indices": oracle_idx,
        # raw model text (str, or list[str] under self_consistency); None for the
        # derived union row, which makes no direct model call.
        "raw_response": raw,
    }


def run_variant(variant, backend, test, train, abs_by_id, aspects, caps, seed,
                oracle_cache, out_path: Path, sc=None, dyn=None) -> None:
    tech = all_techniques()[variant.technique]
    ctxs = {a: _ctx_for(variant, a, caps, train, seed) for a in aspects}
    tmp = out_path.with_suffix(".jsonl.partial")
    latencies: List[float] = []

    with tmp.open("w", encoding="utf-8") as fh:
        for doc_idx in range(len(test)):
            doc = test[doc_idx]
            doc_id, sents = doc["id"], doc["source_sentences"]
            n = len(sents)
            gold = D.gold_for_doc(doc, aspects)
            abs_ref = abs_by_id.get(doc_id) if abs_by_id else None

            prompts = [tech.build(sents, a, ctxs[a]) for a in aspects]
            t0 = time.perf_counter()
            preds, raws = select_document(backend, prompts, aspects, sents, caps, variant, sc, dyn, SYSTEM)
            latencies.append(time.perf_counter() - t0)

            for a in aspects:
                m = M.prf(gold[a], preds[a])
                rm, ro, oracle_idx = {}, {}, []
                if abs_ref and abs_ref.get(a):
                    ref = abs_ref[a]
                    rm = M.rouge(ref, D.indices_to_text(sents, preds[a]))
                    key = (doc_id, a)
                    if key not in oracle_cache:
                        oracle_cache[key] = M.build_oracle_indices(sents, ref)
                    oracle_idx = oracle_cache[key]
                    ro = M.rouge(ref, D.indices_to_text(sents, oracle_idx))
                fh.write(json.dumps(_row(doc_id, doc_idx, a, n, preds[a], gold[a], m, rm, ro, oracle_idx, raw=raws[a])) + "\n")

            # union across aspects
            gold_u = sorted(set().union(*[set(gold[a]) for a in aspects]))
            pred_u = sorted(set().union(*[set(preds[a]) for a in aspects]))
            mU = M.prf(gold_u, pred_u)
            rmU, roU, oracle_u = {}, {}, []
            if abs_ref:
                refU = M.concat_abs_refs(abs_ref, aspects)
                if refU:
                    rmU = M.rouge(refU, D.indices_to_text(sents, pred_u))
                    key = (doc_id, "union")
                    if key not in oracle_cache:
                        oracle_cache[key] = M.build_oracle_indices(sents, refU)
                    oracle_u = oracle_cache[key]
                    roU = M.rouge(refU, D.indices_to_text(sents, oracle_u))
            fh.write(json.dumps(_row(doc_id, doc_idx, "union", n, pred_u, gold_u, mU, rmU, roU, oracle_u)) + "\n")

    tmp.replace(out_path)
    # sidecar: mean latency for this variant
    out_path.with_suffix(".meta.json").write_text(json.dumps({
        "variant": variant.slug,
        "mean_latency_s": (sum(latencies) / len(latencies)) if latencies else 0.0,
        "n_docs": len(test),
    }, indent=2))


def run(model_alias: str,
        experiment_yaml: str = "configs/experiment.yaml",
        models_yaml: str = "configs/models.yaml",
        grid_yaml: str = "configs/grid.yaml") -> None:
    cfg = yaml.safe_load(open(experiment_yaml))
    aspects = list(cfg["dataset"]["aspects"])
    seed = int(cfg["fewshot"]["seed"])
    inf = cfg.get("inference", {})
    sc = cfg.get("self_consistency", {})
    dyn = cfg.get("dynamic_capper", {})
    # self_consistency needs sampling; otherwise stay greedy/deterministic.
    temperature = float(sc.get("temperature", 0.7)) if sc.get("enabled") else float(inf.get("temperature", 0.0))
    gen = GenConfig(
        max_new_tokens=int(inf.get("max_new_tokens", 128)),
        temperature=temperature,
        seed=int(inf.get("seed", seed)),
    )
    resume = bool(cfg.get("output", {}).get("resume", True))
    results_dir = Path(cfg.get("output", {}).get("results_dir", "results"))

    test = D.load_split(cfg["dataset"].get("eval_split", "test"))
    train = D.load_split(cfg["capping"].get("source_split", "train"))
    abs_by_id = D.load_abstractive_refs()
    caps = D.aspect_caps_from_gold(train, aspects)

    backend = get_backend(model_alias, models_yaml, gen)
    variants = resolve_variants(grid_yaml)

    out_dir = results_dir / model_alias
    out_dir.mkdir(parents=True, exist_ok=True)
    oracle_cache: Dict = {}

    for v in variants:
        out_path = out_dir / f"{v.slug}.jsonl"
        if resume and out_path.exists():
            print(f"[skip] {model_alias}/{v.slug} (exists)")
            continue
        print(f"[run]  {model_alias}/{v.slug}")
        run_variant(v, backend, test, train, abs_by_id, aspects, caps, seed,
                    oracle_cache, out_path, sc=sc, dyn=dyn)
        backend.free_memory()  # release cached VRAM between variants (anti-fragmentation)
