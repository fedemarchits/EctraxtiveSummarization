"""Aggregate per-doc JSONL results into a tidy summary table.

Reads results/<model>/<variant_slug>.jsonl (+ .meta.json for latency) and emits
one row per (model, variant, aspect) with mean P/R/F1 and ROUGE. Pure stdlib +
pandas; no model deps.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

_METRICS = [
    "precision", "recall", "f1",
    "rouge1_model", "rouge2_model", "rougeL_model",
    "rouge1_oracle", "rouge2_oracle", "rougeL_oracle", "oracle_gap_rougeL",
]


def _read_jsonl(path: Path) -> List[Dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def summarize(results_dir: str = "results") -> "pandas.DataFrame":  # noqa: F821
    import pandas as pd

    root = Path(results_dir)
    rows: List[Dict] = []
    for jsonl in sorted(root.glob("*/*.jsonl")):
        model = jsonl.parent.name
        slug = jsonl.stem
        technique, shot, cap = slug.split("__")
        recs = _read_jsonl(jsonl)
        if not recs:
            continue
        meta_path = jsonl.with_suffix(".meta.json")
        latency = json.loads(meta_path.read_text())["mean_latency_s"] if meta_path.exists() else None

        df = pd.DataFrame(recs)
        for aspect, grp in df.groupby("aspect"):
            row = {
                "model": model, "technique": technique, "shot": shot, "cap": cap,
                "aspect": aspect, "n": len(grp), "mean_latency_s": latency,
            }
            row.update({k: float(grp[k].mean()) for k in _METRICS if k in grp})
            rows.append(row)

    df = pd.DataFrame(rows)
    return df.sort_values(["model", "technique", "shot", "cap", "aspect"]).reset_index(drop=True)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/summary.csv")
    args = ap.parse_args()
    df = summarize(args.results)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
