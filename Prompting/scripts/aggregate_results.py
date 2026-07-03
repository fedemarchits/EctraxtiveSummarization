"""Aggregate per-doc JSONL results into results/summary.csv — stdlib only.

Mirror of engine/report.py but with no pandas dependency, so it runs on a bare
python3 (e.g. the cluster host). One row per (model, variant, aspect) with mean
P/R/F1, ROUGE, oracle gap, and latency.

Usage:  python3 scripts/aggregate_results.py [results_dir] [out_csv]
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys
from collections import defaultdict

METRICS = [
    "precision", "recall", "f1",
    "rouge1_model", "rouge2_model", "rougeL_model",
    "rouge1_oracle", "rouge2_oracle", "rougeL_oracle", "oracle_gap_rougeL",
]


def main() -> None:
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    out_csv = sys.argv[2] if len(sys.argv) > 2 else os.path.join(results_dir, "summary.csv")

    rows = []
    for jf in sorted(glob.glob(os.path.join(results_dir, "*", "*.jsonl"))):
        model = os.path.basename(os.path.dirname(jf))
        slug = os.path.basename(jf)[:-6]  # strip ".jsonl"
        try:
            technique, shot, cap = slug.split("__")
        except ValueError:
            continue
        with open(jf, encoding="utf-8") as fh:
            recs = [json.loads(line) for line in fh if line.strip()]
        if not recs:
            continue
        meta = jf[:-6] + ".meta.json"
        latency = json.load(open(meta))["mean_latency_s"] if os.path.exists(meta) else ""

        by_aspect = defaultdict(list)
        for r in recs:
            by_aspect[r.get("aspect", "")].append(r)

        for aspect, grp in by_aspect.items():
            row = {
                "model": model, "technique": technique, "shot": shot, "cap": cap,
                "aspect": aspect, "n": len(grp), "mean_latency_s": latency,
            }
            for k in METRICS:
                vals = [g[k] for g in grp if k in g and g[k] is not None]
                row[k] = sum(vals) / len(vals) if vals else ""
            rows.append(row)

    rows.sort(key=lambda r: (r["model"], r["technique"], r["shot"], r["cap"], r["aspect"]))
    cols = ["model", "technique", "shot", "cap", "aspect", "n", "mean_latency_s"] + METRICS
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
