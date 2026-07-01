"""CLI entry point.

    python run.py --list                 # print the resolved technique x variant grid
    python run.py --model qwen35_4b      # run all active variants for one model

Config is read from configs/*.yaml. Runs are resume-safe (a variant whose
JSONL already exists is skipped). Model execution happens on the server; --list
is offline and needs no model deps.
"""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="print all variants and exit")
    ap.add_argument("--model", help="model alias from configs/models.yaml to run")
    ap.add_argument("--experiment", default="configs/experiment.yaml")
    ap.add_argument("--models", default="configs/models.yaml")
    ap.add_argument("--grid", default="configs/grid.yaml")
    args = ap.parse_args()

    if args.model:
        from engine.runner import run
        run(args.model, args.experiment, args.models, args.grid)
        return

    # default / --list: offline grid inspection
    from engine.grid import resolve_variants
    variants = resolve_variants(args.grid)
    for v in sorted(variants, key=lambda x: x.slug):
        print(v.slug)
    print(f"\n{len(variants)} variants across "
          f"{len({v.technique for v in variants})} techniques")


if __name__ == "__main__":
    main()
