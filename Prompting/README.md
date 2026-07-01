# Prompting

Clean rebuild of the LLM extractive-summarization prompt experiments for the
expanded plan: **every technique in all 4 variants** (zero/one-shot x
capped/uncapped) across **Qwen3.5 + Gemma 4**, four matched size tiers each.

Started fresh on purpose. The old `paper/track_a/aclsum_pipeline/` is treated as
a reference to port from — behind tests — not as trusted code.

## Layout

| path                 | what                                                            |
|----------------------|----------------------------------------------------------------|
| `prompts/`           | the core: techniques as a registry; variants generated in `base.py` |
| `prompts/techniques/`| one file per technique (`vanilla.py` is the template)          |
| `engine/`            | thin runtime (data, backends, metrics, runner) — STUBS, port behind tests |
| `configs/`           | `models.yaml`, `experiment.yaml`, `grid.yaml` — the grid is data, not code |
| `data/shots/`        | few-shot exemplars, drawn from TRAIN only                       |
| `results/`           | per-run output `results/<model>/<variant_slug>.jsonl`          |
| `tests/`             | trust is earned here                                            |
| `run.py`             | CLI                                                             |

## Principles

1. **Prompts are the product** → isolated, one per file, no duplicated variant plumbing.
2. **Grid is config** → add a model or toggle a cell by editing YAML.
3. **Engine is thin and tested** → port from the old pipeline one module at a time.

## Techniques

9 existing: vanilla, least_to_most, tool_augmented, simulated_tool_augmented,
scoring_based, self_ask, chain_of_thought, explanation_based, salience_inference.

3 new: `self_critique` (one-shot form = self_refinement),
`contrastive_joint` (one-shot form = joint_self_ask),
`negative_aware` (one-shot only).

Orthogonal wrappers (not variants): `self_consistency` (majority vote),
`dynamic_llm_capper` (LLM prunes its own list instead of a fixed cap).

## Status

- [x] prompt registry + variant expansion (`prompts/base.py`, `registry.py`)
- [x] all 12 techniques (9 existing + 3 new), 46 variants
- [x] one-shot examples: answer-only + cached real reasoning traces (`rationale.py`)
- [x] few-shot exemplar selector (`fewshot.py`, TRAIN-only)
- [x] engine ported: data, postprocess, metrics, backends, grid, runner, report
- [x] TRAIN-median K capping
- [x] orthogonal wrappers: `self_consistency`, `dynamic_llm_capper` (config-toggled)
- [x] HF model ids verified on the Hub (`configs/models.yaml`)
- [ ] endpoint access for large tiers (env: OPENAI_BASE_URL / OPENAI_API_KEY) — set on server
- [ ] server run (GPU) + full pytest with deps installed

## Run (on the server)

```bash
pip install -r requirements.txt
python run.py --list                 # offline: inspect the 46-variant grid
python -m scripts.build_rationales   # once: cache reference reasoning traces
python run.py --model qwen35_4b      # run one model (resume-safe)
python -m engine.report --out results/summary.csv
pytest -q                            # tests (pure-logic subset needs no models)
```
