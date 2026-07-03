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

## Results so far

Aggregated from `results/summary.csv` (one row per model × variant × aspect).
Metrics: per-sentence **F1/P/R** on selected indices, **ROUGE-L** of the built
summary, **oracle gap** (ROUGE-L below the best achievable selection; lower is
better). `union` = combined all-aspects measure. Latency = mean s/doc.

Only three models have the **full 54-variant** sweep (`qwen35_2b`, `qwen35_4b`,
`gemma4_e2b`); the rest are partial, so their means are over a subset of variants
and are provisional. Technique/axis tables are computed on the three complete
models only.

### Completion

| Model | Variants | Sweep |
|-------|:---:|:---:|
| `qwen35_2b` | 54 / 54 | full |
| `qwen35_4b` | 54 / 54 | full |
| `gemma4_e2b` | 54 / 54 | full |
| `qwen35_9b` | ~48 / 54 | partial |
| `gemma4_e4b` | ~24 / 54 | partial |
| `gemma4_12b` | ~7 / 54 | partial |
| `qwen35_27b`, `gemma4_31b` | 0 / 54 | not run |

### Per-model quality

| Model | Sweep | F1 | Precision | Recall | ROUGE-L | Oracle gap | Latency (s) |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `qwen35_4b` | full | 0.609 | 0.626 | 0.661 | 0.176 | 0.225 | 3.34 |
| `gemma4_e2b` | full | 0.567 | 0.626 | 0.570 | 0.185 | 0.216 | 3.51 |
| `qwen35_2b` | full | 0.461 | 0.438 | 0.674 | 0.137 | 0.263 | 4.48 |
| `qwen35_9b` | partial | 0.624 | 0.642 | 0.678 | 0.178 | 0.223 | 4.20 |
| `gemma4_e4b` | partial | 0.610 | 0.638 | 0.648 | 0.176 | 0.224 | 7.96 |
| `gemma4_12b` | partial | 0.616 | 0.607 | 0.696 | 0.169 | 0.232 | 47.08 |

### Technique ranking — F1 (complete models)

| Rank | Technique | F1 | Precision | Recall | ROUGE-L |
|:---:|-----------|:---:|:---:|:---:|:---:|
| 1 | `self_ask_trace` | 0.558 | 0.584 | 0.641 | 0.171 |
| 2 | `least_to_most` | 0.555 | 0.565 | 0.651 | 0.165 |
| 3 | `negative_aware` | 0.555 | 0.582 | 0.626 | 0.172 |
| 4 | `self_ask` | 0.554 | 0.561 | 0.659 | 0.165 |
| 5 | `salience_inference` | 0.551 | 0.587 | 0.610 | 0.172 |
| 6 | `contrastive_joint` | 0.550 | 0.575 | 0.642 | 0.170 |
| 7 | `salience_inference_trace` | 0.549 | 0.591 | 0.604 | 0.174 |
| 8 | `self_critique` | 0.549 | 0.555 | 0.645 | 0.163 |
| 9 | `explanation_based` | 0.545 | 0.561 | 0.631 | 0.165 |
| 10 | `chain_of_thought` | 0.544 | 0.559 | 0.646 | 0.166 |
| 11 | `scoring_based` | 0.543 | 0.557 | 0.644 | 0.164 |
| 12 | `simulated_tool_augmented` | 0.542 | 0.558 | 0.639 | 0.166 |
| 13 | `chain_of_thought_trace` | 0.538 | 0.549 | 0.652 | 0.164 |
| 14 | `scoring_based_trace` | 0.538 | 0.559 | 0.633 | 0.167 |
| 15 | `vanilla` | 0.536 | 0.549 | 0.623 | 0.161 |
| 16 | `tool_augmented` | 0.530 | 0.545 | 0.604 | 0.160 |

### Axis effects (complete models)

| Contrast | F1 | ROUGE-L |
|----------|:---:|:---:|
| `zero_shot` | 0.559 | 0.164 |
| `one_shot` | 0.537 | 0.167 |
| `capped` | 0.549 | 0.187 |
| `uncapped` | 0.543 | 0.145 |

### Trace vs. answer-only exemplar (one-shot pairs)

| Base technique | Answer-only F1 | Trace F1 | Δ |
|----------------|:---:|:---:|:---:|
| `self_ask` | 0.541 | 0.558 | +0.016 |
| `salience_inference` | 0.535 | 0.549 | +0.014 |
| `scoring_based` | 0.527 | 0.538 | +0.011 |
| `chain_of_thought` | 0.530 | 0.538 | +0.008 |

### By aspect (complete models)

| Aspect | F1 | Precision | Recall |
|--------|:---:|:---:|:---:|
| `union` (all) | 0.614 | 0.633 | 0.669 |
| `outcome` | 0.604 | 0.641 | 0.670 |
| `approach` | 0.546 | 0.544 | 0.673 |
| `challenge` | 0.418 | 0.435 | 0.528 |
