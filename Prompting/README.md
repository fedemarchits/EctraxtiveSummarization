# Prompting

Clean rebuild of the LLM extractive-summarization prompt experiments for the
expanded plan: **every technique in all 4 variants** (zero/one-shot x
capped/uncapped) across **Qwen3.5 + Gemma 4**, four matched size tiers each.

## Layout

| path                  | what                                                                       |
| --------------------- | -------------------------------------------------------------------------- |
| `prompts/`            | the core: techniques as a registry; variants generated in `base.py`        |
| `prompts/techniques/` | one file per technique (`vanilla.py` is the template)                      |
| `engine/`             | thin runtime (data, backends, metrics, runner) — STUBS, port behind tests  |
| `configs/`            | `models.yaml`, `experiment.yaml`, `grid.yaml` — the grid is data, not code |
| `data/shots/`         | few-shot exemplars, drawn from TRAIN only                                  |
| `results/`            | per-run output `results/<model>/<variant_slug>.jsonl`                      |
| `tests/`              | trust is earned here                                                       |
| `run.py`              | CLI                                                                        |

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

## Results so far

Aggregated from `results/summary.csv` (one row per model × variant × aspect).
Metrics: per-sentence **F1 / P / R** on selected indices, **ROUGE-L** of the built
summary, **oracle gap** (ROUGE-L below the best achievable selection; lower is
better). `6 / 8` grid models are done; the two largest are endpoint-only and not
yet run.

### Where each model was computed

| Model        | GPU                       | Backend         | Precision | Status        |
| ------------ | ------------------------- | --------------- | --------- | ------------- |
| `qwen35_2b`  | RTX 3090 (faretra)        | HF transformers | bf16      | ✅ done       |
| `qwen35_4b`  | RTX 3090 (faretra)        | HF transformers | bf16      | ✅ done       |
| `qwen35_9b`  | A100 40GB (vast.ai)       | **vLLM**        | bf16      | ✅ done       |
| `gemma4_e2b` | RTX 3090 (faretra)        | HF transformers | bf16      | ✅ done       |
| `gemma4_e4b` | A100 40GB (vast.ai)       | **vLLM**        | bf16      | ✅ done       |
| `gemma4_12b` | A100 40GB (vast.ai)       | **vLLM**        | bf16      | ✅ done       |
| `qwen35_27b` | — (endpoint / OpenRouter) | endpoint        | full      | ❌ to compute |
| `gemma4_31b` | — (endpoint / OpenRouter) | endpoint        | full      | ❌ to compute |

gemma-4 is too new for the CUDA-12 cluster's vLLM, so `9b / e4b / 12b` were run
on a rented A100 (CUDA-13 → modern vLLM). All six are bf16, so quality is
comparable; **latency is not** (vLLM batches, HF does not).

### Prompt variants computed (per model)

Each technique × **shot** (`zero`/`one`) × **cap** (`capped`/`uncapped`) — the
4-cell grid, except `negative_aware` and the `_trace` ablations (one-shot only).
**54 variants per model.**

| Technique                | zero·cap | zero·unc | one·cap | one·unc |
| ------------------------ | :------: | :------: | :-----: | :-----: |
| vanilla                  |    ✅    |    ✅    |   ✅    |   ✅    |
| chain_of_thought         |    ✅    |    ✅    |   ✅    |   ✅    |
| least_to_most            |    ✅    |    ✅    |   ✅    |   ✅    |
| self_ask                 |    ✅    |    ✅    |   ✅    |   ✅    |
| explanation_based        |    ✅    |    ✅    |   ✅    |   ✅    |
| salience_inference       |    ✅    |    ✅    |   ✅    |   ✅    |
| scoring_based            |    ✅    |    ✅    |   ✅    |   ✅    |
| tool_augmented           |    ✅    |    ✅    |   ✅    |   ✅    |
| simulated_tool_augmented |    ✅    |    ✅    |   ✅    |   ✅    |
| self_critique            |    ✅    |    ✅    |   ✅    |   ✅    |
| contrastive_joint        |    ✅    |    ✅    |   ✅    |   ✅    |
| negative_aware           |    —     |    —     |   ✅    |   ✅    |
| chain_of_thought_trace   |    —     |    —     |   ✅    |   ✅    |
| salience_inference_trace |    —     |    —     |   ✅    |   ✅    |
| scoring_based_trace      |    —     |    —     |   ✅    |   ✅    |
| self_ask_trace           |    —     |    —     |   ✅    |   ✅    |

### Results by model

Mean over all 54 variants × 3 aspects. `latency` = mean s/doc (backend-dependent,
not a cross-model comparison).

| Model        |  F1   | Precision | Recall | ROUGE-L | Oracle gap | Latency (s) |
| ------------ | :---: | :-------: | :----: | :-----: | :--------: | :---------: |
| `qwen35_2b`  | 0.461 |   0.438   | 0.674  |  0.137  |   0.263    |    4.48     |
| `qwen35_4b`  | 0.609 |   0.626   | 0.661  |  0.176  |   0.225    |    3.34     |
| `qwen35_9b`  | 0.623 |   0.641   | 0.678  |  0.177  |   0.223    |    0.84     |
| `gemma4_e2b` | 0.567 |   0.626   | 0.570  |  0.185  |   0.216    |    3.51     |
| `gemma4_e4b` | 0.611 |   0.647   | 0.639  |  0.179  |   0.222    |    0.43     |
| `gemma4_12b` | 0.631 |   0.642   | 0.681  |  0.179  |   0.222    |    1.18     |
| `qwen35_27b` |   ✗   |     ✗     |   ✗    |    ✗    |     ✗      |      ✗      |
| `gemma4_31b` |   ✗   |     ✗     |   ✗    |    ✗    |     ✗      |      ✗      |

### Comparison 1 — capped vs. uncapped (per model)

| Model        | F1 capped | F1 uncapped | ROUGE-L capped | ROUGE-L uncapped |
| ------------ | :-------: | :---------: | :------------: | :--------------: |
| `qwen35_2b`  |   0.486   |    0.437    |   **0.178**    |      0.097       |
| `qwen35_4b`  |   0.602   |    0.616    |   **0.191**    |      0.160       |
| `qwen35_9b`  |   0.620   |    0.627    |   **0.194**    |      0.161       |
| `gemma4_e2b` |   0.558   |    0.576    |   **0.192**    |      0.177       |
| `gemma4_e4b` |   0.603   |    0.619    |   **0.187**    |      0.170       |
| `gemma4_12b` |   0.623   |    0.638    |   **0.192**    |      0.166       |

F1 is roughly flat (uncapped slightly higher except the weak 2B), but **ROUGE-L
is consistently better capped** — a fixed sentence budget stops over-selection
from diluting the summary.

### Comparison 2 — zero-shot vs. one-shot (per model, F1)

| Model        | zero-shot | one-shot |   Δ    |
| ------------ | :-------: | :------: | :----: |
| `qwen35_2b`  |   0.483   |  0.446   | −0.037 |
| `qwen35_4b`  |   0.605   |  0.612   | +0.007 |
| `qwen35_9b`  |   0.615   |  0.629   | +0.014 |
| `gemma4_e2b` |   0.589   |  0.552   | −0.037 |
| `gemma4_e4b` |   0.589   |  0.626   | +0.037 |
| `gemma4_12b` |   0.625   |  0.635   | +0.010 |

The exemplar **hurts the smallest models** (`2b`, `e2b`) and **helps as size
grows** — capacity to use a worked example appears with scale.

### Comparison 3 — one-shot: answer-only vs. rationale (trace)

For the four techniques with a `_trace` variant (`chain_of_thought`,
`salience_inference`, `scoring_based`, `self_ask`): one-shot exemplar as a plain
answer vs. the 397B reasoning trace.

| Model        | answer-only F1 | trace F1 |     Δ      |
| ------------ | :------------: | :------: | :--------: |
| `qwen35_2b`  |     0.450      |  0.450   |   −0.000   |
| `qwen35_4b`  |     0.608      |  0.617   |   +0.008   |
| `qwen35_9b`  |     0.629      |  0.630   |   +0.001   |
| `gemma4_e2b` |     0.542      |  0.571   | **+0.029** |
| `gemma4_e4b` |     0.630      |  0.638   |   +0.008   |
| `gemma4_12b` |     0.635      |  0.632   |   −0.004   |

Rationale exemplars help most models slightly (largest gain on `gemma4_e2b`), but
the effect is small and not universal — no gain on the weakest (`2b`) or the
largest (`12b`).

### Still to compute

`qwen35_27b` and `gemma4_31b` — the large tier, endpoint-only (need a hosted
gemma-4 / qwen-3.5 slug + API credit). Everything else is done.

### Top 5 prompts (mean F1 over all 6 models)

| Rank | Technique                  |  F1   | Precision | Recall | ROUGE-L |
| :--: | -------------------------- | :---: | :-------: | :----: | :-----: |
|  1   | `self_ask_trace`           | 0.596 |   0.622   | 0.657  |  0.176  |
|  2   | `negative_aware`           | 0.593 |   0.626   | 0.639  |  0.178  |
|  3   | `salience_inference_trace` | 0.591 |   0.636   | 0.625  |  0.181  |
|  4   | `salience_inference`       | 0.591 |   0.631   | 0.629  |  0.180  |
|  5   | `self_critique`            | 0.590 |   0.601   | 0.662  |  0.171  |

Winners are the **reasoning-trace** and **precision-oriented** techniques (note
the high precision, 0.62–0.64, they probably win by cutting false positives?).

For reference:: the baseline `vanilla` is last at **0.558** and `tool_augmented`
second-last at 0.571.
