# engine/

Thin core that drives the grid. Ported from the old
`paper/track_a/aclsum_pipeline/`, behavior-preserving, with tests in `tests/`.

| module          | responsibility                                                   | status |
|-----------------|------------------------------------------------------------------|--------|
| `config.py`     | fixed aspect set                                                 | done   |
| `data.py`       | load ACLSum, gold labels, text reconstruction, TRAIN-median K    | done   |
| `postprocess.py`| robust JSON parse, clean indices, cap                            | done, tested |
| `metrics.py`    | P/R/F1, ROUGE-1/2/L, greedy oracle ROUGE-L + gap                 | done   |
| `backends.py`   | local HF vs OpenAI-compatible endpoint; `.generate_batch()`      | done   |
| `grid.py`       | grid.yaml + registry -> active Variants (intersected w/ valid)   | done, tested |
| `runner.py`     | model x variant x doc x aspect -> per-doc JSONL + latency meta   | done   |
| `report.py`     | aggregate JSONL -> summary table (pandas)                        | done   |

Runs on the server:

```bash
python run.py --list                 # offline: inspect the 46-variant grid
python -m scripts.build_rationales   # once: cache reference reasoning traces
python run.py --model qwen35_4b      # run one model (resume-safe)
python -m engine.report --results results --out results/summary.csv
```

Endpoint models read `OPENAI_BASE_URL` / `OPENAI_API_KEY` from the environment.
Local models need a GPU (bf16). Pure-logic tests (`tests/test_engine_pure.py`)
need no models or network.
