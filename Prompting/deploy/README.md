# deploy/

Server run for `Prompting/`. Mirrors the old `track_a/deploy` (SLURM + Docker,
RTX 3090). The whole `Prompting/` folder is mounted at `/workspace`; results
land in `Prompting/results/` (persisted through the mount).

## Files
- `Dockerfile.gpu` — torch 2.5.1 / CUDA 12.1 base + `requirements.txt`
- `run.sh` — sets env, optional trace build, `python run.py "$@"` (inside container)
- `run_docker.sh` — docker wrapper; mounts the project, passes tokens/keys
- `sbatch_script.sh` — SLURM submit, one model per job

## Build the image (once, on the server)
```bash
cd Prompting
docker build -f deploy/Dockerfile.gpu -t prompting-gpu:latest .
```

## Run
```bash
# smallest local model, and cache reasoning traces on this first run
BUILD_RATIONALES=1 HF_TOKEN=hf_xxx MODEL=gemma4_e2b sbatch deploy/sbatch_script.sh

# more local models (resume-safe; submit in parallel)
HF_TOKEN=hf_xxx MODEL=qwen35_4b  sbatch deploy/sbatch_script.sh
HF_TOKEN=hf_xxx MODEL=gemma4_12b sbatch deploy/sbatch_script.sh

# endpoint tiers (27B / 31B): no GPU download, needs the API env
OPENAI_BASE_URL=... OPENAI_API_KEY=... MODEL=qwen35_27b sbatch deploy/sbatch_script.sh
```

## Aggregate after
```bash
python -m engine.report --results results --out results/summary.csv
```

## Notes
- One model per job; each writes `results/<model>/<variant>.jsonl` and skips
  variants already done.
- `BUILD_RATIONALES=1` runs once per project (traces are model-agnostic — they
  come from the reference model in `configs/experiment.yaml`). Set it on the
  first job only.
- Medium tier in bf16 (`gemma4_12b`, `qwen35_9b`) is tight on a single 24 GB
  3090. If it OOMs, add a second GPU (`--gres=gpu:...:2`, the backend uses
  `device_map=auto`) or switch that model to 4-bit / an endpoint.
