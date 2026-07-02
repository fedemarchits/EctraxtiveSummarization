#!/bin/bash
# sbatch_script.sh — submit one model's full variant grid to SLURM.
#
# Usage:
#   HF_TOKEN=hf_xxx MODEL=gemma4_e2b sbatch deploy/sbatch_script.sh
#
#   # endpoint tiers (27B / 31B) also need:
#   OPENAI_BASE_URL=... OPENAI_API_KEY=... MODEL=qwen35_27b sbatch deploy/sbatch_script.sh
#
#   # first run of a model family: also cache reasoning traces
#   BUILD_RATIONALES=1 HF_TOKEN=hf_xxx MODEL=gemma4_e2b sbatch deploy/sbatch_script.sh
#
# One model per job. Submit several for the full grid (each is resume-safe).
#
# This cluster uses --gpus-per-node= (not --gres=gpu:) and -N 1. Use
# --gpus-per-node so multiple GPUs land on ONE node (plain --gpus spreads them
# 1/node -> "required nodes (2)" error). Override on the CLI for sharding,
# e.g. gemma4_12b on 2x 3090 (faretra only):
#   MODEL=gemma4_12b sbatch -N 1 --gpus-per-node=nvidia_geforce_rtx_3090:2 -w faretra deploy/sbatch_script.sh
# Single-GPU models (default) can go to any node — but the repo + docker image
# must exist on that node (no shared FS). Pin with -w <node> if unsure.

#SBATCH --job-name=prompting
#SBATCH -N 1
#SBATCH --gpus-per-node=nvidia_geforce_rtx_3090:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00

set -e
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURM_NODELIST"
echo "Model      : ${MODEL:?set MODEL=<alias from configs/models.yaml>}"
echo "Start time : $(date)"

# Local (non-endpoint) models need HF_TOKEN for gated repos (Gemma etc.).
if [ -z "$OPENAI_BASE_URL" ] && [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN unset — gated models (Gemma, some Qwen) will fail to download."
fi

export IMAGE_NAME="${IMAGE_NAME:-prompting-gpu:latest}"

bash deploy/run_docker.sh deploy/run.sh --model "$MODEL" ${EXTRA_ARGS:-}

echo "End time : $(date)"
