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

#SBATCH --job-name=prompting
#SBATCH --gres=gpu:nvidia_geforce_rtx_3090:1
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
