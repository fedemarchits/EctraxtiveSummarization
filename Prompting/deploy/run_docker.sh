#!/bin/bash
# run_docker.sh — run a script inside the GPU container with the Prompting/
# folder mounted at /workspace. Remaining args are forwarded to that script.
#
# Env:
#   IMAGE_NAME       docker image (default: prompting-gpu:latest)
#   HF_TOKEN         HuggingFace token (gated models: Gemma, some Qwen)
#   OPENAI_BASE_URL  } endpoint backend for large tiers (27B / 31B)
#   OPENAI_API_KEY   }
#   BUILD_RATIONALES=1  build reasoning traces before the run
set -e

IMAGE_NAME="${IMAGE_NAME:-prompting-gpu:latest}"
SCRIPT="${1:-deploy/run.sh}"
shift || true

# Mount the Prompting/ root (parent of this deploy/ dir), not the cwd.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Load secrets once from Prompting/.env (HF_TOKEN, OPENAI_*), if present.
# .env is the source of truth for secrets; put them there instead of the CLI.
if [ -f "$PROJECT_ROOT/.env" ]; then
    echo "Loading secrets from .env"
    set -a
    # shellcheck disable=SC1090
    . "$PROJECT_ROOT/.env"
    set +a
fi

# Which physical GPU(s) to use inside the container. Priority:
#   GPUS env  >  SLURM's CUDA_VISIBLE_DEVICES  >  0
# Pick a FREE card here (see `nvidia-smi`); GPU 0 is often busy on a shared node.
# For 2-GPU sharding (e.g. gemma4_12b): GPUS=0,1
GPU_SEL="${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"

echo "Docker image : $IMAGE_NAME"
echo "Project root : $PROJECT_ROOT"
echo "Script       : $SCRIPT   args: $*"
echo "HF_TOKEN     : $([ -n "$HF_TOKEN" ] && echo set || echo UNSET)"
echo "endpoint     : ${OPENAI_BASE_URL:-<unset>}"
echo "GPU(s)       : $GPU_SEL"

docker run --rm \
    --gpus all \
    --shm-size=16g \
    -v "$PROJECT_ROOT":/workspace \
    -v /llms:/llms \
    -e HF_TOKEN="$HF_TOKEN" \
    -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
    -e OPENAI_BASE_URL="$OPENAI_BASE_URL" \
    -e OPENAI_API_KEY="$OPENAI_API_KEY" \
    -e BUILD_RATIONALES="${BUILD_RATIONALES:-0}" \
    -e CUDA_VISIBLE_DEVICES="$GPU_SEL" \
    -e HF_HOME=/workspace/.cache/huggingface \
    "$IMAGE_NAME" \
    bash "$SCRIPT" "$@"
