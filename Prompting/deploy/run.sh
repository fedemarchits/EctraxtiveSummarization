#!/bin/bash
# run.sh — set env, optionally build reasoning traces, then run the pipeline.
# All args are forwarded to `python run.py` (e.g. --model qwen35_4b).
set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
# Reduce CUDA allocator fragmentation on tight-fit models (E4B etc.).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p /tmp && chmod 1777 /tmp 2>/dev/null || true
export TMPDIR=/tmp TEMP=/tmp TMP=/tmp

cd /workspace   # the Prompting/ folder is mounted here

echo "=============================="
echo "Prompting pipeline"
echo "args        : $*"
echo "CUDA        : $CUDA_VISIBLE_DEVICES"
echo "HF_HOME     : $HF_HOME"
echo "endpoint    : ${OPENAI_BASE_URL:-<unset>}"
echo "=============================="

# bitsandbytes / vllm aren't baked into the image; install once at runtime only
# if missing (bitsandbytes: 4-bit weights; vllm: fast backend for gemma4_12b).
python -c "import bitsandbytes" 2>/dev/null || pip install -q bitsandbytes
python -c "import vllm" 2>/dev/null || pip install -q vllm

# One-time: cache reference reasoning traces (skips files that already exist
# is handled inside; set BUILD_RATIONALES=1 to (re)generate).
if [ "${BUILD_RATIONALES:-0}" = "1" ]; then
    echo ">> building reasoning traces"
    python -u -m scripts.build_rationales
fi

python -u run.py "$@"
