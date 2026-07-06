#!/bin/bash
# vast_setup.sh — provision a fresh rented GPU box (vast.ai) for the vLLM grid.
#
# Unlike the 3090 cluster (CUDA 12.4, capped at the pinned vllm==0.6.6 that does
# NOT know gemma-4), a rented instance has a recent driver, so we install the
# LATEST vllm — the one that supports gemma-4 / qwen-3.5 — with the torch it ships.
#
# Usage:
#   HF_TOKEN=hf_xxx bash deploy/vast_setup.sh
#   python run.py --model gemma4_12b --models configs/models.vast.yaml
set -e

# Absolute project root (Prompting/), regardless of where this is invoked from.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${HF_TOKEN:?set HF_TOKEN=hf_... first (gemma / qwen are gated)}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

# Install deps EXCEPT the cluster pins: vllm/torch here must be the modern ones,
# and bitsandbytes isn't needed (bf16 on a big card). Everything else is shared.
grep -ivE '^(vllm|torch|torchvision|torchaudio|bitsandbytes)' "$ROOT/requirements.txt" > /tmp/req_vast.txt
pip install -r /tmp/req_vast.txt
pip install -U vllm

python - <<'PY'
import torch, vllm
print(f"torch {torch.__version__} | cuda {torch.version.cuda} | vllm {vllm.__version__}")
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"gpu: {p.name}  {p.total_memory/1e9:.0f}GB")
PY

echo
echo "ready. run e.g.:"
echo "  HF_TOKEN=\$HF_TOKEN python run.py --model gemma4_12b --models configs/models.vast.yaml"
