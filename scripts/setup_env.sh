#!/bin/bash
# Bootstrap the full Python environment (GPU-capable) for training.
# Run ONCE before any SLURM jobs. Idempotent-ish.
#
# This script should be run ON A GPU NODE (so that CUDA-compatible torch is
# installed correctly). On NEXUS / UMIACS, srun into an interactive session:
#   srun -p class --gres=gpu:1 --mem=16G --pty bash
# then run this script.
#
# Afterwards every SLURM batch script can just `source .venv/bin/activate`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ ! -d .venv ]; then
    echo "Creating .venv ..."
    python3.11 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip

echo "Installing core deps from requirements.txt ..."
pip install -r requirements.txt

# Override torch with a CUDA build if we're on a GPU node and CUDA is visible
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU detected; installing CUDA-enabled torch ..."
    pip install --upgrade torch torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "No GPU visible; leaving current torch install as-is."
fi

# WavTokenizer is distributed as a GitHub repo, not on PyPI
if [ ! -d third_party/WavTokenizer ]; then
    echo "Cloning WavTokenizer into third_party/ ..."
    mkdir -p third_party
    git clone https://github.com/jishengpeng/WavTokenizer.git third_party/WavTokenizer
fi

echo ""
echo "Env ready. Test with:"
echo "  source .venv/bin/activate"
echo "  python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
