#!/bin/bash
#SBATCH --job-name=798-tokenize
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_tokenize_%j.out
#SBATCH --error=logs/slurm_tokenize_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=amukherj@umd.edu

# Runs the full tokenization pipeline:
#   1. WavTokenizer encode: input pairs + target pairs -> discrete codes
#   2. Train a BPE codebook on the input codes
#   3. Apply BPE to input and target codes
#
# Prereqs: scripts/setup_env.sh has been run, data/paired_edits/ exists.

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p logs

source .venv/bin/activate

echo "=== GPU ==="
nvidia-smi || true

if ! python -c "import torch,sys; sys.exit(0 if torch.version.cuda is not None else 1)"; then
    echo "torch lacks CUDA; installing CUDA build ..."
    pip install --upgrade torch torchaudio --index-url https://download.pytorch.org/whl/cu121
fi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda is not None)"

echo "=== 1/3 Encode input pairs with WavTokenizer ==="
python tokenization/encode_wavtokenizer.py \
    --input data/paired_edits/input/ \
    --output data/tokens/input_wavtok/

echo "=== 2/3 Encode target pairs with WavTokenizer ==="
python tokenization/encode_wavtokenizer.py \
    --input data/paired_edits/target/ \
    --output data/tokens/target_wavtok/

echo "=== 3/3 Train + apply BPE ==="
python tokenization/train_bpe.py \
    --codes_dir data/tokens/input_wavtok/ \
    --output data/tokens/bpe_model/

python tokenization/apply_bpe.py \
    --codes_dir data/tokens/input_wavtok/ \
    --bpe_model data/tokens/bpe_model/ \
    --output data/tokens/input_bpe/

python tokenization/apply_bpe.py \
    --codes_dir data/tokens/target_wavtok/ \
    --bpe_model data/tokens/bpe_model/ \
    --output data/tokens/target_bpe/

echo "Done."
echo "Tokenized pairs: $(ls data/tokens/input_bpe/ | wc -l) input, $(ls data/tokens/target_bpe/ | wc -l) target"
