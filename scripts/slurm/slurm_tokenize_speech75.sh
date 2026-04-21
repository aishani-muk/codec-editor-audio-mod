#!/bin/bash
#SBATCH --job-name=798-tok-sp75
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm_tokenize_sp75_%j.out
#SBATCH --error=logs/slurm_tokenize_sp75_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=amukherj@umd.edu

# Re-tokenize the paired input/target clips with
# novateur/WavTokenizer-large-speech-75token (75 tok/s, speech-heavy),
# better matched to Hindustani vocal than the generic unify-40 model.
#
# Outputs land in `data/tokens/input_wavtok_speech75/` and
# `data/tokens/target_wavtok_speech75/`, leaving the existing
# unify-40 codes intact for ablation.

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p logs

# Stable 'latest' symlinks so `scripts/tail_logs.sh` can live-follow this job.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    ln -sfn "slurm_tokenize_sp75_${SLURM_JOB_ID}.out" logs/tokenize_latest.out
    ln -sfn "slurm_tokenize_sp75_${SLURM_JOB_ID}.err" logs/tokenize_latest.err
fi

source .venv/bin/activate

# Unbuffered output for live tailing.
export PYTHONUNBUFFERED=1

echo "=== GPU ==="
nvidia-smi || true

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda is not None)"

MODEL="novateur/WavTokenizer-large-speech-75token"

echo "=== 1/2 Encode input pairs with speech-75token ==="
python -u tokenization/encode_wavtokenizer.py \
    --input data/paired_edits/input/ \
    --output data/tokens/input_wavtok_speech75/ \
    --model "$MODEL"

echo "=== 2/2 Encode target pairs with speech-75token ==="
python -u tokenization/encode_wavtokenizer.py \
    --input data/paired_edits/target/ \
    --output data/tokens/target_wavtok_speech75/ \
    --model "$MODEL"

echo "Done."
echo "Tokenized pairs:"
printf "  input:  %s\n  target: %s\n" \
    "$(ls data/tokens/input_wavtok_speech75/ | wc -l)" \
    "$(ls data/tokens/target_wavtok_speech75/ | wc -l)"
