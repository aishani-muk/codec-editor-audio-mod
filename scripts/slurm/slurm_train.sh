#!/bin/bash
#SBATCH --job-name=798-train
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm_train_%j.out
#SBATCH --error=logs/slurm_train_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=amukherj@umd.edu

# Full training run of the proposed codec-to-codec editor.
# Prereq: smoke test passed (loss decreasing), data tokenized.

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p logs

# Publish a stable 'latest' symlink so `scripts/tail_logs.sh` always finds
# the newest run's output without needing the jobid.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    ln -sfn "slurm_train_${SLURM_JOB_ID}.out" logs/train_latest.out
    ln -sfn "slurm_train_${SLURM_JOB_ID}.err" logs/train_latest.err
fi

source .venv/bin/activate

# Force unbuffered output everywhere so `tail -F logs/train_latest.out`
# actually shows live progress.
export PYTHONUNBUFFERED=1

echo "=== GPU ==="
nvidia-smi || true

if ! python -c "import torch,sys; sys.exit(0 if torch.version.cuda is not None else 1)"; then
    echo "torch lacks CUDA; installing CUDA build ..."
    pip install --upgrade torch torchaudio --index-url https://download.pytorch.org/whl/cu121
fi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda is not None)"

RUN_NAME="${1:-proposed_v1}"
echo "Run name: $RUN_NAME"

python -u train.py \
    --config configs/proposed.yaml \
    --run_name "$RUN_NAME"

echo "Done."
echo "Best checkpoint: checkpoints/$RUN_NAME/best/"
echo "Plot loss curves:"
echo "  python scripts/plot_run.py --log checkpoints/$RUN_NAME/training_log.jsonl"
