#!/bin/bash
#SBATCH --job-name=798-medsmoke
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm_medsmoke_%j.out
#SBATCH --error=logs/slurm_medsmoke_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=amukherj@umd.edu

# 2000-step medium-smoke. Validates that the conditional editor actually
# learns on this dataset before we commit to the 12h full run.
#
# Watch loss trajectory in logs/slurm_medsmoke_${SLURM_JOB_ID}.out and
# checkpoints/medium_smoke_*/training_log.jsonl.
#
# Success criteria:
#   - val loss monotonically decreasing over the 10 eval points
#   - train loss < val loss, gap stable
#   - no NaNs

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p logs configs/_generated

source .venv/bin/activate

echo "=== GPU ==="
nvidia-smi || true

if ! python -c "import torch,sys; sys.exit(0 if torch.version.cuda is not None else 1)"; then
    echo "torch lacks CUDA; installing CUDA build ..."
    pip install --upgrade torch torchaudio --index-url https://download.pytorch.org/whl/cu121
fi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda is not None)"

python train.py --config configs/smoke/medium_smoke.yaml --run_name "medium_smoke_$(date +%s)" \
  || { echo "MEDIUM SMOKE FAILED"; exit 1; }

echo ""
echo "Medium smoke done. Check loss curve:"
echo "  python scripts/plot_run.py --log checkpoints/medium_smoke_*/training_log.jsonl"
