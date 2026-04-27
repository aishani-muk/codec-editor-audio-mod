#!/bin/bash
#SBATCH --job-name=celtic-track-a
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/celtic_track_a_%j.out
#SBATCH --error=logs/celtic_track_a_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=amukherj@umd.edu

# Day 4: train Track A (GPT-2 + Celtic loss stack).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if [[ ! -f .venv/bin/activate ]]; then
    echo "ERROR: $(pwd) has no .venv/bin/activate; submit from modelling/." >&2
    exit 2
fi
mkdir -p logs
ln -sfn "celtic_track_a_${SLURM_JOB_ID}.out" logs/celtic_track_a_latest.out

source /scratch0/$USER/celtic-venv/bin/activate
source jamendo_pipeline/env.sh
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

nvidia-smi || true

# Build token→PCD / token→rhythm matrices on first run (no-op on reruns).
python -u -m jamendo_pipeline.losses.build_token_matrices || true

python -u -m jamendo_pipeline.train.train_track_a \
    --config jamendo_pipeline/configs/track_a_gpt2.yaml

echo "Track A done: checkpoints/celtic_track_a/best.pt"
