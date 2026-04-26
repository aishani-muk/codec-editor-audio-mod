#!/bin/bash
#SBATCH --job-name=celtic-track-c
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=logs/celtic_track_c_%j.out
#SBATCH --error=logs/celtic_track_c_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=amukherj@umd.edu

# Day 6: train Track C (MusicGen encoder → GPT-2 head hybrid).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if [[ ! -f .venv/bin/activate ]]; then
    echo "ERROR: $(pwd) has no .venv/bin/activate; submit from modelling/." >&2
    exit 2
fi
mkdir -p logs
ln -sfn "celtic_track_c_${SLURM_JOB_ID}.out" logs/celtic_track_c_latest.out

source .venv/bin/activate
source jamendo_pipeline/env.sh
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

nvidia-smi || true

python -u -m jamendo_pipeline.train.train_track_c \
    --config jamendo_pipeline/configs/track_c_hybrid.yaml

echo "Track C done: checkpoints/celtic_track_c/best.pt"
