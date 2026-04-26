#!/bin/bash
#SBATCH --job-name=celtic-track-b
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/celtic_track_b_%j.out
#SBATCH --error=logs/celtic_track_b_%j.err
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=amukherj@umd.edu

# Day 5: train Track B (MusicGen-small + LoRA on Celtic natural pairs).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if [[ ! -f .venv/bin/activate ]]; then
    echo "ERROR: $(pwd) has no .venv/bin/activate; submit from modelling/." >&2
    exit 2
fi
mkdir -p logs
ln -sfn "celtic_track_b_${SLURM_JOB_ID}.out" logs/celtic_track_b_latest.out

source .venv/bin/activate
source jamendo_pipeline/env.sh
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

nvidia-smi || true

python -u -m jamendo_pipeline.train.train_track_b \
    --config jamendo_pipeline/configs/track_b_musicgen_lora.yaml

echo "Track B done: checkpoints/celtic_track_b/best.pt"
