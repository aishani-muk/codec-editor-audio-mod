#!/bin/bash
#SBATCH --job-name=celtic-mine
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/celtic_mine_%j.out
#SBATCH --error=logs/celtic_mine_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=amukherj@umd.edu

# Day 3: Music2Emo scoring + calm↔energetic pair mining.
# CPU-heavy (Music2Emo), needs no GPU.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if [[ ! -f .venv/bin/activate ]]; then
    echo "ERROR: $(pwd) has no .venv/bin/activate; submit from modelling/." >&2
    exit 2
fi
mkdir -p logs
ln -sfn "celtic_mine_${SLURM_JOB_ID}.out" logs/celtic_mine_latest.out
ln -sfn "celtic_mine_${SLURM_JOB_ID}.err" logs/celtic_mine_latest.err

source /scratch0/$USER/celtic-venv/bin/activate
source jamendo_pipeline/env.sh

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2

python -u jamendo_pipeline/scripts/04_mine_mood_pairs.py \
    --min_delta_arousal 1.0

echo "Done. Pairs: $JAMENDO_CACHE/pairs/pairs.jsonl"
