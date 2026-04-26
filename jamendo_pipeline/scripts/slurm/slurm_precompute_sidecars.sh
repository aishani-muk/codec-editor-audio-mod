#!/bin/bash
#SBATCH --job-name=celtic-side
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/celtic_precompute_%j.out
#SBATCH --error=logs/celtic_precompute_%j.err

# Day 3: PCD + MERT + CREPE tonic + rhythm sidecars (1 GPU for MERT).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if [[ ! -f .venv/bin/activate ]]; then
    echo "ERROR: $(pwd) has no .venv/bin/activate; submit from modelling/." >&2
    exit 2
fi
mkdir -p logs
ln -sfn "celtic_precompute_${SLURM_JOB_ID}.out" logs/celtic_precompute_latest.out

source .venv/bin/activate
source jamendo_pipeline/env.sh
export PYTHONUNBUFFERED=1

python -u jamendo_pipeline/scripts/05_precompute_sidecars.py
python -u jamendo_pipeline/scripts/07_tokenize_celtic.py \
    --also_test_clips

echo "Done. Sidecars + tokens under $JAMENDO_CACHE"
