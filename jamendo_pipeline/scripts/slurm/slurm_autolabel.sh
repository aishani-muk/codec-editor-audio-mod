#!/bin/bash
#SBATCH --job-name=celtic-autolabel
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/celtic_autolabel_%j.out
#SBATCH --error=logs/celtic_autolabel_%j.err

# Day 2: tune-type auto-labeling. CPU-parallel with 8 workers.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if [[ ! -f .venv/bin/activate ]]; then
    echo "ERROR: $(pwd) has no .venv/bin/activate; submit from modelling/." >&2
    exit 2
fi
mkdir -p logs
ln -sfn "celtic_autolabel_${SLURM_JOB_ID}.out" logs/celtic_autolabel_latest.out
ln -sfn "celtic_autolabel_${SLURM_JOB_ID}.err" logs/celtic_autolabel_latest.err

source .venv/bin/activate
source jamendo_pipeline/env.sh

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python -u jamendo_pipeline/scripts/02_autolabel_tune_types.py \
    --workers 4

echo "Done. Summary: $JAMENDO_CACHE/clips/autolabel_summary.json"
