#!/bin/bash
#SBATCH --job-name=celtic-pull
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/celtic_pull_%j.out
#SBATCH --error=logs/celtic_pull_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=amukherj@umd.edu

# Day 1: pull Celtic corpus from Jamendo + extract 10-s WAV clips.
# CPU-only; streams MP3s to /scratch0 and decodes to 24 kHz mono WAVs.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if [[ ! -f .venv/bin/activate ]]; then
    echo "ERROR: $(pwd) has no .venv/bin/activate; submit from modelling/." >&2
    exit 2
fi
mkdir -p logs
ln -sfn "celtic_pull_${SLURM_JOB_ID}.out" logs/celtic_pull_latest.out
ln -sfn "celtic_pull_${SLURM_JOB_ID}.err" logs/celtic_pull_latest.err

source .venv/bin/activate
source jamendo_pipeline/env.sh

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2

python -u jamendo_pipeline/scripts/01_pull_celtic_corpus.py \
    --n_tracks "${1:-2000}" \
    --clips_per_track "${2:-4}"

echo "Done. Corpus at: $JAMENDO_CACHE/clips"
