#!/bin/bash
#SBATCH --job-name=celtic-eval
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/celtic_eval_%j.out
#SBATCH --error=logs/celtic_eval_%j.err
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=amukherj@umd.edu

# Day 7: unified evaluation across all trained tracks × u-levels.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if [[ ! -f .venv/bin/activate ]]; then
    echo "ERROR: $(pwd) has no .venv/bin/activate; submit from modelling/." >&2
    exit 2
fi
mkdir -p logs
ln -sfn "celtic_eval_${SLURM_JOB_ID}.out" logs/celtic_eval_latest.out

source .venv/bin/activate
source jamendo_pipeline/env.sh
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
nvidia-smi || true

TRACKS="${1:-a,b,c}"
US="${2:-0.0,0.3,0.6,0.9}"

# 1) Generate edited WAVs + per-clip metrics for every (track, u) pair.
python -u -m jamendo_pipeline.evaluation.run_full_eval \
    --tracks "$TRACKS" --u "$US" \
    --results_dir jamendo_pipeline/results

# 2) Bootstrap-aggregate each per_clip.jsonl into a summary.json.
RESULTS=jamendo_pipeline/results
IFS=',' read -ra TRACK_ARR <<< "$TRACKS"
IFS=',' read -ra U_ARR <<< "$US"
for t in "${TRACK_ARR[@]}"; do
    for u in "${U_ARR[@]}"; do
        UDIR="$RESULTS/celtic_track_${t}/u${u}"
        IN="$UDIR/per_clip.jsonl"
        OUT="$UDIR/summary.json"
        if [[ -f "$IN" ]]; then
            python -u jamendo_pipeline/scripts/aggregate_with_ci.py \
                --input "$IN" --output "$OUT" \
                --iters 1000 --stratify tune_type
        fi
    done
done

# 3) Build final figures + A/B HTML.
python -u jamendo_pipeline/scripts/make_panels.py \
    --results_dir "$RESULTS" \
    --tracks "$TRACKS" --u "$US" \
    --out "$RESULTS/celtic_panels.png"

python -u jamendo_pipeline/scripts/make_ab_html.py \
    --results_dir "$RESULTS" \
    --tracks "$TRACKS" --u 0.6 \
    --out "$RESULTS/ab_pairs.html"

echo "Eval pipeline done. See $RESULTS/"
