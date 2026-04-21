#!/bin/bash
#SBATCH --job-name=798-eval
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_eval_%j.out
#SBATCH --error=logs/slurm_eval_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=amukherj@umd.edu

set -euo pipefail
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p logs results

source .venv/bin/activate

if ! python -c "import torch,sys; sys.exit(0 if torch.version.cuda is not None else 1)"; then
    echo "torch lacks CUDA; installing CUDA build ..."
    pip install --upgrade torch torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

RUN_NAME="${1:-proposed_v1}"
CKPT_DIR="checkpoints/$RUN_NAME/best"
CONFIG="checkpoints/$RUN_NAME/config.yaml"
TEST_DIR="data/test_clips"

[ -d "$CKPT_DIR" ]  || { echo "ERROR: $CKPT_DIR missing. Train first."; exit 1; }
[ -d "$TEST_DIR" ]  || { echo "ERROR: $TEST_DIR missing. Run scripts/stage_test_clips.sh first."; exit 1; }

echo "=== 1/3 Proposed inference @ u in {0.0 0.3 0.6 0.9} ==="
for u in 0.0 0.3 0.6 0.9; do
    out_dir="results/$RUN_NAME/u${u}"
    mkdir -p "$out_dir"
    for wav in "$TEST_DIR"/*.wav; do
        stem=$(basename "$wav" .wav)
        python infer_stream.py \
            --input "$wav" \
            --output "$out_dir/${stem}.wav" \
            --checkpoint "$CKPT_DIR" \
            --config "$CONFIG" \
            --u "$u"
    done
done

echo "=== 2/3 Baselines (DSP + EnCodec-MLP) @ u=0.6 ==="
mkdir -p results/baseline_dsp results/baseline_encodec_mlp
python baselines/dsp_baseline.py \
    --input "$TEST_DIR" --output results/baseline_dsp --u 0.6
python baselines/encodec_mlp_baseline.py \
    --input "$TEST_DIR" --output results/baseline_encodec_mlp --u 0.6

echo "=== 3/3 Evaluate every condition ==="
for cond_dir in results/$RUN_NAME/u* results/baseline_dsp results/baseline_encodec_mlp; do
    name=$(basename "$cond_dir")
    python evaluate.py \
        --input "$TEST_DIR" \
        --output "$cond_dir" \
        --tonic_dir "$TEST_DIR" \
        --results "$cond_dir/eval_results.json"
done

echo "Done. Aggregate metrics in each results/*/eval_results.json"
