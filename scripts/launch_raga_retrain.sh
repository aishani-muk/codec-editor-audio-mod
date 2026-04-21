#!/bin/bash
# Stage full Saraga 1.5 Hindustani, then retrain the PCD+ raga classifier on
# all ragas (not just Kalyan thaat). Runs fully on CPU in the background.
#
# Outputs:
#   data/saraga_all_ragas/...                     (staged WAVs + metadata)
#   checkpoints/raga_classifier_pcd_v2_full/      (trained model)
#   logs/raga_retrain.out  +  _latest symlink
#   logs/raga_retrain.pid
#   logs/raga_retrain_progress.jsonl              (per-epoch heartbeat)

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

# Refuse double-launch.
if [ -f logs/raga_retrain.pid ]; then
    OLD=$(cat logs/raga_retrain.pid)
    if kill -0 "$OLD" 2>/dev/null; then
        echo "raga retrain already running at pid=$OLD. Skipping."
        exit 0
    fi
    rm -f logs/raga_retrain.pid
fi

source .venv/bin/activate
export PYTHONUNBUFFERED=1

RAW_DIR="${RAW_DIR:-/fs/classhomes/amukherj/research/saraga1.5_hindustani}"
STAGED_DIR="${STAGED_DIR:-data/saraga_all_ragas}"
OUT_DIR="${OUT_DIR:-checkpoints/raga_classifier_pcd_v2_full}"
RUN_LOG="logs/raga_retrain.out"

{
    echo "=== $(date '+%F %T') ==="
    echo "=== 1/2 Staging Saraga 1.5 Hindustani (all ragas) ==="
    python -u scripts/stage_saraga_all_ragas.py \
        --raw_dir "$RAW_DIR" \
        --output  "$STAGED_DIR" \
        --workers 4
    echo
    echo "=== 2/2 Training PCD+ classifier on staged data ==="
    python -u train_raga_classifier_pcd_v2.py \
        --saraga_dir "$STAGED_DIR" \
        --output     "$OUT_DIR" \
        --clip_seconds 30.0 \
        --hop_seconds 10.0 \
        --pretrain_epochs 20 \
        --epochs 50
    echo
    echo "=== $(date '+%F %T') Done. ==="
} > "$RUN_LOG" 2>&1 &

PID=$!
echo "$PID" > logs/raga_retrain.pid
ln -sfn "raga_retrain.out" logs/raga_retrain_latest.out

echo "raga retrain launched."
echo "  pid:  $PID"
echo "  log:  $RUN_LOG  (symlinked as logs/raga_retrain_latest.out)"
echo
echo "Tail with:  scripts/tail_logs.sh raga_retrain"
echo "Status:     scripts/status.sh"
echo "Stop with:  kill \$(cat logs/raga_retrain.pid)"
