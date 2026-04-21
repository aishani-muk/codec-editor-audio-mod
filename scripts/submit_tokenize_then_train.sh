#!/bin/bash
# Submit the SLURM job chain: tokenize -> train.
#
# Step 1: tokenize all 7920 paired inputs + 7920 targets with WavTokenizer
#         speech-75token. ~25 min on A4000.
# Step 2: full codec-editor training (12 h, fp16), runs only if tokenize
#         succeeds (SLURM --dependency=afterok).
#
# Usage:   scripts/submit_tokenize_then_train.sh [run_name=proposed_v1]
# Output:  job IDs + the symlinks used by scripts/tail_logs.sh.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

RUN_NAME="${1:-proposed_v1}"

echo "▶ Submitting tokenization job ..."
TOK_JID=$(sbatch --parsable scripts/slurm/slurm_tokenize_speech75.sh)
echo "  tokenize jobid:  $TOK_JID"
echo "  tokenize log:    logs/slurm_tokenize_sp75_${TOK_JID}.out"

echo
echo "▶ Submitting training job (depends on afterok:${TOK_JID}) ..."
TRAIN_JID=$(sbatch --parsable \
    --dependency=afterok:${TOK_JID} \
    scripts/slurm/slurm_train.sh "$RUN_NAME")
echo "  train jobid:     $TRAIN_JID"
echo "  train log:       logs/slurm_train_${TRAIN_JID}.out"
echo "  run name:        $RUN_NAME"

echo
echo "▶ Submitted. squeue -u $USER says:"
squeue -u "$USER" -o "%.10i %.12j %.8T %.10M %.10L %.4C %.8m %.20R"

echo
echo "Live-tail with:"
echo "  scripts/tail_logs.sh tokenize   # while tokenization runs"
echo "  scripts/tail_logs.sh train      # once tokenization finishes"
echo "  scripts/status.sh --watch       # global dashboard"
