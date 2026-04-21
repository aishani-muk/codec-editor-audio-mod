#!/bin/bash
# Launch M2E V/A scoring in the background (nohup). Script is resumable;
# relaunching after a kill just picks up from the last cached key.
#
# Output:
#   logs/m2e_scoring.out            (stdout of the script; live, unbuffered)
#   logs/m2e_scoring.pid            (PID of the nohup'd python process)
#   logs/m2e_scoring_latest.out     (symlink to the current .out for tail_logs.sh)
#   logs/m2e_scoring_progress.jsonl (per-call heartbeat; timestamps + ETA)
#   results/va_cache.jsonl          (append-only cache of every scored clip)

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs results

# Refuse to launch if a prior run is still alive.
if [ -f logs/m2e_scoring.pid ]; then
    OLD=$(cat logs/m2e_scoring.pid)
    if kill -0 "$OLD" 2>/dev/null; then
        echo "M2E scorer already running at pid=$OLD. Skipping."
        echo "  (kill $OLD first to relaunch)"
        exit 0
    fi
    rm -f logs/m2e_scoring.pid
fi

source .venv/bin/activate
export PYTHONUNBUFFERED=1

nohup python -u scripts/m2e_cpu_score.py > logs/m2e_scoring.out 2>&1 &
PID=$!
echo "$PID" > logs/m2e_scoring.pid
ln -sfn "m2e_scoring.out" logs/m2e_scoring_latest.out

echo "M2E scorer launched."
echo "  pid:       $PID"
echo "  log:       logs/m2e_scoring.out  (symlinked as logs/m2e_scoring_latest.out)"
echo "  progress:  logs/m2e_scoring_progress.jsonl"
echo "  cache:     results/va_cache.jsonl"
echo
echo "Tail with:   scripts/tail_logs.sh m2e_scoring"
echo "Status:      scripts/m2e_status.sh"
echo "Stop with:   kill \$(cat logs/m2e_scoring.pid)"
