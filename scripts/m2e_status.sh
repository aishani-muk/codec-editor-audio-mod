#!/bin/bash
# Concise status for the M2E V/A scoring run:
#   - PID alive?
#   - Latest heartbeat row (pct, elapsed, eta)
#   - Cache size
#
# Usage:   scripts/m2e_status.sh
set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "── M2E V/A scoring status @ $(date '+%H:%M:%S') ───────────────────────"

PIDFILE=logs/m2e_scoring.pid
if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE")
    if kill -0 "$PID" 2>/dev/null; then
        ETIME=$(ps -o etime= -p "$PID" 2>/dev/null | tr -d ' ')
        PCPU=$(ps -o pcpu= -p "$PID" 2>/dev/null | tr -d ' ')
        echo "  process:  RUNNING  pid=$PID  etime=$ETIME  cpu=${PCPU}%"
    else
        echo "  process:  NOT RUNNING (stale pidfile at $PIDFILE)"
    fi
else
    echo "  process:  NOT RUNNING (no pidfile)"
fi

PROG=logs/m2e_scoring_progress.jsonl
if [ -s "$PROG" ]; then
    LAST=$(tail -n 1 "$PROG")
    echo "  latest heartbeat: $LAST"
else
    echo "  latest heartbeat: (progress log empty)"
fi

CACHE=results/va_cache.jsonl
if [ -s "$CACHE" ]; then
    N=$(wc -l < "$CACHE")
    SIZE=$(du -sh "$CACHE" 2>/dev/null | cut -f1)
    echo "  cache:    $N rows ($SIZE)  →  $CACHE"
else
    echo "  cache:    (empty)"
fi

echo
echo "Tail live: scripts/tail_logs.sh m2e_scoring"
