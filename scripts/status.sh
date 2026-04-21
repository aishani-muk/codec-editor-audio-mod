#!/bin/bash
# Print a concise status summary of every long-running job in the project:
#   - SLURM queue for the current user
#   - Background PIDs for known job types (via .pid files)
#   - Last 5 lines of every logs/*_latest.out symlink
#
# Usage:
#   scripts/status.sh           # one-shot snapshot
#   scripts/status.sh --watch   # refresh every 10 s

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

_snapshot() {
    echo "======================================================================"
    echo " STATUS @ $(date '+%Y-%m-%d %H:%M:%S')"
    echo "======================================================================"

    echo
    echo "── SLURM queue (squeue -u $USER) ────────────────────────────────────"
    squeue -u "$USER" -o "%.10i %.12j %.8T %.10M %.10L %.4C %.8m %.20R" 2>/dev/null \
        || echo "  (squeue unavailable)"

    echo
    echo "── Background PIDs (logs/*.pid) ──────────────────────────────────────"
    shopt -s nullglob
    any_pid=0
    for pidfile in logs/*.pid; do
        any_pid=1
        pid=$(cat "$pidfile" 2>/dev/null || echo "?")
        name=$(basename "$pidfile" .pid)
        if [ "$pid" != "?" ] && kill -0 "$pid" 2>/dev/null; then
            etime=$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')
            pcpu=$(ps -o pcpu= -p "$pid" 2>/dev/null | tr -d ' ')
            pmem=$(ps -o pmem= -p "$pid" 2>/dev/null | tr -d ' ')
            echo "  [RUN]  $name  pid=$pid  etime=$etime  cpu=${pcpu}%  mem=${pmem}%"
        else
            echo "  [GONE] $name  pid=$pid  (stale .pid file; removing)"
            rm -f "$pidfile"
        fi
    done
    [ "$any_pid" -eq 0 ] && echo "  (no tracked background jobs)"

    echo
    echo "── Latest logs (tail -n 5 logs/*_latest.out) ─────────────────────────"
    for log in logs/*_latest.out; do
        [ -e "$log" ] || continue
        echo
        echo "  ▶ $log"
        tail -n 5 "$log" 2>/dev/null | sed 's/^/     /'
    done
    shopt -u nullglob

    echo
    echo "── Disk (du -sh data/ .cache/ ~/) ────────────────────────────────────"
    df -h /fs/classhomes 2>/dev/null | tail -n 1 | awk '{printf "  classhomes: %s used of %s (%s full)\n",$3,$2,$5}'
    du -sh /fs/classhomes/amukherj 2>/dev/null | awk '{printf "  ~:          %s\n",$1}'
    du -sh data 2>/dev/null | awk '{printf "  data/:      %s\n",$1}'
    du -sh checkpoints 2>/dev/null | awk '{printf "  ckpts/:     %s\n",$1}'

    echo
}

if [ "${1:-}" = "--watch" ]; then
    while true; do
        clear
        _snapshot
        sleep 10
    done
else
    _snapshot
fi
