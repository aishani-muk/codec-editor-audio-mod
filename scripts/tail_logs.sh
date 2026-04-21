#!/bin/bash
# Live-tail every logs/*_latest.out symlink side-by-side.
# Symlinks are refreshed by each phase's launcher so this picks up new jobs
# automatically without needing to restart tail.
#
# Usage:
#   scripts/tail_logs.sh            # tail all known latest logs
#   scripts/tail_logs.sh train      # tail only logs/train_latest.out
#   scripts/tail_logs.sh train m2e  # tail a subset

set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

shopt -s nullglob
logs=()
if [ "$#" -eq 0 ]; then
    for f in logs/*_latest.out logs/*_latest.err; do
        logs+=("$f")
    done
else
    for arg in "$@"; do
        for f in logs/${arg}_latest.*; do
            logs+=("$f")
        done
    done
fi
shopt -u nullglob

if [ "${#logs[@]}" -eq 0 ]; then
    echo "No logs found. Expected symlinks at logs/*_latest.{out,err}." >&2
    exit 1
fi

echo "Tailing:"
for l in "${logs[@]}"; do echo "  $l"; done
echo "--- (Ctrl-C to exit) ---"
exec tail -Fn 50 "${logs[@]}"
