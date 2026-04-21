#!/bin/bash
# Stage the 2 Yaman tracks (held-out test set) into a flat directory.
# Also copies tonic/pitch metadata, renaming them to match the audio stem
# so evaluate.py can find them.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TEST_DIR="data/test_clips"
rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR"

declare -a SRC=(
    "data/saraga_kalyan_thaat/Hindustani Alaps by Kaustuv Kanti Ganguli/Raag Yaman"
    "data/saraga_kalyan_thaat/Raag Bahar, Gaud Malhar & Yaman by Omkar Dadarkar/Raag Yaman"
)
declare -a NAME=("yaman_ganguli" "yaman_dadarkar")

for i in "${!SRC[@]}"; do
    src="${SRC[$i]}"
    name="${NAME[$i]}"
    cp "$src/Raag Yaman.wav"           "$TEST_DIR/${name}.wav"
    cp "$src/Raag Yaman.ctonic.txt"    "$TEST_DIR/${name}.ctonic.txt"
    cp "$src/Raag Yaman.pitch.txt"     "$TEST_DIR/${name}.pitch.txt"
    cp "$src/Raag Yaman.json"          "$TEST_DIR/${name}.json"
done

echo "Staged test clips:"
ls -lh "$TEST_DIR"
