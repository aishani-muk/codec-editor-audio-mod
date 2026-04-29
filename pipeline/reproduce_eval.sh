#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
VENV="${VENV:-/fs/classhomes/amukherj/research/798/modelling/.venv}"
[[ -f "$VENV/bin/activate" ]] || { echo "ERROR: VENV=$VENV not found." >&2; exit 1; }
source "$VENV/bin/activate"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

OUT_BASE=results/lora_repro
mkdir -p "$OUT_BASE"
for u in 0.0 0.3 0.6 0.9; do
    OUT="$OUT_BASE/u${u}"
    mkdir -p "$OUT"
    echo "=== LoRA inference @ u=$u ==="
    python infer_stream_musicgen.py \
        --ckpt_dir checkpoints/editor_v3_lora \
        --input_dir data/demo_inputs \
        --output_dir "$OUT" \
        --u "$u"
done

echo
echo "=== u-collapse check (per clip, should be identical across u) ==="
for stem in yaman_c00 todi_c00 shree_c00; do
    echo "  $stem:"
    for u in 0.0 0.3 0.6 0.9; do
        H=$(md5sum "$OUT_BASE/u${u}/${stem}.wav" 2>/dev/null | awk '{print $1}')
        printf "    u=%s  md5=%s\n" "$u" "$H"
    done
done
echo
echo "If all 4 md5s per clip match, u-collapse is reproduced."
