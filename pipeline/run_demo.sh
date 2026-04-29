#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
VENV="${VENV:-/fs/classhomes/amukherj/research/798/modelling/.venv}"
[[ -f "$VENV/bin/activate" ]] || { echo "ERROR: VENV=$VENV not found. Set VENV=/path/to/venv or pip install -r requirements.txt into ./.venv first." >&2; exit 1; }
source "$VENV/bin/activate"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python regenerate_demo.py
