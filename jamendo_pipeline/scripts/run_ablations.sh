#!/usr/bin/env bash
# Day 7: Track A ablation sweep.
#
# 7 ablations × ~1 h each on 1 GPU = ~7 h. Each variant overrides ONE
# knob in the Track A config and saves to its own checkpoint dir so the
# results/ablation_table.md can be assembled by scripts/make_ablation_table.py
# afterwards.
#
# Usage:
#   bash scripts/run_ablations.sh                 # runs all 7
#   bash scripts/run_ablations.sh no_rhythm       # runs just one

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"
if [[ ! -f .venv/bin/activate ]]; then
    echo "ERROR: $(pwd) has no .venv/bin/activate; run from modelling/." >&2
    exit 2
fi
source jamendo_pipeline/env.sh

ABLATIONS=(
    "full"                # baseline — reuses the main Track A run
    "no_pcd_kl"
    "no_anti_collapse"
    "no_mert_perceptual"
    "no_rhythm_preserve"  # Celtic-specific aux
    "no_curriculum_u"
    "no_ema"
    "no_scheduled_sampling"
)

ONLY="${1:-all}"

for abl in "${ABLATIONS[@]}"; do
    if [[ "$ONLY" != "all" && "$ONLY" != "$abl" ]]; then
        continue
    fi
    echo "=== ablation: $abl ==="
    # Skip the "full" pseudo-ablation; it's the main run.
    if [[ "$abl" == "full" ]]; then
        continue
    fi

    # Build a per-ablation config via YAML override. We pass these via
    # env vars and let a tiny wrapper script load the base config and
    # rewrite the specific weight to 0 / flag to false.
    CKPT_NAME="celtic_track_a_abl_${abl}"
    OVERRIDE="$abl" \
    python -u -c "
import os, sys, yaml
from pathlib import Path
REPO=Path('$REPO_ROOT')
sys.path.insert(0, str(REPO))
from jamendo_pipeline.train.train_track_a import load_merged_config
cfg = load_merged_config('jamendo_pipeline/configs/track_a_gpt2.yaml')
abl = os.environ['OVERRIDE']
tr  = cfg['training']
if abl == 'no_pcd_kl':            tr['pcd_kl_weight'] = 0.0
elif abl == 'no_anti_collapse':    tr['anti_collapse_weight'] = 0.0
elif abl == 'no_mert_perceptual':  tr['mert_perceptual_weight'] = 0.0
elif abl == 'no_rhythm_preserve':  tr['rhythm_preserve_weight'] = 0.0
elif abl == 'no_curriculum_u':     tr['curriculum_u'] = [{'until_step': 10**9, 'u_choices': 'uniform'}]
elif abl == 'no_ema':              tr['ema_decay'] = 0.0
elif abl == 'no_scheduled_sampling': tr['scheduled_sampling']['enabled'] = False
tr['max_steps'] = min(tr.get('max_steps', 25000), 3000)   # 3k steps per plan
cfg['checkpoints']['name'] = '$CKPT_NAME'
out = Path('jamendo_pipeline/configs') / f'_abl_{abl}.yaml'
out.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(out)
"
    python -u -m jamendo_pipeline.train.train_track_a \
        --config "jamendo_pipeline/configs/_abl_${abl}.yaml"
done

echo "All ablations done. Build the table with:"
echo "  python -u jamendo_pipeline/scripts/make_ablation_table.py"
