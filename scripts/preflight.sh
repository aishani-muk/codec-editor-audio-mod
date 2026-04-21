#!/bin/bash
# Pre-flight validation before submitting SLURM jobs.
# Run on a login node; exits 0 if ready, 1 if something's missing.

set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

STAGE="${1:-all}"
fail=0
ok()   { echo "  [ok]   $1"; }
warn() { echo "  [warn] $1"; }
err()  { echo "  [err]  $1"; fail=1; }

check_stage_data() {
    echo "Stage: data"
    [ -d data/saraga_kalyan_thaat ] && ok "data/saraga_kalyan_thaat" || err "missing data/saraga_kalyan_thaat"
    n=$(find data/saraga_kalyan_thaat -name '*.wav' 2>/dev/null | wc -l)
    [ "$n" -eq 15 ] && ok "$n WAVs" || err "expected 15 WAVs, found $n"
    [ -d data/raga_features ] && ok "data/raga_features" || warn "raga_features missing (run extract_raga_features.py)"
    [ -d data/test_clips ] && ok "data/test_clips" || warn "test_clips missing (run stage_test_clips.sh)"
}

check_stage_pairs() {
    echo "Stage: pairs"
    [ -d data/paired_edits/input ] || { err "data/paired_edits/input missing"; return; }
    n_in=$(ls data/paired_edits/input | wc -l)
    n_tg=$(ls data/paired_edits/target | wc -l)
    n_me=$(ls data/paired_edits/meta | wc -l)
    [ "$n_in" -eq "$n_tg" ] && [ "$n_in" -eq "$n_me" ] \
        && ok "$n_in input/target/meta triples" \
        || err "mismatched counts: in=$n_in target=$n_tg meta=$n_me"
    [ "$n_in" -ge 100 ] && ok "pair count >=100" || warn "only $n_in pairs (prepare_pairs still running?)"
}

check_stage_tokens() {
    echo "Stage: tokens"
    for d in input_wavtok target_wavtok input_bpe target_bpe bpe_model; do
        if [ -d "data/tokens/$d" ]; then
            n=$(ls "data/tokens/$d" 2>/dev/null | wc -l)
            ok "data/tokens/$d ($n files)"
        else
            err "data/tokens/$d missing (run slurm_tokenize.sh)"
        fi
    done
}

check_stage_env() {
    echo "Stage: env"
    [ -d .venv ] && ok ".venv exists" || { err ".venv missing (run scripts/setup_env.sh)"; return; }
    source .venv/bin/activate
    python -c "import torch; import transformers; import librosa; import soundfile" 2>/dev/null \
        && ok "torch/transformers/librosa import" \
        || err "missing python deps (run setup_env.sh)"
    if command -v nvidia-smi >/dev/null 2>&1; then
        python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null \
            && ok "CUDA torch available" \
            || warn "GPU present but torch is CPU-only (re-run setup_env.sh on a GPU node)"
    else
        warn "no GPU on this node (expected on login node)"
    fi
    [ -d third_party/WavTokenizer ] && ok "WavTokenizer cloned" || warn "third_party/WavTokenizer missing"
}

case "$STAGE" in
    data)    check_stage_data ;;
    pairs)   check_stage_pairs ;;
    tokens)  check_stage_tokens ;;
    env)     check_stage_env ;;
    all)     check_stage_env; check_stage_data; check_stage_pairs; check_stage_tokens ;;
    *) echo "Usage: $0 [env|data|pairs|tokens|all]"; exit 2 ;;
esac

exit $fail
