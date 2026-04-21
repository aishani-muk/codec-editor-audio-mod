# Codec-to-Codec Streaming Music Modulation — Implementation

## Overview

End-to-end pipeline for real-time audio modulation toward a "neutral" mood using a
stress-proxy control signal. Evaluated on Indian classical music preservation
(Saraga Hindustani, raga Yaman) and emotional shift (DEAM valence/arousal).

## Directory Layout

```
modelling/
├── README.md, RESEARCH_GUIDE.md   # Project docs + research walkthrough
├── requirements.txt               # All pip dependencies
├── train.py                       # Editor fine-tune (paired-edit objective)
├── pretrain_codec_lm.py           # Raw-codec LM pretraining stage
├── infer_stream.py                # Streaming inference demo
├── evaluate.py                    # Eval: tonic, PCD JSD, smoothness, DEAM, raga-id
├── configs/
│   ├── base.yaml                  # Shared hyperparams (speech-75, residual, raga)
│   ├── proposed.yaml              # 22M editor
│   ├── proposed_small.yaml        # 6M editor (recommended after smoke)
│   ├── pretrain_codec_lm.yaml     # Codec-LM pretrain
│   ├── baseline_dsp.yaml          # DSP-only baseline
│   ├── smoke/                     # Quick-validation configs
│   └── ablations/                 # abl_no_raga, abl_concat_arch, abl_unify40
├── data/
│   ├── download_saraga.py, download_deam.py, download_wesad.py
│   ├── filter_saraga.py, mp3_to_wav.py, extract_raga_features.py
│   ├── prepare_pairs.py           # Synthetic (input, target, λ) generator
│   └── test_clips/                # 2 held-out Yaman clips for eval
├── tokenization/
│   └── encode_wavtokenizer.py     # WAV → WavTokenizer speech-75 tokens
├── models/
│   ├── codec_editor.py            # Conditional codec-to-codec editor
│   ├── stress_proxy.py            # u(t) generator + embedding
│   └── overlap_add.py             # Streaming overlap-add crossfader
├── evaluation/                    # (added) MERT-based feature wrapper + helpers
├── baselines/
│   ├── dsp_baseline.py            # pedalboard EQ/dynamics/pitch
│   └── encodec_mlp_baseline.py    # EnCodec + linear token map
├── scripts/
│   ├── plot_run.py                # Live training-log dashboard
│   ├── preflight.sh, setup_env.sh, stage_test_clips.sh
│   └── slurm/                     # SLURM batch scripts
├── legacy/                        # Not on critical path (BPE, orchestrator, WESAD)
├── checkpoints/                   # Auto-populated during training
└── third_party/WavTokenizer/      # Upstream codec source
```

## Pipeline: Proposed vs Baseline

### Proposed Pipeline
1. **Tokenize**: WavTokenizer speech-75 (75 tok/s, single codebook, 4096 codes — vocal-optimised)
2. **Pretrain**: unconditional next-token LM on target-side codec tokens (`pretrain_codec_lm.py`) — gives the embedding a music-codec prior before paired fine-tuning
3. **Edit**: Conditional GPT-2 transformer — input tokens + stress u(t) + raga-label → edited tokens; trained on (input, target, λ) pairs with windowed crops matching the streaming inference window
4. **Decode**: WavTokenizer decode → overlap-add crossfade for gapless streaming

### Baseline: DSP
1. **Process**: pedalboard chain (parametric EQ, compressor, pitch shift) conditioned on u(t)
2. **No codec/ML**: direct waveform processing
3. **Purpose**: ablate whether the codec-to-codec approach outperforms classic DSP

### Baseline: EnCodec + MLP
1. **Tokenize**: EnCodec 24kHz at 6kbps (multi-codebook)
2. **Edit**: Shallow MLP mapping input tokens → edited tokens, conditioned on u(t)
3. **Decode**: EnCodec decode
4. **Purpose**: ablate whether WavTokenizer + BPE + transformer outperforms simpler codec editing

## Evaluation Metrics
- **Tonic drift** (cents, Hz) — Saraga tonic annotations
- **Pitch-histogram JSD** — tonic-normalized, octave-folded
- **Velocity TV / Jerk RMS** — glide smoothness proxies
- **DEAM Δvalence / Δarousal** — shift toward neutral (low-arousal) target
- **PESQ / UTMOS** — audio quality

## Checkpointing Strategy
All training runs save to `checkpoints/<run_name>/`:
```
checkpoints/
├── proposed_v1/
│   ├── config.yaml                # Frozen config for this run
│   ├── step_1000/                 # Periodic checkpoint
│   │   ├── model.safetensors
│   │   ├── optimizer.pt
│   │   └── scheduler.pt
│   ├── step_2000/
│   ├── best/                      # Best by val JSD
│   └── training_log.jsonl         # Per-step metrics
├── baseline_encodec_mlp_v1/
└── ...
```

## Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download & prepare data
python data/download_saraga.py --raga yaman --output data/saraga_kalyan_thaat/
python data/download_deam.py --output data/deam/
python data/prepare_pairs.py --config configs/proposed.yaml

# 3. Tokenize (GPU — use the SLURM script)
sbatch scripts/slurm/slurm_tokenize_speech75.sh

# 4. Codec-LM pretrain, then editor fine-tune
python pretrain_codec_lm.py --config configs/pretrain_codec_lm.yaml --run_name pretrain_v1
python train.py --config configs/proposed_small.yaml --run_name proposed_v1 \
    --pretrain_ckpt checkpoints/pretrain_v1/pretrain.pt

# 5. Smoke-validate loss trajectory before committing to 12 h
sbatch scripts/slurm/slurm_medium_smoke.sh

# 6. Evaluate proposed + DSP baseline
python evaluate.py --checkpoint checkpoints/proposed_v1/best/ --config configs/proposed_small.yaml
python baselines/dsp_baseline.py --config configs/baseline_dsp.yaml
```

> BPE-compressed tokens are disabled in `configs/base.yaml` (only ~5 % compression on music codes vs. ~40 % on speech). The BPE tooling survives in `legacy/tokenization/` for future ablations.

## Cluster (NEXUS) Usage
```bash
# Example SLURM submission
sbatch --gres=gpu:1 --mem=32G --time=12:00:00 \
  --wrap="python train.py --config configs/proposed.yaml --run_name proposed_v1"
```
