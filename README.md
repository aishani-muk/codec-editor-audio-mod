# Codec-to-Codec Streaming Music Modulation — Implementation

## Overview

End-to-end pipeline for real-time audio modulation toward a "neutral" mood using a
stress-proxy control signal. Evaluated on Indian classical music preservation
(Saraga Hindustani, raga Yaman) and emotional shift (DEAM valence/arousal).

## Directory Layout

```
modelling/
├── README.md                     # This file
├── requirements.txt              # All pip dependencies
├── configs/
│   ├── base.yaml                 # Shared hyperparams
│   ├── proposed.yaml             # Proposed pipeline config
│   └── baseline_dsp.yaml         # DSP baseline config
├── data/
│   ├── download_saraga.py        # Download Saraga Hindustani Yaman subset
│   ├── download_deam.py          # Download DEAM valence/arousal annotations
│   ├── prepare_pairs.py          # Generate synthetic paired edits for training
│   └── README.md                 # Data preparation instructions
├── tokenize/
│   ├── encode_wavtokenizer.py    # Encode WAVs → WavTokenizer discrete codes
│   ├── train_bpe.py              # Train codec-BPE tokenizer on encoded codes
│   └── apply_bpe.py              # Apply trained BPE to compress token sequences
├── models/
│   ├── codec_editor.py           # Conditional codec-to-codec transformer editor
│   ├── stress_proxy.py           # Stress-proxy signal generation & embedding
│   └── overlap_add.py            # Windowed inference + crossfaded overlap-add
├── baselines/
│   ├── dsp_baseline.py           # DSP baseline: pedalboard EQ/dynamics/pitch
│   └── encodec_mlp_baseline.py   # EnCodec + learned linear token map baseline
├── train.py                      # Main training loop (HF Trainer + checkpoints)
├── evaluate.py                   # Evaluation: tonic drift, JSD, smoothness, DEAM
├── infer_stream.py               # Streaming inference demo
└── checkpoints/                  # Auto-populated during training
    └── .gitkeep
```

## Pipeline: Proposed vs Baseline

### Proposed Pipeline
1. **Tokenize**: WavTokenizer-large-unify-40token (40 tok/s, single codebook, 4096 codes)
2. **Compress**: codec-BPE trained on Saraga + DEAM audio
3. **Edit**: Conditional GPT-2-small transformer (input BPE tokens + stress embedding → edited BPE tokens)
4. **Decode**: BPE detokenize → WavTokenizer decode → overlap-add crossfade

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
python data/download_saraga.py --raga yaman --output data/saraga_yaman/
python data/download_deam.py --output data/deam/
python data/prepare_pairs.py --config configs/proposed.yaml

# 3. Tokenize
python tokenize/encode_wavtokenizer.py --input data/saraga_yaman/ --output data/tokens/wavtok/
python tokenize/train_bpe.py --codes_dir data/tokens/wavtok/ --output data/tokens/bpe_model/
python tokenize/apply_bpe.py --codes_dir data/tokens/wavtok/ --bpe_model data/tokens/bpe_model/

# 4. Train
python train.py --config configs/proposed.yaml --run_name proposed_v1

# 5. Evaluate
python evaluate.py --checkpoint checkpoints/proposed_v1/best/ --config configs/proposed.yaml

# 6. Run baselines
python baselines/dsp_baseline.py --config configs/baseline_dsp.yaml
python evaluate.py --baseline dsp --config configs/baseline_dsp.yaml
```

## Cluster (NEXUS) Usage
```bash
# Example SLURM submission
sbatch --gres=gpu:1 --mem=32G --time=12:00:00 \
  --wrap="python train.py --config configs/proposed.yaml --run_name proposed_v1"
```
