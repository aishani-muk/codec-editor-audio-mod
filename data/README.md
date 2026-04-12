# Data Preparation

## Step-by-step Pipeline

### 1. Download Saraga Hindustani (Yaman subset)
```bash
python data/download_saraga.py --raga yaman --output data/saraga_yaman/
```
If programmatic download fails (needs Dunya API token), download manually from
[Zenodo](https://zenodo.org/record/4301737) and place Yaman recordings + tonic/pitch
annotations in `data/saraga_yaman/`.

### 2. Download DEAM annotations
```bash
python data/download_deam.py --output data/deam/
```
Audio must be obtained separately from [Kaggle](https://www.kaggle.com/datasets/imsparsh/deam-mediaeval-dataset-emotional-analysis-in-music).

### 3. Generate synthetic paired edits
```bash
python data/prepare_pairs.py --audio_dir data/saraga_yaman/ --output data/paired_edits/
```
Creates `input/`, `target/`, and `meta/` subdirectories with paired WAVs at λ ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}.

### 4. Tokenize with WavTokenizer
```bash
# Tokenize input audio
python tokenize/encode_wavtokenizer.py \
    --input data/paired_edits/input/ \
    --output data/tokens/input_wavtok/

# Tokenize target audio
python tokenize/encode_wavtokenizer.py \
    --input data/paired_edits/target/ \
    --output data/tokens/target_wavtok/
```

### 5. Train & apply codec-BPE
```bash
# Train BPE on all tokenized audio
python tokenize/train_bpe.py \
    --codes_dir data/tokens/input_wavtok/ \
    --output data/tokens/bpe_model/ \
    --vocab_size 8192

# Apply BPE to both input and target
python tokenize/apply_bpe.py \
    --codes_dir data/tokens/input_wavtok/ \
    --bpe_model data/tokens/bpe_model/ \
    --output data/tokens/input_bpe/

python tokenize/apply_bpe.py \
    --codes_dir data/tokens/target_wavtok/ \
    --bpe_model data/tokens/bpe_model/ \
    --output data/tokens/target_bpe/
```

## Expected Directory Structure After Preparation
```
data/
├── saraga_yaman/           # Raw Yaman recordings + annotations
│   ├── YMN-01.wav
│   ├── YMN-01.pitch
│   ├── YMN-01.tonic
│   └── ...
├── deam/                   # DEAM valence/arousal annotations
│   ├── annotations/
│   └── audio/
├── paired_edits/           # Synthetic paired data
│   ├── input/              # Original clips as WAV
│   ├── target/             # Transformed clips as WAV
│   └── meta/               # Lambda values + metadata
└── tokens/                 # Tokenized sequences
    ├── input_wavtok/       # WavTokenizer codes (.npy)
    ├── target_wavtok/
    ├── bpe_model/          # Trained BPE tokenizer
    ├── input_bpe/          # BPE-compressed codes (.npy)
    └── target_bpe/
```
