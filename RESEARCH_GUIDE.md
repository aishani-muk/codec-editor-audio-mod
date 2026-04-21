# Research Guide: Low-Latency Codec-to-Codec Streaming Raga Modulation with Stress-Proxy Neurofeedback

## Table of Contents
1. [The Big Picture](#1-the-big-picture)
2. [Stress Proxy: What It Is and SOTA Options](#2-stress-proxy)
3. [The Complete Data Architecture](#3-data-architecture)
4. [The Pipeline: End to End](#4-pipeline-end-to-end)
5. [Why Embeddings and Transformers for Audio ML](#5-why-embeddings)
6. [Indian Classical Music: Raga Yaman Specifics](#6-raga-yaman)
7. [Baselines and How to Beat Them](#7-baselines)
8. [Training and Evaluation Workflow](#8-training-eval)
9. [NEXUS Cluster Execution Guide](#9-nexus)
10. [Key References](#10-references)

---

## 1. The Big Picture <a name="1-the-big-picture"></a>

### What this system does
A listener is playing a Hindustani classical raga (Yaman). A **stress signal** is measured (or simulated) in real time — imagine brainwaves from an EEG headset, or heart-rate variability from a wearable. When the signal indicates elevated stress/arousal, the system **edits the audio stream in place** to gently steer the music toward a calmer ("neutral") profile, then eases off as the listener returns to baseline. The audio never stops, never regenerates from scratch — it is the *same* continuous raga, with bounded acoustic modifications applied in real time at low latency.

### Why this is hard (and novel)
- **Low latency**: The edit loop must close in <200 ms or the listener perceives lag.
- **Same-stream editing**: Unlike text-to-music generation (which creates new audio), this system *modifies* the existing stream token by token — codec-to-codec.
- **Cultural preservation**: Hindustani ragas have strict melodic grammar (ascending/descending phrases, characteristic glides called *meend*, a fixed tonic). The editor must preserve these while still modulating affect.
- **Continuous control**: The stress proxy is a time-varying signal, not a one-shot label. The editor must respond smoothly and continuously.

### The closed-loop concept
```
┌─────────────────────────────────────────────────┐
│                   LISTENER                       │
│  (wearing EEG headset / wearable / simulated)    │
└───────────┬───────────────────────▲──────────────┘
            │ physiological signal  │ modulated audio
            ▼                       │
   ┌────────────────┐      ┌───────┴────────┐
   │ Stress Proxy   │      │ Audio Output    │
   │ s(t) ∈ [0,1]   │──►   │ (same stream,  │
   │ (WESAD / DEAP  │      │  bounded edits) │
   │  model or sim) │      └───────▲────────┘
   └────────┬───────┘              │
            │ u(t) = clip(a·s+b)   │
            ▼                       │
   ┌────────────────────────────────┴──┐
   │  Codec-to-Codec Streaming Editor  │
   │  WavTokenizer → BPE → Transformer │
   │  → BPE decode → WavTokenizer dec  │
   │  → overlap-add crossfade          │
   └───────────────────────────────────┘
```

---

## 2. Stress Proxy: What It Is and SOTA Options <a name="2-stress-proxy"></a>

### What is a stress proxy?
A stress proxy is a **continuous scalar signal s(t) ∈ [0, 1]** that estimates the listener's current stress/arousal level in real time. In a deployed system this comes from physiological sensors; for research, we can either:
1. **Use a pre-recorded dataset** (WESAD, DEAP) to train a stress estimator, then run it on simulated sensor streams.
2. **Generate synthetic stress trajectories** that mimic realistic onset/decay profiles for controlled experiments.

### SOTA Open-Source Stress Datasets

#### WESAD (Wearable Stress and Affect Detection) — **Recommended primary source**
- **What**: 15 subjects, chest + wrist sensors (ECG, EDA, EMG, respiration, temperature, accelerometer), lab-induced stress (Trier Social Stress Test) vs. baseline vs. amusement.
- **Labels**: 3 affective states — *baseline*, *stress*, *amusement* (+ meditation for some subjects).
- **Why it's ideal**: Provides real physiological time series with ground-truth stress labels. You can train a simple classifier (Random Forest or small LSTM) on WESAD features to produce a continuous stress probability s(t).
- **Access**: Free at [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/WESAD) or [Kaggle](https://www.kaggle.com/datasets/orvile/wesad-wearable-stress-affect-detection-dataset).
- **Key paper**: Schmidt et al., "Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection," ICMI 2018.

#### DEAP (Database for Emotion Analysis using Physiological Signals)
- **What**: 32 subjects, 32-channel EEG + peripheral signals, watching 40 one-minute music videos. Rated on valence, arousal, dominance, liking.
- **Why useful**: The valence/arousal ratings are continuous and music-evoked — directly relevant to your use case. The arousal dimension maps to stress-like activation.
- **Access**: [DEAP homepage](http://eecs.qmul.ac.uk/mmv/datasets/deap/) (requires registration).
- **Key paper**: Koelstra et al., "DEAP: A Database for Emotion Analysis using Physiological Signals," IEEE TAC 2012.

#### How to build the stress proxy model

**Option A: WESAD-trained binary classifier (simplest, recommended for v1)**
```
1. Load WESAD pickle files (15 subjects, chest sensor data)
2. Extract sliding-window features:
   - ECG: HRV metrics (RMSSD, pNN50, LF/HF ratio) via pyHRV or NeuroKit2
   - EDA: skin conductance level (SCL), skin conductance responses (SCRs)
   - Respiration: breathing rate, depth
3. Train a Random Forest or Gradient Boosted classifier:
   - Input: feature vector per 4-second window
   - Output: P(stress) ∈ [0, 1]  ← this IS your s(t)
4. For your audio pipeline, run this model on sliding windows
   to produce a continuous stress trajectory
```

**Option B: Synthetic stress trajectory (for controlled ablation)**
```python
# Already implemented in models/stress_proxy.py
# Generates a ramp-up at a specified onset time,
# sustained plateau, and optional decay.
# Useful for reproducible experiments where you
# control exactly when and how much stress is applied.
```

**Option C: DEAP arousal regression (music-specific)**
```
1. Use DEAP's per-second arousal annotations
2. Train a small LSTM on EEG band-power features → continuous arousal
3. Threshold/normalize to [0,1] as s(t)
4. Advantage: trained on music-listening data specifically
```

### The mapping from s(t) to edit intensity u(t)
```
u(t) = clip(a · s(t) + b, 0, 1)
```
- `a` and `b` are tunable gain/offset parameters.
- `u(t) = 0` → no modification (passthrough)
- `u(t) = 1` → maximum calming edit
- Typical operating point: `u = 0.6` for moderate modulation

### Why this simulates brainwave control
In the full deployed system, `s(t)` would come from a live EEG headset (e.g., Emotiv EPOC, OpenBCI). The WESAD/DEAP models serve as a **validated proxy** — they've been shown to accurately detect stress from physiological signals. The key insight from Ehrlich et al. (2019, "A closed-loop music-based BCI for emotion mediation") is that users can **intentionally modulate** their own brain signals to steer the music, creating a genuine neurofeedback loop. Your system replaces their MIDI-based music generation with a codec-to-codec editor that preserves the original raga.

---

## 3. The Complete Data Architecture <a name="3-data-architecture"></a>

### Datasets you need

| Dataset | Role | What you get | Access |
|---------|------|-------------|--------|
| **Saraga Hindustani** | Evaluation & fine-tuning audio | Yaman raga recordings + tonic (.tonic files, Hz) + pitch tracks (.pitch, time-Hz pairs) + raga metadata | [Zenodo](https://zenodo.org/record/4301737), [GitHub](https://github.com/MTG/saraga) |
| **WESAD** | Stress proxy training | 15 subjects × {ECG, EDA, EMG, resp, temp, accel} with stress/baseline/amusement labels | [UCI ML](https://archive.ics.uci.edu/ml/datasets/WESAD), [Kaggle](https://www.kaggle.com/datasets/orvile/wesad-wearable-stress-affect-detection-dataset) |
| **DEAM** | Valence/arousal evaluation | ~1800 songs with per-song and per-second valence/arousal annotations (1-9 scale) | [DEAM homepage](https://cvml.unige.ch/databases/DEAM/) |
| **DEAP** (optional) | Music-specific arousal model | 32-ch EEG + peripherals during music listening, V/A/D/L ratings | [DEAP homepage](http://eecs.qmul.ac.uk/mmv/datasets/deap/) |

### Why Saraga specifically for raga Yaman?
- Saraga is the **largest open dataset** for Indian art music research (Srinivasamurthy et al., 2021).
- Each Hindustani recording includes **tonic frequency** (the Sa, the absolute pitch reference) and **pitch tracks** (time-aligned fundamental frequency).
- The file-path CSV has raga labels, so filtering for Yaman is straightforward.
- Yaman is an ideal test case: it's one of the most performed evening ragas, with a well-defined ascending/descending structure (*aaroha/avroha*), characteristic *teevra Ma* (augmented 4th), and prominent use of smooth glides (*meend*) that test the editor's preservation capability.

### Data flow through the pipeline
```
Saraga Yaman WAVs
        │
        ▼
  prepare_pairs.py     ──► Synthetic paired edits at λ ∈ {0.0, 0.2, ..., 1.0}
        │                    (original WAV + "calmed" version)
        ▼
  encode_wavtokenizer.py ──► .npy discrete codes (1 codebook, 40 tok/s)
        │
        ▼
  train_bpe.py          ──► BPE tokenizer vocabulary (8192 merges)
        │
        ▼
  apply_bpe.py          ──► Compressed BPE token sequences (.npy)
        │
        ▼
  train.py              ──► Train codec-to-codec transformer editor
        │                    (input BPE tokens + stress embed → edited BPE tokens)
        ▼
  evaluate.py           ──► Tonic drift, JSD, velocity TV, jerk RMS
```

---

## 4. The Pipeline: End to End <a name="4-pipeline-end-to-end"></a>

### Stage 1: Tokenization (WavTokenizer)
**What it does**: Converts a 24 kHz waveform into a sequence of discrete token IDs from a learned codebook of 4096 entries, at just 40 tokens per second.

**Why WavTokenizer over EnCodec/SoundStream**:
- EnCodec at comparable quality uses 4–8 codebooks × 75 frames/sec = 300–600 tokens/sec.
- WavTokenizer achieves comparable perceptual quality with **a single codebook at 40 tokens/sec** (ICLR 2025).
- Fewer tokens = shorter sequences = faster transformer inference = lower latency.

**Checkpoint**: `novateur/WavTokenizer-large-unify-40token` on HuggingFace.

### Stage 2: Token Compression (Acoustic BPE)
**What it does**: Applies byte-pair encoding to the discrete token stream, merging frequently co-occurring token pairs into single tokens. This further compresses the sequence.

**Why BPE on top of an already-compressed codec**:
- Even at 40 tok/s, a 30-second clip = 1200 tokens. Transformer attention is O(n²).
- BPE typically achieves K/T ≈ 0.7 (30% reduction), bringing 1200 → ~840 tokens.
- This is the same principle as subword tokenization in NLP (GPT uses BPE on text) — find repeated patterns and represent them as single units.

**Implementation**: `codec-bpe` library (pip install codec-bpe), which natively supports WavTokenizer's single-codebook output.

### Stage 3: Conditional Editor (Transformer)
**What it does**: A GPT-2-style decoder-only transformer that takes:
- Input: BPE-compressed token sequence from the original audio
- Conditioning: stress-proxy embedding u(t) added to the input embeddings
- Output: edited BPE token sequence that, when decoded, sounds "calmer"

**Architecture choices**:
- 6 layers, 8 heads, d_model=512 (~25M parameters) — small enough for real-time
- Operates in **overlapping windows**: L=2.0s window, H=0.5s hop
- Uses **role embeddings** (0=input, 1=target) so the model knows which tokens are source vs. expected output

### Stage 4: Streaming Decode + Overlap-Add
**What it does**:
1. BPE detokenize the edited tokens back to raw codec codes
2. WavTokenizer decode codes → waveform
3. Overlap-add with cosine crossfades between adjacent windows to prevent clicks

**Why crossfaded overlap-add**:
- Each window is processed independently, so there are discontinuities at boundaries.
- Cosine crossfading smoothly blends overlapping regions, eliminating audible clicks.
- This is the same technique used in STFT-based audio processing (e.g., phase vocoder).

---

## 5. Why Embeddings and Transformers for Audio ML <a name="5-why-embeddings"></a>

### Why turn audio into discrete tokens at all?

**The fundamental problem**: Raw audio at 24 kHz = 24,000 samples per second. A 30-second clip = 720,000 floating-point values. No transformer can attend over 720K positions efficiently.

**The solution (neural audio codecs)**:
1. An encoder neural network compresses the waveform into a low-dimensional continuous representation.
2. A vector quantizer (VQ) maps each continuous frame to the **nearest entry in a learned codebook** — producing a discrete token ID.
3. A decoder neural network reconstructs audio from these token IDs.

This is analogous to how text works: natural language is already discrete (words/characters), so LLMs operate on token sequences naturally. Audio codecs **make audio discrete** so that the same powerful sequence-modeling techniques (transformers, attention) can be applied.

**WavTokenizer's codebook**: 4096 entries (like a "vocabulary" of 4096 audio phonemes). Each token represents ~25 ms of audio (at 40 tok/s). The codebook is learned during training to capture the most useful acoustic patterns.

### What are embeddings and why do we need them?

An **embedding** is a learned dense vector representation of a discrete item. When the transformer sees token ID 2847, it looks up `embedding_table[2847]` to get a 512-dimensional vector. This vector is what the transformer actually processes.

**Why not just feed the raw token IDs?**
- Token IDs are arbitrary integers with no inherent relationships.
- Embeddings place semantically similar tokens nearby in vector space — the model learns that token 2847 (a certain vowel-like sound) is more related to token 1203 (a similar sound) than to token 3999 (percussion).
- This learned geometry is what makes transformers powerful: attention patterns operate in this continuous space.

### Why the stress proxy needs an embedding too

The stress proxy `u(t)` is a continuous scalar ∈ [0, 1]. We could just concatenate it directly to the token embedding, but:
1. **Bucketed embedding** (our approach): Quantize u into 64 buckets → look up a learned 64-dim vector → project to model dimension.
2. This gives the model a **richer, learnable representation** of each stress level.
3. The model can learn non-linear relationships (e.g., "stress at 0.3 means barely touch the high frequencies; stress at 0.8 means also shift pitch down").
4. It's the same principle as positional embeddings in standard transformers — a continuous signal (position or stress level) is made into a learned vector.

### Why GPT-2 architecture for codec editing?

**Decoder-only transformers** (GPT-style) are ideal for this task because:
1. **Autoregressive generation**: The model predicts one token at a time, conditioned on all previous tokens. This is exactly what we need for streaming — predict the next edited token given the input context.
2. **KV-cache**: During inference, past computations are cached so each new token only requires one forward pass, not reprocessing the entire sequence. Critical for low latency.
3. **Proven at scale**: GPT-2 with 6 layers is tiny by modern standards (~25M params), well within real-time inference budget on a single GPU.
4. **Not encoder-decoder**: We don't need a separate encoder because the input and output are the same modality (audio tokens). A single decoder with role embeddings suffices.

### The training objective

The model is trained with **teacher forcing** on paired data:
```
Input:  [input_token_1, input_token_2, ..., input_token_T, target_token_1, target_token_2, ...]
Labels: [-100, -100, ..., -100, target_token_1, target_token_2, ..., target_token_T']
```
- `-100` means "ignore this position in the loss" (standard PyTorch cross-entropy convention).
- The model learns to predict edited tokens given input tokens + stress conditioning.
- This is exactly how language models are fine-tuned on (prompt, completion) pairs.

---

## 6. Indian Classical Music: Raga Yaman Specifics <a name="6-raga-yaman"></a>

### Why raga Yaman?
- **Most commonly taught/performed** evening raga — well-documented and widely recorded.
- **Clear structure**: Aaroha (ascending) uses all *teevra* (augmented) Ma; Avroha (descending) is the natural Kalyan scale.
- **Rich in meend (glides)**: Yaman phrases frequently slide between notes, making it a demanding test for pitch continuity preservation.
- **Available in Saraga**: Multiple high-quality recordings with tonic + pitch annotations.

### What the editor must NOT break
1. **Tonic stability**: The Sa (tonic) must remain unchanged — it's the gravitational center of the raga.
2. **Pitch-class distribution**: Yaman has a characteristic histogram shape (strong Ga, teevra Ma, Pa, Ni). The octave-folded pitch histogram must be preserved.
3. **Meend continuity**: Smooth glides between notes (especially Re→Ga, Pa→Dha→Ni→Sa) must not become jerky or quantized.
4. **Raga grammar**: Certain note combinations and phrases define Yaman. The editor should NOT introduce forbidden note patterns.

### What the editor CAN change
- **Dynamic range**: Gentle compression to reduce peaks (calming effect).
- **Spectral balance**: Subtle high-frequency reduction (less "bright" = less arousing).
- **Micro-pitch**: Very slight downward pitch tendency (lower pitch = lower perceived arousal).
- **Tempo nuance**: Barely perceptible slowing (within raga's natural tempo variation).

### The metrics and why each matters
| Metric | What it measures | Acceptable range | Why |
|--------|-----------------|------------------|-----|
| Tonic drift (cents) | Shift in estimated tonic frequency | < 15 cents | Human pitch discrimination threshold in Indian classical music |
| JSD (pitch histogram) | Change in raga's pitch-class fingerprint | < 0.05 | Above this, listeners may perceive a different raga |
| Velocity TV | Abruptness of pitch changes | < 15% increase | Measures whether meend becomes jerky |
| Jerk RMS | Second-order pitch roughness | < 15% increase | Detects micro-stuttering in glides |

---

## 7. Baselines and How to Beat Them <a name="7-baselines"></a>

### Baseline 1: DSP-only (pedalboard)
- **What**: Parametric EQ + compressor + pitch shift, all conditioned on u(t).
- **Strengths**: Zero latency, deterministic, no training needed.
- **Weaknesses**: Cannot make context-dependent decisions. Applies the same EQ curve regardless of which raga phrase is playing. No understanding of musical structure.
- **Expected metrics**: Higher tonic drift (~9 cents), higher JSD (~0.058), more glide disruption.

### Baseline 2: EnCodec + MLP
- **What**: EnCodec tokenizer (multi-codebook, higher token rate) + simple feed-forward MLP for token editing.
- **Strengths**: Uses a codec (better than raw DSP), learnable.
- **Weaknesses**: EnCodec produces much longer token sequences (300–600/sec vs. 40/sec). MLP has no attention mechanism — cannot capture long-range dependencies in raga phrases.
- **Expected metrics**: Even higher tonic drift (~11 cents), JSD (~0.071).

### Why the proposed pipeline wins
1. **WavTokenizer** compresses 15× more aggressively than EnCodec → shorter sequences → transformer can see more musical context per window.
2. **BPE** further reduces sequence length → faster inference.
3. **Transformer** has attention → can learn that "when you're in the middle of a Ga→Ma meend, don't modify pitch, only dynamics" — context-dependent editing that DSP and MLP cannot do.
4. **Overlap-add** with cosine crossfades eliminates boundary artifacts.

---

## 8. Training and Evaluation Workflow <a name="8-training-eval"></a>

### Phase 1: Data Preparation
```bash
# Download Saraga Yaman recordings + annotations
python data/download_saraga.py --raga yaman --output data/saraga_kalyan_thaat/

# Download DEAM for valence/arousal evaluation
python data/download_deam.py --output data/deam/

# Download WESAD for stress proxy training (manual from Kaggle)
# Place in data/wesad/

# Generate synthetic paired edits
python data/prepare_pairs.py --audio_dir data/saraga_kalyan_thaat/ --output data/paired_edits/
```

### Phase 2: Tokenization
```bash
# Encode all audio with WavTokenizer
python tokenization/encode_wavtokenizer.py --input data/paired_edits/input/ --output data/tokens/input_wavtok/
python tokenization/encode_wavtokenizer.py --input data/paired_edits/target/ --output data/tokens/target_wavtok/

# Train BPE tokenizer
python tokenization/train_bpe.py --codes_dir data/tokens/input_wavtok/ --output data/tokens/bpe_model/

# Apply BPE compression
python tokenization/apply_bpe.py --codes_dir data/tokens/input_wavtok/ --bpe_model data/tokens/bpe_model/ --output data/tokens/input_bpe/
python tokenization/apply_bpe.py --codes_dir data/tokens/target_wavtok/ --bpe_model data/tokens/bpe_model/ --output data/tokens/target_bpe/
```

### Phase 3: Training
```bash
# Train proposed pipeline
python train.py --config configs/proposed.yaml --run_name proposed_v1

# Monitor in TensorBoard
tensorboard --logdir checkpoints/proposed_v1/
```

### Phase 4: Evaluation
```bash
# Evaluate proposed pipeline
python evaluate.py --input data/saraga_kalyan_thaat/ --output results/proposed_v1/ --tonic_dir data/saraga_kalyan_thaat/

# Evaluate DSP baseline
python baselines/dsp_baseline.py --input data/saraga_kalyan_thaat/ --output results/baseline_dsp/ --u 0.6
python evaluate.py --input data/saraga_kalyan_thaat/ --output results/baseline_dsp/ --tonic_dir data/saraga_kalyan_thaat/

# Evaluate EnCodec+MLP baseline
python baselines/encodec_mlp_baseline.py --input data/saraga_kalyan_thaat/ --output results/baseline_encodec_mlp/ --u 0.6
python evaluate.py --input data/saraga_kalyan_thaat/ --output results/baseline_encodec_mlp/ --tonic_dir data/saraga_kalyan_thaat/
```

### Phase 5: Ablation Studies
```bash
# Ablation: BPE on/off
python train.py --config configs/proposed_no_bpe.yaml --run_name proposed_no_bpe_v1

# Ablation: u(t) sweep
for u in 0.0 0.2 0.4 0.6 0.8 1.0; do
  python evaluate.py --input data/saraga_kalyan_thaat/ --output results/u_sweep_$u/ --u $u
done

# Ablation: window/hop
# (modify configs/proposed_window_sweep.yaml for each L/H combination)
```

---

## 9. NEXUS Cluster Execution Guide <a name="9-nexus"></a>

### Directory setup on NEXUS
```bash
# Clone your repo to the cluster
cd /scratch/$USER/
git clone <your-repo-url> codec-modulation
cd codec-modulation/modelling

# Create conda environment
conda create -n raga-mod python=3.11
conda activate raga-mod
pip install -r requirements.txt

# Install WavTokenizer (clone the repo for custom loading)
git clone https://github.com/jishengpeng/WavTokenizer.git third_party/WavTokenizer
```

### SLURM job scripts

**Data preparation (CPU, 1 hour)**:
```bash
#!/bin/bash
#SBATCH --job-name=raga-data
#SBATCH --output=logs/data_%j.out
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4

conda activate raga-mod
python data/prepare_pairs.py --audio_dir data/saraga_kalyan_thaat/ --output data/paired_edits/
```

**Tokenization (GPU, 30 min)**:
```bash
#!/bin/bash
#SBATCH --job-name=raga-tokenize
#SBATCH --output=logs/tokenize_%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00

conda activate raga-mod
python tokenization/encode_wavtokenizer.py --input data/paired_edits/input/ --output data/tokens/input_wavtok/
python tokenization/encode_wavtokenizer.py --input data/paired_edits/target/ --output data/tokens/target_wavtok/
python tokenization/train_bpe.py --codes_dir data/tokens/input_wavtok/ --output data/tokens/bpe_model/
python tokenization/apply_bpe.py --codes_dir data/tokens/input_wavtok/ --bpe_model data/tokens/bpe_model/ --output data/tokens/input_bpe/
python tokenization/apply_bpe.py --codes_dir data/tokens/target_wavtok/ --bpe_model data/tokens/bpe_model/ --output data/tokens/target_bpe/
```

**Training (GPU, 12 hours)**:
```bash
#!/bin/bash
#SBATCH --job-name=raga-train
#SBATCH --output=logs/train_%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00

conda activate raga-mod
python train.py --config configs/proposed.yaml --run_name proposed_v1
```

### Checkpoint organization
Every run creates:
```
checkpoints/<run_name>/
├── config.yaml          # Frozen config (reproducibility)
├── step_2000/model.pt   # Periodic checkpoints
├── step_4000/model.pt
├── best/model.pt        # Best by validation loss
└── training_log.jsonl   # Per-step loss, LR, etc.
```

---

## 10. Key References <a name="10-references"></a>

### Audio Codecs & Tokenization
- **WavTokenizer** (Ji et al., ICLR 2025): Ultra-low token rate codec. [GitHub](https://github.com/jishengpeng/WavTokenizer)
- **Acoustic BPE** (Shen et al., ICASSP 2024): BPE for discrete audio tokens. [arXiv:2310.14580](https://arxiv.org/abs/2310.14580)
- **codec-BPE** (Sanders): BPE extended for RVQ codecs. [GitHub](https://github.com/AbrahamSanders/codec-bpe)
- **EnCodec** (Défossez et al., 2022): Multi-codebook baseline codec. [GitHub](https://github.com/facebookresearch/encodec)

### Stress & Emotion Detection
- **WESAD** (Schmidt et al., ICMI 2018): Wearable stress dataset. [UCI ML](https://archive.ics.uci.edu/ml/datasets/WESAD)
- **DEAP** (Koelstra et al., IEEE TAC 2012): EEG emotion during music. [Homepage](http://eecs.qmul.ac.uk/mmv/datasets/deap/)
- **DEAM** (Aljanaki et al., 2017): Music emotion annotations. [Homepage](https://cvml.unige.ch/databases/DEAM/)

### Closed-Loop Music BCI
- **Ehrlich et al. (2019)**: "A closed-loop, music-based BCI for emotion mediation." [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6422328/) — **The closest prior work to your system.**
- **witheFlow** (Dervakos et al., 2025): Real-time emotion-driven audio effects. [arXiv:2510.02171](https://arxiv.org/abs/2510.02171)

### Indian Classical Music
- **Saraga** (Srinivasamurthy et al., 2021): Open datasets for Indian art music. [GitHub](https://github.com/MTG/saraga)
- **Koduri et al.**: Raga recognition via tonic-aligned pitch histograms. [PDF](https://repositori.upf.edu/bitstream/10230/32476/1/koduri_jnmr_raga.pdf)

### HRV & Physiological Signal Processing
- **NeuroKit2**: Python toolbox for neurophysiological signal processing. [GitHub](https://github.com/neuropsychology/NeuroKit)
- **pyHRV**: HRV analysis toolbox. [GitHub](https://github.com/PGomes92/pyhrv)
- **WearableHRV**: Wearable HRV processing. [JOSS paper](https://joss.theoj.org/papers/10.21105/joss.06240)
