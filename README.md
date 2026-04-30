# Codec-token editor for music affect modulation

A real-time streaming pipeline (RTF < 0.12) that edits audio codec tokens under a continuous u(t) control signal, evaluated on Hindustani raga (Saraga, primary) and Celtic folk (Jamendo, cross-domain). Three editor variants on the same architecture surface three distinct conditioning patterns — silent passthrough, aggressive metric-direction edits with spectral drift, and cross-domain sign-flip. Quantified with bootstrap-CI metrics below.

## File Structure

```
pipeline/
├── README.md
├── requirements.txt
├── run_demo.sh                   # regenerate raga DSP demo WAVs
├── reproduce_eval.sh             # rerun the lora-eval u-collapse check
├── regenerate_demo.py            # python entry point for the DSP demo
│
├── demo/                         # streaming wrapper + u(t) profiles
├── baselines/                    # DSP back-end (EQ + comp + pitch shift)
├── models/                       # codec_editor (GPT-2), musicgen_lora, stress_proxy
├── tokenization/                 # encode_wavtokenizer
├── evaluation/                   # metrics
├── infer_stream_musicgen.py      # batch lora-edit driver
├── evaluate.py                   # bootstrap-CI metrics
│
├── configs/
├── checkpoints/
│   ├── editor_v3/best/           # 87 MB · Saraga GPT-2 codec editor
│   ├── editor_v3_lora/best/      # 25 MB · Saraga LoRA-MusicGen adapter
│   └── celtic_track_a/best.pt    # 87 MB · Celtic GPT-2 codec editor
│
├── data/
│   ├── demo_inputs/              # 3 Saraga clips (Yaman, Todi, Shree)
│   └── celtic_inputs/            # 3 Jamendo Celtic clips
│
├── results/
│   ├── demo.html                 # combined raga + Celtic A/B page
│   ├── ab_demo.html              # raga-only A/B (DSP at ramp + pulse)
│   ├── celtic_demo.html          # Celtic-only A/B (Track A across u)
│   ├── celtic_panels.png         # bootstrap-CI panels
│   ├── demo_clips/               # 6 raga DSP-edited WAVs
│   ├── celtic_demo_clips/        # 12 Celtic Track A-edited WAVs
│   ├── saraga_eval/              # n=30 bootstrap CI tables
│   └── celtic_eval_track_{a,b,c}/  # n=29 bootstrap CI tables
│
└── third_party/                  # Music2Emotion + WavTokenizer (symlinks)
```

## Results (bootstrap 95 % CI at u=0.6)

Plan-gate targets:
**ΔA ≤ −0.40 · drift ≤ 100 c · jsd ≤ 0.05 · |vel_tv| ≤ 50 %**.

### Saraga test set (n = 30 — primary domain)

| system | ΔA | drift c | jsd | vel_tv % | observation |
|---|---|---|---|---|---|
| `baseline_dsp_v3` | **+0.32** | 67.5 | 0.027 | +10 | edits modestly; M2E reads opposite-sign on Hindustani vocal |
| `editor_v3_lora` (Saraga LoRA-MusicGen) | **−0.05** | 1.7 | 0.000 | +0.01 | output is bit-identical across u ∈ {0.0, 0.3, 0.6, 0.9} |
| `editor_v3` (Saraga GPT-2 codec editor) | **−1.35** | 292 | 0.38 | +128 | calms aggressively in metric direction; substantial spectral / pitch drift |
| `celtic_track_a` (Celtic-trained, on Saraga) | **+0.47** | 0 | 0.021 | +100 | tonic preserved; M2E sign flips relative to Celtic test |

### Celtic test set (n = 29 — cross-domain)

| system | ΔA | jsd | vel_tv % | meter_match | observation |
|---|---|---|---|---|---|
| `celtic_track_a` | **−1.55** | 0.022 | +106 | 0.45 | u-magnitude does not modulate edit; meter retained on ~½ of clips |
| `celtic_track_b` (LoRA) | 0.00 | 0.000 | 0.00 | 1.00 | inference passthrough at every u |
| `celtic_track_c` (hybrid encoder→GPT-2) | 0.00 | 0.000 | 0.00 | 1.00 | no `generate()` wired; passthrough by design |

## Hypotheses for the observed behaviour

- **`editor_v3_lora` is u-blind**: T5 likely collapses the "intensity 0.x" text suffix to near-identical embeddings, so the ~3 M-param LoRA never receives a discriminating signal at conditioning time. The model learns "edit ≈ identity."
- **`editor_v3` calms in the metric direction but drifts**: the numerical u-embedding (bucketed → projected → added to token embeddings) gives the editor a strong conditioning signal, but absent an explicit raga-grammar regulariser the GPT-2 head trades pitch-class structure for arousal reduction. PCD-KL contributes (0.38 vs ~0.5 with the loss off) but isn't strong enough alone.
- **`celtic_track_a` sign-flips across domains**: same architecture trained on Celtic mood pairs produces metric-direction edits on Celtic and metric-opposite-direction edits on Saraga. Two non-exclusive hypotheses: (a) the Celtic-domain edit rule is acoustically wrong on Hindustani vocal; (b) M2E was trained on Western popular music and its arousal scorer is biased on Hindustani vocal — the same acoustic changes that read "calmer" in folk read "more aroused" in raga.
- **DSP baseline is consistent with hypothesis (b)**: parametric EQ + compressor + pitch-shift on Hindustani vocal sound subjectively softer to a human listener (less brightness, less peakiness) but M2E reports +0.32. A listener study would settle which side is right.
- **u is currently a switch, not a magnitude**: across the three trained editors, u-dependence ranges from "absent" (lora — bit-identical MD5s) to "non-monotone" (editor_v3: −0.79 → −0.68 → −1.35 → −1.17 across u 0.0/0.3/0.6/0.9) to "single-step" (celtic_track_a: −1.55 across all u). Smooth modulation of edit magnitude — required by the continuous- neurofeedback framing — has not yet been learned.

## Three observation patterns, one infrastructure

- `editor_v3_lora`: does not edit measurably (preservation perfect by vacuity).
- `editor_v3`: edits in metric direction; structural preservation is weak.
- `celtic_track_a`: domain-dependent sign of ΔA; structural preservation is intermediate (tonic kept, rhythm distorted).

These are independent modes of the same conditional-generation problem riding on a shared end-to-end streaming pipeline.

## What's solid

1. **Streaming infrastructure runs end-to-end.** WavTokenizer encode → BPE → conditional editor → cosine-crossfade overlap-add. Real-time factor < 0.12 on the demo clips (8× faster than playback).
2. **Three trained checkpoints, one architecture class.** `editor_v3` and `celtic_track_a` are both `models/codec_editor.py::CodecEditor`; the cross-domain comparison is therefore on identical model code.
3. **Cross-domain experiment is honest.** Same architecture, two cultural corpora, evaluated on identical inputs; the sign-flip finding is empirical, not a bug.
4. **Numerical u-embedding outperforms T5 text-suffix conditioning** for codec editors on the "produce any edit at all" criterion. `editor_v3` and `celtic_track_a` both edit; `editor_v3_lora` does not.
5. **Pipeline regenerates from scratch.** `python regenerate_demo.py` from `pipeline/` produces bit-identical raga DSP demo WAVs in under 5 seconds.

## What's open

- Tonic-stability target (< 15 cents) is exceeded by `editor_v3` (292–551 cents) and roughly met by `celtic_track_a` (0–small).
- PCD-JSD target (< 0.05) is met by `editor_v3_lora` and `celtic_track_a` (0.000 / 0.02) and exceeded by `editor_v3` (0.37–0.43).
- Velocity-TV target (< 15 % increase) is exceeded by both ML editors (+100 % to +141 %).
- Smooth u-modulation is not yet learned by any of the three editors.
- A v3 PCD-aware raga-preservation classifier was deferred; the raga-pres percentage column is not filled.
- Celtic Track B inference debug and Celtic Track C `generate()` implementation are recoverable next steps but not in this drop.

## Regenerating from scratch (optional)

```bash
cd pipeline
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./run_demo.sh
```

The current 6 WAVs in `results/demo_clips/` are bit-identical to what this command produces (DSP back-end is deterministic). The 12 Celtic WAVs in `results/celtic_demo_clips/` were rendered by `jamendo_pipeline/evaluation/run_full_eval.py` during the Celtic training chain and are saved verbatim.

## Pointers

- Combined demo: [`pipeline/results/demo.html`](pipeline/results/demo.html)
- Per-domain demos: [`pipeline/results/ab_demo.html`](pipeline/results/ab_demo.html), [`pipeline/results/celtic_demo.html`](pipeline/results/celtic_demo.html)
- Bootstrap-CI tables: [`pipeline/results/saraga_eval/`](pipeline/results/saraga_eval/), [`pipeline/results/celtic_eval_track_a/`](pipeline/results/celtic_eval_track_a/)
- Source pipeline (training + eval drivers): [`pipeline/`](pipeline/)
