# Pipeline — streaming codec-token editor

Self-contained deliverable. Demo WAVs, A/B HTML, trained checkpoints,
and bootstrap-CI numbers for both the **raga arm** (Saraga Hindustani)
and the **Celtic arm** (Jamendo) of the project.

## Layout

```
pipeline/
├── README.md
├── requirements.txt              # main + celtic env deps
├── run_demo.sh                   # regenerate raga DSP demo WAVs
├── reproduce_eval.sh             # rerun lora-eval u-collapse check
├── regenerate_demo.py            # python entry point
│
├── demo/                         # streaming pipeline wrapper + u(t) profiles
├── baselines/                    # DSP back-end (EQ + comp + pitch shift)
├── models/                       # codec_editor (GPT-2), musicgen_lora, stress_proxy
├── tokenization/                 # encode_wavtokenizer
├── evaluation/                   # metrics
├── infer_stream_musicgen.py      # batch lora-edit driver
├── evaluate.py                   # bootstrap-CI metrics
├── configs/
│   ├── editor_v3_lora.yaml       # raga arm LoRA-MusicGen config
│   └── proposed.yaml             # streaming pipeline config
│
├── checkpoints/
│   ├── editor_v3/best/           # 87 MB - Saraga GPT-2 codec editor (best.pt at step 2000)
│   ├── editor_v3_lora/best/      # 25 MB - Saraga LoRA-MusicGen adapter
│   └── celtic_track_a/best.pt    # 87 MB - Celtic GPT-2 codec editor
│
├── data/
│   ├── demo_inputs/              # 3 Saraga raga clips (Yaman, Todi, Shree)
│   └── celtic_inputs/            # 3 Jamendo Celtic clips (jig, reel, jig)
│
├── results/
│   ├── ab_demo.html              # raga A/B (DSP edit at ramp + pulse u(t))
│   ├── celtic_demo.html          # celtic A/B (Track A edit at u=0.0/0.3/0.6/0.9)
│   ├── celtic_panels.png         # bootstrap-CI panels across A/B/C tracks
│   ├── celtic_ab_pairs.html      # per-clip celtic input/output table
│   ├── demo_clips/               # 6 raga DSP-edited WAVs (ramp + pulse)
│   ├── celtic_demo_clips/        # 12 celtic Track A-edited WAVs (3 clips × 4 u)
│   ├── saraga_eval/              # Saraga eval bootstrap CIs (n=30)
│   │   ├── editor_v3_eval/u{0.0,0.3,0.6,0.9}/eval_results.json
│   │   ├── editor_v3_lora/u{0.0,...}/eval_results.json
│   │   ├── rescue_v3_table*.md   # human-readable summaries
│   │   ├── rescue_v3_success_gate*.md  # PASS/FAIL gate verdicts
│   │   └── rescue_v3_panels.png  # ΔA/drift/jsd vs u figures
│   ├── celtic_eval_track_a/      # n=29 bootstrap CI per (u, metric)
│   ├── celtic_eval_track_b/      # MusicGen LoRA - silent passthrough (failure)
│   └── celtic_eval_track_c/      # hybrid encoder - passthrough (no generate())
│
└── third_party/                  # Music2Emotion + WavTokenizer (symlinks)
```

## What's deliverable as-is (no env needed)

- `results/ab_demo.html` — raga DSP demo (3 clips × 2 u-profiles)
- `results/celtic_demo.html` — celtic Track A demo (3 clips × 4 u-levels)
- `results/celtic_panels.png` — bootstrap-CI figure
- `results/celtic_eval_track_*/u*/summary.json` — full numerical results

Open the two HTMLs in a browser; everything is self-contained.

## Headline numbers (bootstrap 95% CI at u=0.6)

### Saraga test set (n=30, primary domain — `RESEARCH_GUIDE.md` focus)

| system | ΔA | drift c | jsd | vel_tv % | verdict |
|---|---|---|---|---|---|
| `baseline_dsp_v3` | **+0.32** | 67.5 | 0.027 | +10 | FAIL (wrong direction) |
| `editor_v3_lora` (Saraga LoRA-MusicGen) | **−0.05** | 1.7 | 0.000 | +0.01 | FAIL (u-collapsed; bit-identical MD5s) |
| `editor_v3` (Saraga GPT-2 codec editor) | **−1.35** | 292 | 0.38 | +128 | FAIL (right direction, breaks structure) |
| `celtic_track_a` (Celtic-trained, on Saraga) | **+0.47** | 0 | 0.021 | +100 | FAIL (wrong direction; tonic preserved) |

Plan gate: `ΔA ≤ −0.40, drift ≤ 100c, jsd ≤ 0.05, |vel_tv| ≤ 50`.

### Celtic test set (n=29, cross-domain validation)

| system | ΔA | drift c | jsd | vel_tv % | meter_match | verdict |
|---|---|---|---|---|---|---|
| `celtic_track_a` | **−1.55** | (n/a) | 0.022 | +106 | 0.45 | FAIL (u-collapsed; vel_TV breached) |
| `celtic_track_b` (LoRA) | 0.00 | — | 0.000 | 0.00 | 1.00 | FAIL (silent passthrough — debug pending) |
| `celtic_track_c` (hybrid) | 0.00 | — | 0.000 | 0.00 | 1.00 | n/a (no `generate()` — passthrough by design) |

### Three failure modes, one infrastructure

- `editor_v3_lora`: doesn't edit at all (T5 prompt collapse).
- `editor_v3`: edits in correct direction on Saraga, breaks raga structure.
- `celtic_track_a` on Saraga: edits, wrong direction; on Celtic: edits, right direction. Same model, sign flips.
- DSP baseline: edits, wrong direction (mild) — likely M2E Western-pop bias on Hindustani vocal.

Full bootstrap CI tables: `results/saraga_eval/` and `results/celtic_eval_track_*/`.
Pros/cons writeup: `../modelling/RESULTS.md §4b`.

## Regenerating from scratch (optional)

The main `.venv` was used for the raga DSP demo. To regenerate
`results/demo_clips/*.wav`:

```bash
cd pipeline
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./run_demo.sh
```

The current 6 WAVs in `results/demo_clips/` are bit-identical to what
this command produces (DSP backend is deterministic).

For the Celtic regen, the trained `celtic_track_a/best.pt` plus
`/scratch0/$USER/celtic-venv` (UMIACS tron33-local) are the dependencies.
The 12 demo WAVs in `results/celtic_demo_clips/` were rendered by
`run_full_eval.py` during the celtic chain and are saved verbatim.

## What's running and what's not

- **Raga arm DSP demo**: works end-to-end. The DSP edit visibly
  modulates spectral balance + dynamics + pitch with u(t). Does NOT
  match M2E's notion of "calmer" on Hindustani vocal (M2E has Western
  pop bias; ΔA = +0.32 is wrong-direction by metric, but listeners
  hear muffling/softening = subjective calming).
- **Raga arm LoRA editor**: trained, u-blind. Output bit-identical
  across u ∈ {0.0, 0.3, 0.6, 0.9}. Likely cause: T5 collapses the
  "intensity 0.x" suffix to near-identical embeddings.
- **Celtic Track A editor**: trained, **edits aggressively** (ΔA ≈ −1.55,
  20× larger than lora's 0.05 on raga). Numerical u-embedding works as
  a switch but doesn't modulate magnitude — model still u-blind. Also
  breaks structure: vel_TV +106%, meter retention 45%.
- **Celtic Track B (LoRA)**: trained, but inference produces silent
  passthrough at every u — likely per-clip decode-fallback firing
  universally; needs debug.
- **Celtic Track C (hybrid)**: trained, but `HybridMusicGenEditor`
  has no `generate()` method, so `build_track_c_generator` returns
  passthrough by design.

The "infrastructure works, conditional generation is the open problem"
framing in `../modelling/RESULTS.md` is supported by these numbers.
