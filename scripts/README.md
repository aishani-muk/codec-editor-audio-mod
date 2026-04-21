# Scripts

Operational scripts for the 1-week deliverable pipeline.

## Setup (one-time)

```bash
# Get onto a GPU node for the install (so CUDA torch is picked correctly)
srun -p class --gres=gpu:1 --mem=16G --pty bash
bash scripts/setup_env.sh
```

This creates `.venv/`, installs `requirements.txt`, clones WavTokenizer into
`third_party/`, and upgrades torch to the CUDA build when a GPU is visible.

## Pipeline order

Every SLURM script assumes `.venv` is already set up.

```bash
# 1. Data prep (CPU; can run on login node as a background job)
python data/filter_saraga.py --zip <path_to_zip> --output data/saraga_kalyan_thaat
python data/mp3_to_wav.py --data_dir data/saraga_kalyan_thaat --sr 24000
python data/extract_raga_features.py --data_dir data/saraga_kalyan_thaat
python data/prepare_pairs.py --audio_dir data/saraga_kalyan_thaat --num_workers 4

# 2. Tokenization (GPU)
sbatch scripts/slurm/slurm_tokenize_speech75.sh

# 3. Medium-smoke training (2000 steps) — MUST pass before big run
sbatch scripts/slurm/slurm_medium_smoke.sh
# inspect: logs/slurm_medsmoke_<jobid>.out
# expected: train info_frac_pct rising past 20% by step 2000

# 4. Full training (12 h budget)
sbatch scripts/slurm/slurm_train.sh proposed_v1

# 5. Plot loss / metric curves (local, quick)
python scripts/plot_run.py --log checkpoints/proposed_v1/training_log.jsonl

# 6. Inference + eval (proposed + baselines)
sbatch scripts/slurm/slurm_eval.sh proposed_v1
```

## Monitoring

```bash
# Job status
squeue -u $USER

# Tail a running job
tail -f logs/slurm_train_<jobid>.out

# Cancel
scancel <jobid>
```

## Common pitfalls

- **Forgetting to `source .venv/bin/activate`** — the SLURM scripts do this
  for you, but if you run commands manually you'll hit "ModuleNotFoundError".
- **Using `python3` instead of `python` after activating** — stick to `python`
  once the venv is active; `python3` may still point at the system interpreter.
- **Running heavy CPU jobs on the login node** — `prepare_pairs.py` can run
  here because it's ~26 min with 2 workers, but `encode_wavtokenizer.py`
  needs a GPU; use `sbatch scripts/slurm_tokenize.sh`.
- **`tokenize` vs `tokenization`** — the directory was renamed to avoid
  shadowing Python's stdlib `tokenize`, which breaks `matplotlib` at import.
