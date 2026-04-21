# legacy/

Scripts and configs that are not on the active critical path but are kept
for future optional use. Nothing in this directory is imported or invoked
by the current pipeline. Everything here ran at some point and is
preserved intentionally.

## Contents

- `run_pipeline.py` — end-to-end orchestrator that chained data prep →
  tokenization → editor training → evaluation. Predates the speech-75
  redesign and the `pretrain_codec_lm.py` stage. Reactivate by updating
  its stage list to include pretraining + continuous-λ pair generation.

- `tokenization/train_bpe.py`, `tokenization/apply_bpe.py` — codec-BPE
  tooling built on `codec-bpe`. We disabled BPE in `configs/base.yaml`
  because it compressed music codes by only ~5 % (vs. ~40 % for speech,
  which was the original motivation). Reactivate by flipping
  `bpe.enabled: true` in a config and running
  `python legacy/tokenization/train_bpe.py` before training.

- `train_stress_classifier.py` — WESAD-based stress-proxy classifier
  trainer. Not needed until a real EEG/HR input replaces the synthetic
  `u(t)` trajectory (see `models/stress_proxy.py`). Reactivate by
  downloading WESAD and running this script; the output model is what
  `infer_stream.py` would consume in a sensor-driven deployment.

- `configs/abl_with_bpe.yaml` — ablation config that would re-enable
  BPE. Kept because it remains a valid ablation direction if we ever
  want to report "BPE vs no-BPE on music codec sequences" in the paper.

## Rule

Do NOT import from `legacy/` in active code paths. If you revive a file,
move it out of `legacy/` first so the dependency graph stays honest.
