"""
Pipeline orchestrator — run the full flow in one command.

Steps executed in order:
  1. Download / verify datasets (Saraga Yaman, WESAD, DEAM)
  2. Train WESAD stress-proxy classifier
  3. Generate synthetic paired edits
  4. Tokenize with WavTokenizer
  5. Train & apply codec-BPE
  6. Train codec-to-codec editor
  7. Run inference on test set
  8. Evaluate metrics
  9. Run baselines & compare

Usage:
    # Full pipeline
    python run_pipeline.py --config configs/proposed.yaml --run_name proposed_v1

    # Resume from a specific stage
    python run_pipeline.py --config configs/proposed.yaml --run_name proposed_v1 --start_stage 6

    # Only evaluate (assumes training is done)
    python run_pipeline.py --config configs/proposed.yaml --run_name proposed_v1 --start_stage 7
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def run(cmd: str, cwd: str = "."):
    """Run a command, stream output, abort on failure."""
    print(f"\n{'='*60}")
    print(f"  {cmd}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"\nFAILED (exit {result.returncode}): {cmd}")
        sys.exit(result.returncode)


def main(config_path: str, run_name: str, start_stage: int = 1):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if "_base_" in cfg:
        base_path = Path(config_path).parent / cfg["_base_"]
        with open(base_path) as f:
            base_cfg = yaml.safe_load(f)
        base_cfg.update({k: v for k, v in cfg.items() if k != "_base_"})
        cfg = base_cfg

    py = sys.executable
    saraga = cfg["data"]["saraga_dir"]
    pairs = cfg["data"]["pairs_dir"]
    tokens = cfg["data"]["tokens_dir"]
    bpe_model = cfg["bpe"]["model_dir"]

    # ── Stage 1: Data download ──
    if start_stage <= 1:
        print("\n\n▸ STAGE 1: Verify / download datasets")
        run(f"{py} data/download_saraga.py --raga yaman --output {saraga}")
        run(f"{py} data/download_wesad.py --output data/wesad")
        run(f"{py} data/download_deam.py --output {cfg['data']['deam_dir']}")

    # ── Stage 2: Train stress proxy ──
    if start_stage <= 2:
        print("\n\n▸ STAGE 2: Train WESAD stress-proxy classifier")
        run(f"{py} models/train_stress_classifier.py "
            f"--data_dir data/wesad --output models/stress_model")

    # ── Stage 3: Generate paired edits ──
    if start_stage <= 3:
        print("\n\n▸ STAGE 3: Generate synthetic paired training data")
        run(f"{py} data/prepare_pairs.py "
            f"--audio_dir {saraga} --output {pairs} "
            f"--max_clip_sec {cfg['data']['max_clip_sec']}")

    # ── Stage 4: Tokenize ──
    if start_stage <= 4:
        print("\n\n▸ STAGE 4: Tokenize with WavTokenizer")
        model_name = cfg["wavtokenizer"]["model_name"]
        run(f"{py} tokenization/encode_wavtokenizer.py "
            f"--input {pairs}/input --output {tokens}/input_wavtok "
            f"--model {model_name}")
        run(f"{py} tokenization/encode_wavtokenizer.py "
            f"--input {pairs}/target --output {tokens}/target_wavtok "
            f"--model {model_name}")

    # ── Stage 5: BPE ──
    if start_stage <= 5 and cfg["bpe"]["enabled"]:
        print("\n\n▸ STAGE 5: Train & apply codec-BPE")
        run(f"{py} tokenization/train_bpe.py "
            f"--codes_dir {tokens}/input_wavtok --output {bpe_model} "
            f"--vocab_size {cfg['bpe']['vocab_size']}")
        run(f"{py} tokenization/apply_bpe.py "
            f"--codes_dir {tokens}/input_wavtok --bpe_model {bpe_model} "
            f"--output {tokens}/input_bpe")
        run(f"{py} tokenization/apply_bpe.py "
            f"--codes_dir {tokens}/target_wavtok --bpe_model {bpe_model} "
            f"--output {tokens}/target_bpe")

    # ── Stage 6: Train editor ──
    if start_stage <= 6:
        print("\n\n▸ STAGE 6: Train codec-to-codec editor")
        run(f"{py} train.py --config {config_path} --run_name {run_name}")

    # ── Stage 7: Inference on test set ──
    if start_stage <= 7:
        print("\n\n▸ STAGE 7: Run inference on Saraga Yaman test recordings")
        ckpt = f"checkpoints/{run_name}/best"
        out_dir = f"results/{run_name}"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        for wav in sorted(Path(saraga).glob("*.wav")):
            run(f"{py} infer_stream.py "
                f"--input {wav} --output {out_dir}/{wav.stem}_modulated.wav "
                f"--checkpoint {ckpt} --config {config_path} "
                f"--stress_profile pulse --peak 0.6")

    # ── Stage 8: Evaluate ──
    if start_stage <= 8:
        print("\n\n▸ STAGE 8: Evaluate proposed pipeline")
        out_dir = f"results/{run_name}"
        run(f"{py} evaluate.py "
            f"--input {saraga} --output {out_dir} "
            f"--tonic_dir {saraga} "
            f"--results {out_dir}/eval_results.json")

    # ── Stage 9: Baselines ──
    if start_stage <= 9:
        print("\n\n▸ STAGE 9: Run and evaluate baselines")
        # DSP baseline
        dsp_dir = "results/baseline_dsp"
        run(f"{py} baselines/dsp_baseline.py "
            f"--input {saraga} --output {dsp_dir} --u 0.6")
        run(f"{py} evaluate.py "
            f"--input {saraga} --output {dsp_dir} "
            f"--tonic_dir {saraga} "
            f"--results {dsp_dir}/eval_results.json")

        # EnCodec+MLP baseline
        enc_dir = "results/baseline_encodec_mlp"
        run(f"{py} baselines/encodec_mlp_baseline.py "
            f"--input {saraga} --output {enc_dir} --u 0.6")
        run(f"{py} evaluate.py "
            f"--input {saraga} --output {enc_dir} "
            f"--tonic_dir {saraga} "
            f"--results {enc_dir}/eval_results.json")

    print("\n\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Results:    results/{run_name}/eval_results.json")
    print(f"  Baseline 1: results/baseline_dsp/eval_results.json")
    print(f"  Baseline 2: results/baseline_encodec_mlp/eval_results.json")
    print(f"  Checkpoint: checkpoints/{run_name}/best/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full pipeline orchestrator")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--start_stage", type=int, default=1,
                        help="Resume from this stage (1-9)")
    args = parser.parse_args()

    main(args.config, args.run_name, args.start_stage)
