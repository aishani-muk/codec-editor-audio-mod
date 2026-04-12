"""
Generate synthetic paired edits for training the codec-to-codec editor.

For each audio clip, apply parameterized DSP transformations at varying
intensity levels λ ∈ [0,1] to produce (input, edited) pairs that simulate
"lower-arousal" modulation. The editor is then trained to map
input_tokens → edited_tokens conditioned on u = λ.

Transformations applied:
  - Slight pitch lowering (up to -0.5 semitones)
  - Gentle low-pass filtering (reduce high-frequency energy)
  - Dynamic range compression (reduce peaks → calmer dynamics)
  - Tempo micro-adjustment (up to -3% slower)

Usage:
    python data/prepare_pairs.py --config configs/proposed.yaml
"""

import argparse
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm


def apply_calming_transform(audio: np.ndarray, sr: int,
                            lam: float) -> np.ndarray:
    """
    Apply a parameterized "calming" transformation at intensity λ ∈ [0,1].

    λ=0 → identity (no change)
    λ=1 → maximum calming effect
    """
    y = audio.copy()

    # 1. Pitch shift: up to -0.5 semitones
    pitch_shift = -0.5 * lam
    if abs(pitch_shift) > 0.01:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift)

    # 2. Low-pass filter: reduce high-frequency energy
    # Cutoff from sr/2 (no filter) down to 6000 Hz at max λ
    cutoff = sr / 2 - (sr / 2 - 6000) * lam
    if cutoff < sr / 2 - 100:
        from scipy.signal import butter, sosfilt
        sos = butter(4, cutoff, btype='low', fs=sr, output='sos')
        y = sosfilt(sos, y).astype(np.float32)

    # 3. Dynamic range compression (simple soft-knee)
    threshold_db = -20 + 10 * (1 - lam)  # lower threshold at higher λ
    threshold = 10 ** (threshold_db / 20)
    ratio = 1 + 2 * lam  # 1:1 at λ=0, 3:1 at λ=1
    mask = np.abs(y) > threshold
    if mask.any():
        excess = np.abs(y[mask]) - threshold
        compressed = threshold + excess / ratio
        y[mask] = np.sign(y[mask]) * compressed

    # 4. Tempo: up to -3% slower
    rate = 1.0 - 0.03 * lam
    if rate < 0.99:
        y = librosa.effects.time_stretch(y, rate=rate)

    # Match original length
    if len(y) > len(audio):
        y = y[:len(audio)]
    elif len(y) < len(audio):
        y = np.pad(y, (0, len(audio) - len(y)))

    return y


def prepare_pairs(audio_dir: str, output_dir: str, sr: int = 24000,
                  lambdas: list[float] | None = None,
                  max_clip_sec: float = 30.0):
    """
    Create paired (original, edited) WAV files for training.

    For each audio file, generates multiple pairs at different λ values.
    """
    if lambdas is None:
        lambdas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    input_dir = Path(output_dir) / "input"
    target_dir = Path(output_dir) / "target"
    meta_dir = Path(output_dir) / "meta"
    for d in [input_dir, target_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(Path(audio_dir).glob("**/*.wav"))
    if not audio_files:
        audio_files = sorted(Path(audio_dir).glob("**/*.mp3"))
    print(f"Found {len(audio_files)} audio files")

    pair_count = 0
    for audio_path in tqdm(audio_files, desc="Generating pairs"):
        y, orig_sr = librosa.load(str(audio_path), sr=sr, mono=True,
                                   duration=max_clip_sec)

        for lam in lambdas:
            y_edited = apply_calming_transform(y, sr, lam)

            pair_name = f"{audio_path.stem}_lam{lam:.1f}"
            sf.write(str(input_dir / f"{pair_name}.wav"), y, sr)
            sf.write(str(target_dir / f"{pair_name}.wav"), y_edited, sr)

            # Save metadata
            np.savez(str(meta_dir / f"{pair_name}.npz"),
                     lambda_val=lam,
                     source_file=str(audio_path.name),
                     duration_sec=len(y) / sr)
            pair_count += 1

    print(f"Generated {pair_count} training pairs in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic paired edits")
    parser.add_argument("--audio_dir", default="data/saraga_yaman",
                        help="Source audio directory")
    parser.add_argument("--output", default="data/paired_edits",
                        help="Output paired data directory")
    parser.add_argument("--sr", type=int, default=24000)
    parser.add_argument("--max_clip_sec", type=float, default=30.0)
    args = parser.parse_args()

    prepare_pairs(args.audio_dir, args.output, args.sr,
                  max_clip_sec=args.max_clip_sec)
