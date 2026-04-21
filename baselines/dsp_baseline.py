"""
DSP baseline: parametric audio effects conditioned on stress proxy.

No ML/codec editing — direct waveform processing using pedalboard.
This serves as a lower-bound baseline to ablate whether the codec-to-codec
transformer actually outperforms classic signal processing.

Effects chain:
  1. Parametric EQ (low-shelf boost, high-shelf cut) → calming frequency balance
  2. Compressor → reduce dynamic peaks → "smoother" feel
  3. Subtle pitch shift → lower pitch → lower perceived arousal

Usage:
    python baselines/dsp_baseline.py \
        --input data/saraga_kalyan_thaat/ \
        --output results/baseline_dsp/ \
        --u 0.6
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

try:
    from pedalboard import (
        Pedalboard, LowShelfFilter, HighShelfFilter,
        Compressor, PitchShift
    )
    HAS_PEDALBOARD = True
except ImportError:
    HAS_PEDALBOARD = False
    print("WARNING: pedalboard not installed. Install with: pip install pedalboard")


def build_dsp_chain(u: float, sr: int = 24000) -> "Pedalboard":
    """
    Build a pedalboard effect chain parameterized by edit intensity u ∈ [0,1].

    u=0 → identity (no processing)
    u=1 → maximum calming effect
    """
    if not HAS_PEDALBOARD:
        raise ImportError("pedalboard required for DSP baseline")

    effects = []

    # Low-shelf: gentle boost below 200 Hz (warmth)
    low_gain = 2.0 * u  # 0 to +2 dB
    if low_gain > 0.1:
        effects.append(LowShelfFilter(
            cutoff_frequency_hz=200.0,
            gain_db=low_gain,
        ))

    # High-shelf: reduce highs above 4 kHz (less "bright/harsh")
    high_gain = -3.0 * u  # 0 to -3 dB
    if abs(high_gain) > 0.1:
        effects.append(HighShelfFilter(
            cutoff_frequency_hz=4000.0,
            gain_db=high_gain,
        ))

    # Compressor: gentle compression at high u
    if u > 0.1:
        effects.append(Compressor(
            threshold_db=-20.0 + 10.0 * (1 - u),
            ratio=1.0 + 2.0 * u,
            attack_ms=10.0,
            release_ms=100.0,
        ))

    # Pitch shift: subtle downward at high u
    pitch_semitones = -0.5 * u
    if abs(pitch_semitones) > 0.05:
        effects.append(PitchShift(semitones=pitch_semitones))

    return Pedalboard(effects)


def process_file(audio_path: str, output_path: str, u: float = 0.6,
                 sr: int = 24000):
    """Process a single audio file through the DSP chain."""
    y, orig_sr = librosa.load(audio_path, sr=sr, mono=True)
    board = build_dsp_chain(u, sr)
    y_out = board(y.reshape(1, -1), sr).squeeze()

    # Match length
    if len(y_out) > len(y):
        y_out = y_out[:len(y)]
    elif len(y_out) < len(y):
        y_out = np.pad(y_out, (0, len(y) - len(y_out)))

    sf.write(output_path, y_out, sr)
    return y_out


def run_baseline(input_dir: str, output_dir: str, u: float = 0.6,
                 sr: int = 24000):
    """Run DSP baseline on all audio files in a directory."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    audio_files = sorted(Path(input_dir).glob("**/*.wav"))
    if not audio_files:
        audio_files = sorted(Path(input_dir).glob("**/*.mp3"))

    print(f"DSP baseline: processing {len(audio_files)} files at u={u}")
    for path in tqdm(audio_files, desc="DSP baseline"):
        out_path = Path(output_dir) / (path.stem + "_dsp.wav")
        process_file(str(path), str(out_path), u, sr)

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DSP baseline")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--u", type=float, default=0.6)
    parser.add_argument("--sr", type=int, default=24000)
    args = parser.parse_args()

    run_baseline(args.input, args.output, args.u, args.sr)
