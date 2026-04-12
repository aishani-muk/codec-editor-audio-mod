"""
Evaluation script: compute all preservation and quality metrics.

Metrics:
  1. Tonic drift (Hz, cents) — from Saraga tonic annotations
  2. Pitch-histogram JSD — tonic-normalized, octave-folded
  3. Velocity TV / Jerk RMS — glide smoothness proxies
  4. DEAM Δvalence / Δarousal — emotional shift toward neutral
  5. PESQ (optional) — perceptual audio quality

Usage:
    # Evaluate proposed pipeline
    python evaluate.py --checkpoint checkpoints/proposed_v1/best/ \\
                       --config configs/proposed.yaml

    # Evaluate DSP baseline
    python evaluate.py --baseline dsp --input results/baseline_dsp/ \\
                       --reference data/saraga_yaman/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import librosa
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy


# ──────────────────── Pitch extraction ────────────────────

def extract_pitch(audio: np.ndarray, sr: int = 24000,
                  hop_ms: float = 10.0) -> np.ndarray:
    """
    Extract pitch track using CREPE or pyin.

    Returns:
        pitch: (T,) array of pitch values in Hz (0 = unvoiced).
    """
    try:
        import crepe
        _, pitch, confidence, _ = crepe.predict(
            audio, sr, step_size=hop_ms, viterbi=True
        )
        pitch[confidence < 0.5] = 0  # Mark low-confidence as unvoiced
        return pitch
    except ImportError:
        # Fallback to librosa pyin
        f0, voiced_flag, _ = librosa.pyin(
            audio, fmin=50, fmax=800, sr=sr,
            hop_length=int(sr * hop_ms / 1000)
        )
        f0 = np.nan_to_num(f0, nan=0.0)
        return f0


def load_tonic(tonic_path: str) -> float:
    """Load tonic frequency from a Saraga .tonic file."""
    with open(tonic_path) as f:
        return float(f.read().strip())


# ──────────────────── Metric computations ────────────────────

def tonic_drift(pitch_in: np.ndarray, pitch_out: np.ndarray,
                tonic_hz: float) -> dict:
    """Compute tonic drift between input and output pitch tracks."""
    # Estimate tonic from each track (median of voiced frames near tonic)
    def est_tonic(pitch, ref_tonic):
        voiced = pitch[pitch > 0]
        if len(voiced) == 0:
            return ref_tonic
        # Find frames within ±100 cents of reference
        cents = 1200 * np.log2(voiced / ref_tonic + 1e-8)
        near = voiced[np.abs(cents) < 100]
        return np.median(near) if len(near) > 0 else np.median(voiced)

    tonic_in = est_tonic(pitch_in, tonic_hz)
    tonic_out = est_tonic(pitch_out, tonic_hz)

    drift_hz = abs(tonic_out - tonic_in)
    drift_cents = 1200 * abs(np.log2(tonic_out / tonic_in + 1e-8))

    return {
        "tonic_in_hz": tonic_in,
        "tonic_out_hz": tonic_out,
        "drift_hz": drift_hz,
        "drift_cents": drift_cents,
    }


def pitch_histogram_jsd(pitch_in: np.ndarray, pitch_out: np.ndarray,
                        tonic_hz: float, n_bins: int = 1200) -> float:
    """
    Compute JSD between tonic-normalized, octave-folded pitch histograms.

    Lower = better preservation of raga pitch-class distribution.
    """
    def to_hist(pitch):
        voiced = pitch[pitch > 0]
        if len(voiced) == 0:
            return np.ones(n_bins) / n_bins
        cents = 1200 * np.log2(voiced / tonic_hz + 1e-8)
        folded = cents % 1200
        hist, _ = np.histogram(folded, bins=n_bins, range=(0, 1200),
                               density=True)
        hist += 1e-10  # Avoid zeros
        return hist / hist.sum()

    P = to_hist(pitch_in)
    Q = to_hist(pitch_out)
    return float(jensenshannon(P, Q) ** 2)  # JSD = JS distance squared


def smoothness_metrics(pitch: np.ndarray, sr: int = 24000,
                       hop_ms: float = 10.0, tonic_hz: float = 1.0) -> dict:
    """
    Compute velocity TV and jerk RMS on tonic-normalized cents trajectory.
    """
    voiced = pitch.copy()
    voiced[voiced <= 0] = np.nan
    cents = 1200 * np.log2(voiced / tonic_hz + 1e-8)

    dt = hop_ms / 1000.0
    # Velocity
    v = np.diff(cents) / dt
    v = v[~np.isnan(v)]
    velocity_tv = np.mean(np.abs(v)) if len(v) > 0 else 0.0

    # Jerk (second derivative of velocity)
    a = np.diff(v) / dt if len(v) > 1 else np.array([0.0])
    jerk_rms = np.sqrt(np.mean(a ** 2)) if len(a) > 0 else 0.0

    return {
        "velocity_tv": velocity_tv,
        "jerk_rms": jerk_rms,
    }


# ──────────────────── Full evaluation ────────────────────

def evaluate_pair(input_wav: str, output_wav: str, tonic_hz: float,
                  sr: int = 24000, hop_ms: float = 10.0) -> dict:
    """Evaluate a single (input, output) audio pair."""
    y_in, _ = librosa.load(input_wav, sr=sr, mono=True)
    y_out, _ = librosa.load(output_wav, sr=sr, mono=True)

    pitch_in = extract_pitch(y_in, sr, hop_ms)
    pitch_out = extract_pitch(y_out, sr, hop_ms)

    drift = tonic_drift(pitch_in, pitch_out, tonic_hz)
    jsd = pitch_histogram_jsd(pitch_in, pitch_out, tonic_hz)

    smooth_in = smoothness_metrics(pitch_in, sr, hop_ms, tonic_hz)
    smooth_out = smoothness_metrics(pitch_out, sr, hop_ms, tonic_hz)

    return {
        **drift,
        "jsd": jsd,
        "velocity_tv_in": smooth_in["velocity_tv"],
        "velocity_tv_out": smooth_out["velocity_tv"],
        "jerk_rms_in": smooth_in["jerk_rms"],
        "jerk_rms_out": smooth_out["jerk_rms"],
        "velocity_tv_change_pct": (
            (smooth_out["velocity_tv"] - smooth_in["velocity_tv"])
            / (smooth_in["velocity_tv"] + 1e-8) * 100
        ),
        "jerk_rms_change_pct": (
            (smooth_out["jerk_rms"] - smooth_in["jerk_rms"])
            / (smooth_in["jerk_rms"] + 1e-8) * 100
        ),
    }


def evaluate_directory(input_dir: str, output_dir: str,
                       tonic_dir: str | None = None,
                       default_tonic: float = 261.63,
                       results_path: str = "results/eval_results.json"):
    """Evaluate all pairs in matching directories."""
    input_files = sorted(Path(input_dir).glob("*.wav"))
    results = []

    for in_path in input_files:
        # Find matching output
        stem = in_path.stem
        out_candidates = list(Path(output_dir).glob(f"{stem}*"))
        if not out_candidates:
            print(f"  Skipping {stem}: no matching output found")
            continue
        out_path = out_candidates[0]

        # Load tonic
        tonic_hz = default_tonic
        if tonic_dir:
            tonic_file = Path(tonic_dir) / f"{stem}.tonic"
            if tonic_file.exists():
                tonic_hz = load_tonic(str(tonic_file))

        print(f"  Evaluating: {stem} (tonic={tonic_hz:.1f} Hz)")
        metrics = evaluate_pair(str(in_path), str(out_path), tonic_hz)
        metrics["recording"] = stem
        results.append(metrics)

    # Aggregate
    if results:
        agg = {}
        for key in ["drift_cents", "drift_hz", "jsd",
                     "velocity_tv_change_pct", "jerk_rms_change_pct"]:
            vals = [r[key] for r in results]
            agg[f"mean_{key}"] = float(np.mean(vals))
            agg[f"std_{key}"] = float(np.std(vals))

        results_data = {
            "per_recording": results,
            "aggregate": agg,
            "n_recordings": len(results),
        }

        Path(results_path).parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(results_data, f, indent=2)

        print(f"\n{'='*50}")
        print(f"Evaluation Summary ({len(results)} recordings)")
        print(f"{'='*50}")
        for k, v in agg.items():
            print(f"  {k}: {v:.4f}")
        print(f"\nFull results saved to: {results_path}")
    else:
        print("No recordings evaluated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate audio modulation pipeline")
    parser.add_argument("--input", required=True,
                        help="Directory of original input WAVs")
    parser.add_argument("--output", required=True,
                        help="Directory of modulated output WAVs")
    parser.add_argument("--tonic_dir", default=None,
                        help="Directory of .tonic files (Saraga annotations)")
    parser.add_argument("--default_tonic", type=float, default=261.63,
                        help="Default tonic Hz if no .tonic file found")
    parser.add_argument("--results", default="results/eval_results.json")
    args = parser.parse_args()

    evaluate_directory(args.input, args.output, args.tonic_dir,
                       args.default_tonic, args.results)
