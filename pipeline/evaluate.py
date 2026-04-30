"""Pairwise (input, output) eval: tonic drift, PCD JSD, velocity TV, jerk RMS,
M2E ΔV/ΔA, raga-ID preservation.

Usage:
    python evaluate.py --input <in_dir> --output <out_dir> --tonic_dir <saraga_dir>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import librosa

from evaluation.raga_classifier import RagaPredictor
from evaluation.pitch import extract_pitch_with_confidence
from evaluation.tonic import resolve_tonic
from evaluation.pcd import pcd_jsd
from evaluation.emotion_regressor import Music2EmoRegressor


# ──────────────────── Pitch extraction ────────────────────

def extract_pitch(audio: np.ndarray, sr: int = 24000,
                  hop_ms: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Pitch + confidence via PESTO → CREPE → pyin. Low-confidence frames have pitch=0."""
    return extract_pitch_with_confidence(audio, sr=sr, hop_ms=hop_ms,
                                         conf_threshold=0.5)


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


def pitch_histogram_jsd(pitch_in: np.ndarray, conf_in: np.ndarray,
                        pitch_out: np.ndarray, conf_out: np.ndarray,
                        tonic_hz: float) -> float:
    """Salience-weighted JSD between tonic-normalised PCDs (lower is better)."""
    return pcd_jsd(pitch_in, conf_in, pitch_out, conf_out, tonic_hz)


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
                  sr: int = 24000, hop_ms: float = 10.0,
                  raga_predictor: "RagaPredictor | None" = None,
                  emotion_regressor: "Music2EmoRegressor | None" = None
                  ) -> dict:
    """Evaluate a single (input, output) audio pair."""
    y_in, _ = librosa.load(input_wav, sr=sr, mono=True)
    y_out, _ = librosa.load(output_wav, sr=sr, mono=True)

    pitch_in,  conf_in  = extract_pitch(y_in,  sr, hop_ms)
    pitch_out, conf_out = extract_pitch(y_out, sr, hop_ms)

    drift = tonic_drift(pitch_in, pitch_out, tonic_hz)
    jsd = pitch_histogram_jsd(pitch_in, conf_in, pitch_out, conf_out, tonic_hz)

    smooth_in = smoothness_metrics(pitch_in, sr, hop_ms, tonic_hz)
    smooth_out = smoothness_metrics(pitch_out, sr, hop_ms, tonic_hz)

    metrics = {
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

    if raga_predictor is not None and raga_predictor.available:
        # Raga classifier: run on raw WAVs (it handles its own SR / resampling).
        raga_in, r_conf_in = raga_predictor.predict_from_audio(
            input_wav, tonic_hz=tonic_hz,
        )
        raga_out, r_conf_out = raga_predictor.predict_from_audio(
            output_wav, tonic_hz=tonic_hz,
        )
        metrics.update({
            "input_pred_raga": raga_in,
            "input_pred_conf": r_conf_in,
            "output_pred_raga": raga_out,
            "output_pred_conf": r_conf_out,
            "raga_id_preserved": bool(raga_in == raga_out),
        })

    if emotion_regressor is not None and emotion_regressor.available:
        # Music2Emo: (V, A) on DEAM 1-9 scale + MTG-Jamendo mood tags.
        shift = emotion_regressor.predict_pair(input_wav, output_wav)
        metrics.update(shift)
    return metrics


def evaluate_directory(input_dir: str, output_dir: str,
                       tonic_dir: str | None = None,
                       default_tonic: float = 261.63,
                       results_path: str = "results/eval_results.json",
                       raga_checkpoint: str | None = None,
                       use_emotion: bool = False):
    """Evaluate all pairs in matching directories."""
    input_files = sorted(Path(input_dir).glob("*.wav"))
    results = []

    raga_predictor = None
    if raga_checkpoint:
        raga_predictor = RagaPredictor(raga_checkpoint)
        if raga_predictor.available:
            print(f"  Raga classifier loaded: {raga_checkpoint} "
                  f"(feature={raga_predictor.feature_type}, "
                  f"n_classes={len(raga_predictor.vocab)})")
        else:
            print(f"  WARNING: raga checkpoint not found at {raga_checkpoint}; "
                  f"skipping raga-id metric")
            raga_predictor = None

    emotion_regressor = None
    if use_emotion:
        emotion_regressor = Music2EmoRegressor()
        if emotion_regressor.available:
            print(f"  Music2Emo regressor available at "
                  f"{emotion_regressor.MUSIC2EMO_DIR} (will load lazily).")
        else:
            print(f"  WARNING: Music2Emo not found; skipping V/A + mood metric")
            emotion_regressor = None

    for in_path in input_files:
        # Find matching output
        stem = in_path.stem
        out_candidates = list(Path(output_dir).glob(f"{stem}*"))
        if not out_candidates:
            print(f"  Skipping {stem}: no matching output found")
            continue
        out_path = out_candidates[0]

        # Saraga .ctonic.txt → Essentia Gulati → default.
        tonic_hz, tonic_src = resolve_tonic(
            str(in_path), stem=stem,
            tonic_dir=tonic_dir, default_hz=default_tonic,
        )
        print(f"  Evaluating: {stem} (tonic={tonic_hz:.1f} Hz, src={tonic_src})")
        metrics = evaluate_pair(str(in_path), str(out_path), tonic_hz,
                                raga_predictor=raga_predictor,
                                emotion_regressor=emotion_regressor)
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

        # ── Degenerate-output gate (rescue-v3 Day 1.3) ──
        # Flag runs where velocity_tv collapses to -100% (i.e. output is
        # silent or constant) on >= 50% of clips. Stops us from ever
        # reporting "great" drift/jsd numbers for a run that just outputs
        # zeros.
        vel = np.asarray([r["velocity_tv_change_pct"] for r in results],
                         dtype=np.float64)
        collapsed_mask = np.abs(vel) >= 99.5
        n_collapsed = int(collapsed_mask.sum())
        frac_collapsed = float(n_collapsed / max(len(vel), 1))
        is_collapsed = bool(frac_collapsed >= 0.5)
        agg["n_collapsed_clips"] = n_collapsed
        agg["frac_collapsed"] = frac_collapsed
        agg["degenerate_status"] = "COLLAPSED" if is_collapsed else "OK"
        if is_collapsed:
            print(f"  !! DEGENERATE-OUTPUT GATE TRIPPED: "
                  f"{n_collapsed}/{len(vel)} clips "
                  f"({100.0 * frac_collapsed:.1f}%) have "
                  f"|velocity_tv_change_pct| >= 99.5. "
                  f"Tagging aggregate as COLLAPSED.")

        if raga_predictor is not None:
            preserved = [r["raga_id_preserved"] for r in results
                         if "raga_id_preserved" in r]
            if preserved:
                agg["raga_id_preserved_pct"] = (
                    100.0 * sum(preserved) / len(preserved)
                )
                agg["raga_id_n"] = len(preserved)

        if emotion_regressor is not None:
            va_rows = [r for r in results if "delta_valence" in r]
            if va_rows:
                for key in ("delta_valence", "delta_arousal",
                            "dist_to_neutral_in", "dist_to_neutral_out"):
                    agg[f"mean_{key}"] = float(np.mean([r[key] for r in va_rows]))
                agg["moved_toward_neutral_pct"] = (
                    100.0 * sum(r["moved_toward_neutral"] for r in va_rows)
                    / len(va_rows)
                )
                agg["emotion_n"] = len(va_rows)

        results_data = {
            "per_recording": results,
            "aggregate": agg,
            "n_recordings": len(results),
        }

        Path(results_path).parent.mkdir(parents=True, exist_ok=True)

        def _to_py(obj):
            """JSON encoder hook: downcast numpy scalars to Python scalars."""
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Not JSON-serializable: {type(obj).__name__}")

        with open(results_path, "w") as f:
            json.dump(results_data, f, indent=2, default=_to_py)

        print(f"\n{'='*50}")
        print(f"Evaluation Summary ({len(results)} recordings)")
        print(f"{'='*50}")
        for k, v in agg.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
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
    parser.add_argument("--raga_checkpoint", default=None,
                        help="Path to a raga-classifier checkpoint "
                             "(e.g. checkpoints/raga_classifier_pcd/model.pt). "
                             "If provided, logs raga_id_preserved per clip "
                             "and the aggregate raga_id_preserved_pct.")
    parser.add_argument("--emotion", action="store_true",
                        help="Enable Music2Emo-based valence/arousal/mood "
                             "prediction on input and output. Requires "
                             "third_party/Music2Emotion/ to be populated. "
                             "Logs delta_valence, delta_arousal, moods_in, "
                             "moods_out, moved_toward_neutral per clip and "
                             "the aggregate moved_toward_neutral_pct.")
    args = parser.parse_args()

    evaluate_directory(args.input, args.output, args.tonic_dir,
                       args.default_tonic, args.results,
                       raga_checkpoint=args.raga_checkpoint,
                       use_emotion=args.emotion)
