"""
Train a stress-proxy classifier on the WESAD dataset.

Produces a model that maps a window of physiological signals to
P(stress) ∈ [0, 1], which serves as the real-time stress proxy s(t).

Features extracted (per 4-second window):
  - ECG: HRV metrics (RMSSD, SDNN, mean HR)
  - EDA: mean SCL, number of SCR peaks, SCL slope
  - Respiration: breathing rate, breathing depth (amplitude)
  - Temperature: mean, slope

Classifier: Gradient Boosted Trees (scikit-learn)
  - Outputs calibrated probabilities via Platt scaling

Usage:
    python models/train_stress_classifier.py \
        --data_dir data/wesad/ \
        --output models/stress_model/ \
        --window_sec 4.0
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# ──────── Feature extraction ────────

def extract_ecg_features(ecg: np.ndarray, fs: float = 700.0,
                         window_samples: int = 2800) -> dict:
    """Extract basic HRV features from ECG signal."""
    # Simple R-peak detection via thresholding
    threshold = np.mean(ecg) + 1.5 * np.std(ecg)
    peaks, _ = find_peaks(ecg, height=threshold,
                          distance=int(0.5 * fs))  # min 0.5s between beats

    if len(peaks) < 3:
        return {"hr_mean": 0, "rmssd": 0, "sdnn": 0}

    # RR intervals in ms
    rr = np.diff(peaks) / fs * 1000.0
    hr = 60000.0 / rr  # beats per minute

    # HRV metrics
    rmssd = np.sqrt(np.mean(np.diff(rr) ** 2))
    sdnn = np.std(rr)

    return {
        "hr_mean": np.mean(hr),
        "rmssd": rmssd,
        "sdnn": sdnn,
    }


def extract_eda_features(eda: np.ndarray, fs: float = 700.0) -> dict:
    """Extract EDA features: skin conductance level and responses."""
    scl_mean = np.mean(eda)
    scl_std = np.std(eda)

    # Simple SCR detection
    diff_eda = np.diff(eda)
    scr_peaks, _ = find_peaks(diff_eda, height=0.01 * np.max(np.abs(diff_eda)),
                               distance=int(fs))
    n_scr = len(scr_peaks)

    # Slope (linear trend)
    t = np.arange(len(eda))
    if len(eda) > 1:
        slope = np.polyfit(t, eda, 1)[0]
    else:
        slope = 0.0

    return {
        "scl_mean": scl_mean,
        "scl_std": scl_std,
        "n_scr": n_scr,
        "scl_slope": slope,
    }


def extract_resp_features(resp: np.ndarray, fs: float = 700.0) -> dict:
    """Extract respiration features."""
    # Breathing rate from peak detection
    peaks, _ = find_peaks(resp, distance=int(1.5 * fs))  # min 1.5s between breaths
    if len(peaks) < 2:
        return {"breath_rate": 0, "breath_depth": 0}

    breath_intervals = np.diff(peaks) / fs
    breath_rate = 60.0 / np.mean(breath_intervals)  # breaths per minute

    # Depth: mean peak-to-trough amplitude
    troughs, _ = find_peaks(-resp, distance=int(1.5 * fs))
    if len(troughs) > 0 and len(peaks) > 0:
        depth = np.mean(np.abs(resp[peaks[:min(len(peaks), len(troughs))]]
                               - resp[troughs[:min(len(peaks), len(troughs))]]))
    else:
        depth = np.std(resp)

    return {
        "breath_rate": breath_rate,
        "breath_depth": depth,
    }


def extract_temp_features(temp: np.ndarray, fs: float = 700.0) -> dict:
    """Extract temperature features."""
    t = np.arange(len(temp))
    slope = np.polyfit(t, temp, 1)[0] if len(temp) > 1 else 0.0
    return {
        "temp_mean": np.mean(temp),
        "temp_slope": slope,
    }


def extract_window_features(chest_signals: dict, fs: float = 700.0) -> np.ndarray:
    """Extract all features for one window of chest sensor data."""
    feats = {}
    feats.update(extract_ecg_features(chest_signals["ECG"].flatten(), fs))
    feats.update(extract_eda_features(chest_signals["EDA"].flatten(), fs))
    feats.update(extract_resp_features(chest_signals["Resp"].flatten(), fs))
    feats.update(extract_temp_features(chest_signals["Temp"].flatten(), fs))
    return np.array(list(feats.values()), dtype=np.float32), list(feats.keys())


# ──────── Dataset loading ────────

def load_wesad_subject(pkl_path: str, window_sec: float = 4.0,
                       overlap: float = 0.5) -> tuple:
    """
    Load and window one WESAD subject's data.

    Returns:
        X: (n_windows, n_features)
        y: (n_windows,) — 0=baseline, 1=stress
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    labels = data["label"]
    chest = data["signal"]["chest"]
    fs = 700.0  # WESAD chest sensor sampling rate

    window_samples = int(window_sec * fs)
    hop_samples = int(window_samples * (1 - overlap))

    X_list, y_list = [], []

    for start in range(0, len(labels) - window_samples, hop_samples):
        end = start + window_samples
        window_label = labels[start:end]

        # Use only windows that are purely baseline (1) or stress (2)
        unique_labels = np.unique(window_label)
        if len(unique_labels) != 1:
            continue
        label = unique_labels[0]
        if label not in (1, 2):
            continue

        # Extract signals for this window
        window_signals = {
            key: val[start:end] for key, val in chest.items()
            if key in ("ECG", "EDA", "Resp", "Temp")
        }

        try:
            feats, feat_names = extract_window_features(window_signals, fs)
            if not np.any(np.isnan(feats)):
                X_list.append(feats)
                y_list.append(0 if label == 1 else 1)  # 0=baseline, 1=stress
        except Exception:
            continue

    if X_list:
        return np.stack(X_list), np.array(y_list), feat_names
    return np.empty((0, 0)), np.empty(0), []


# ──────── Training ────────

def train_stress_model(data_dir: str, output_dir: str,
                       window_sec: float = 4.0):
    """Train a stress classifier on all WESAD subjects."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    subject_dirs = sorted(Path(data_dir).glob("S*"))
    if not subject_dirs:
        print(f"No subject directories found in {data_dir}")
        print("Please download WESAD first: python data/download_wesad.py")
        return

    all_X, all_y, all_groups = [], [], []
    feat_names = None

    for subj_dir in subject_dirs:
        pkl_files = list(subj_dir.glob("*.pkl"))
        if not pkl_files:
            continue

        subj_id = subj_dir.name
        print(f"Loading {subj_id}...")
        X, y, names = load_wesad_subject(str(pkl_files[0]), window_sec)

        if len(y) > 0:
            all_X.append(X)
            all_y.append(y)
            all_groups.extend([subj_id] * len(y))
            if feat_names is None:
                feat_names = names
            print(f"  {len(y)} windows ({np.sum(y==0)} baseline, {np.sum(y==1)} stress)")

    if not all_X:
        print("No valid data loaded.")
        return

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    groups = np.array(all_groups)

    print(f"\nTotal: {len(y)} windows, {np.sum(y==0)} baseline, {np.sum(y==1)} stress")

    # Leave-one-subject-out cross-validation
    logo = LeaveOneGroupOut()
    accs, f1s, aucs = [], [], []

    for train_idx, test_idx in logo.split(X, y, groups):
        clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            random_state=42
        )
        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict(X[test_idx])
        y_prob = clf.predict_proba(X[test_idx])[:, 1]

        accs.append(accuracy_score(y[test_idx], y_pred))
        f1s.append(f1_score(y[test_idx], y_pred))
        if len(np.unique(y[test_idx])) > 1:
            aucs.append(roc_auc_score(y[test_idx], y_prob))

    print(f"\nLOSO CV Results:")
    print(f"  Accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"  F1:       {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
    if aucs:
        print(f"  AUC:      {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")

    # Train final model on all data with calibration
    base_clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42
    )
    calibrated_clf = CalibratedClassifierCV(base_clf, cv=5, method="sigmoid")
    calibrated_clf.fit(X, y)

    # Save model
    model_path = Path(output_dir) / "stress_classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": calibrated_clf,
            "feature_names": feat_names,
            "window_sec": window_sec,
            "cv_accuracy": np.mean(accs),
            "cv_f1": np.mean(f1s),
        }, f)

    print(f"\nModel saved to {model_path}")
    print(f"Feature names: {feat_names}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train WESAD stress classifier")
    parser.add_argument("--data_dir", default="data/wesad")
    parser.add_argument("--output", default="models/stress_model")
    parser.add_argument("--window_sec", type=float, default=4.0)
    args = parser.parse_args()

    train_stress_model(args.data_dir, args.output, args.window_sec)
