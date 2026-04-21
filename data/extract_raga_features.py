"""
Extract raga-specific musical features from Saraga tonic + pitch annotations.

For each track, computes:
  - tonic_hz: fundamental reference frequency (Sa)
  - pitch_cents_mod_octave: pitch contour in cents, folded to [0, 1200)
  - svara_histogram: 1200-bin histogram of octave-folded pitch (voiced frames only)
  - dominant_svaras: top-K peaks of the histogram, mapped to 12-svara names
  - transition_matrix_12x12: first-order svara transition probabilities
  - aaroha_bigrams / avroha_bigrams: top ascending / descending note pairs
  - velocity_stats: mean absolute d(pitch)/dt in cents/s

Then aggregates across tracks:
  - per_raga: mean histogram + std per raga
  - yaman_signature: Yaman's histogram
  - yaman_vs_others_jsd: Jensen-Shannon divergence Yaman vs. mean-of-others
      (measures how distinguishable Yaman is from the Kalyan-thaat siblings;
       this doubles as a preservation metric during editing)
  - plots (PNG): per-raga histograms, Yaman-vs-others overlay

Usage:
    python data/extract_raga_features.py \
        --data_dir data/saraga_kalyan_thaat/ \
        --output data/raga_features/
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


# 12-svara names in cents (equal-tempered grid, Yaman uses teevra Ma = M')
SVARA_CENTS = {
    0:    "S",
    100:  "r",
    200:  "R",
    300:  "g",
    400:  "G",
    500:  "M",
    600:  "M'",  # teevra Ma (Yaman's characteristic note)
    700:  "P",
    800:  "d",
    900:  "D",
    1000: "n",
    1100: "N",
}


def load_tonic(track_dir: Path) -> float | None:
    """Read the single-float tonic frequency from .ctonic.txt."""
    hits = list(track_dir.glob("*.ctonic.txt"))
    if not hits:
        return None
    with open(hits[0]) as f:
        return float(f.read().strip())


def load_pitch(track_dir: Path) -> np.ndarray | None:
    """Read (time, Hz) pairs from .pitch.txt. Returns (T, 2) array."""
    # Prefer post-processed if present
    for pat in ("*.pitch-pp.txt", "*.pitch.txt"):
        hits = list(track_dir.glob(pat))
        if hits:
            return np.loadtxt(hits[0])
    return None


def load_raga(track_dir: Path) -> str | None:
    """Read raga common_name from the track's .json."""
    hits = list(track_dir.glob("*.json"))
    if not hits:
        return None
    # Some albums may have multiple json files; pick the one matching dir name
    for h in hits:
        if h.stem == track_dir.name:
            hits = [h]
            break
    with open(hits[0]) as f:
        d = json.load(f)
    raags = d.get("raags", [])
    if not raags:
        return None
    return raags[0].get("common_name")


def pitch_to_cents_mod_octave(pitch: np.ndarray, tonic_hz: float,
                               voiced_only: bool = True) -> np.ndarray:
    """Convert (T,2) time+Hz array → 1D cents mod 1200 (voiced frames only)."""
    hz = pitch[:, 1]
    if voiced_only:
        hz = hz[hz > 0]
    if len(hz) == 0:
        return np.array([])
    cents = 1200.0 * np.log2(hz / tonic_hz)
    return np.mod(cents, 1200.0)


def histogram_1cent(cents_mod: np.ndarray) -> np.ndarray:
    """Build 1200-bin normalized histogram over [0, 1200) cents."""
    h, _ = np.histogram(cents_mod, bins=1200, range=(0, 1200))
    total = h.sum()
    return (h / total) if total > 0 else h.astype(float)


def nearest_svara(cent_val: float) -> tuple[int, str]:
    """Snap a cent value to the nearest 12-svara grid position."""
    grid = np.arange(0, 1200, 100)
    idx = int(np.argmin(np.abs(grid - cent_val)))
    c = int(grid[idx])
    return c, SVARA_CENTS[c]


def histogram_to_12svara(hist_1cent: np.ndarray, window: int = 50) -> np.ndarray:
    """
    Collapse the 1200-bin histogram onto the 12-svara grid by summing a
    ±window-cent window around each equal-tempered position.
    """
    h12 = np.zeros(12, dtype=float)
    for i, c in enumerate(np.arange(0, 1200, 100)):
        lo = max(0, c - window)
        hi = min(1200, c + window)
        h12[i] = hist_1cent[lo:hi].sum()
    total = h12.sum()
    return h12 / total if total > 0 else h12


def svara_transitions(pitch: np.ndarray, tonic_hz: float,
                      min_hop_ms: float = 30.0) -> np.ndarray:
    """
    First-order transition matrix between consecutive *distinct* svaras.
    pitch: (T, 2) time+Hz
    Returns 12x12 row-normalized probability matrix.
    """
    times = pitch[:, 0]
    hz = pitch[:, 1]
    mask = hz > 0
    if mask.sum() < 10:
        return np.zeros((12, 12))
    t = times[mask]
    cents = np.mod(1200.0 * np.log2(hz[mask] / tonic_hz), 1200.0)

    # Snap each frame to nearest svara index (0..11)
    svara_idx = np.round(cents / 100.0).astype(int) % 12

    # Remove tiny consecutive runs (debounce)
    keep = [0]
    last_t = t[0]
    last_s = svara_idx[0]
    for i in range(1, len(svara_idx)):
        if svara_idx[i] != last_s and (t[i] - last_t) * 1000 >= min_hop_ms:
            keep.append(i)
            last_s = svara_idx[i]
            last_t = t[i]
    reduced = svara_idx[keep]

    M = np.zeros((12, 12), dtype=float)
    for a, b in zip(reduced[:-1], reduced[1:]):
        M[a, b] += 1
    row_sums = M.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return M / row_sums


def velocity_stats(pitch: np.ndarray, tonic_hz: float) -> dict:
    """Mean |d(cents)/dt| — proxy for how much glide is happening."""
    mask = pitch[:, 1] > 0
    if mask.sum() < 2:
        return {"mean_abs_velocity_cents_per_s": 0.0}
    t = pitch[mask, 0]
    cents = 1200.0 * np.log2(pitch[mask, 1] / tonic_hz)
    dt = np.diff(t)
    dc = np.diff(cents)
    dt[dt < 1e-9] = 1e-9
    v = np.abs(dc / dt)
    return {
        "mean_abs_velocity_cents_per_s": float(v.mean()),
        "median_abs_velocity_cents_per_s": float(np.median(v)),
    }


def jensen_shannon(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    def kl(a, b):
        return float(np.sum(a * np.log(a / b)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def analyze_track(track_dir: Path) -> dict | None:
    tonic = load_tonic(track_dir)
    pitch = load_pitch(track_dir)
    raga = load_raga(track_dir)
    if tonic is None or pitch is None or raga is None:
        return None

    cents_mod = pitch_to_cents_mod_octave(pitch, tonic)
    h1c = histogram_1cent(cents_mod)
    h12 = histogram_to_12svara(h1c)

    # Top 3 dominant svaras
    top_idx = np.argsort(h12)[::-1][:3]
    dominant = [
        {"svara": SVARA_CENTS[int(i) * 100], "fraction": float(h12[i])}
        for i in top_idx
    ]

    trans = svara_transitions(pitch, tonic)

    # Ascending / descending bigrams (by top transition probability)
    bigrams = []
    for i in range(12):
        for j in range(12):
            if i != j and trans[i, j] > 0:
                bigrams.append((SVARA_CENTS[i*100], SVARA_CENTS[j*100],
                                float(trans[i, j]), int(j > i or (j == 0 and i > 6))))
    aaroha = sorted([b for b in bigrams if b[3] == 1], key=lambda x: -x[2])[:6]
    avroha = sorted([b for b in bigrams if b[3] == 0], key=lambda x: -x[2])[:6]

    v = velocity_stats(pitch, tonic)

    return {
        "raga": raga,
        "tonic_hz": float(tonic),
        "duration_sec": float(pitch[-1, 0] - pitch[0, 0]),
        "hist_1cent": h1c.tolist(),
        "hist_12svara": h12.tolist(),
        "dominant_svaras": dominant,
        "aaroha_bigrams": [[a, b, p] for a, b, p, _ in aaroha],
        "avroha_bigrams": [[a, b, p] for a, b, p, _ in avroha],
        "transition_12x12": trans.tolist(),
        "velocity": v,
    }


def run(data_dir: str, output_dir: str, plot: bool = True):
    data_dir = Path(data_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    per_track = []
    for jpath in sorted(data_dir.glob("*/*/*.json")):
        tdir = jpath.parent
        # Skip manifest / top-level files
        if tdir.name in ("_saraga_staging",):
            continue
        result = analyze_track(tdir)
        if result is None:
            continue
        result["track_path"] = str(tdir.relative_to(data_dir))
        per_track.append(result)
        print(f"  {result['raga']:<20} "
              f"tonic={result['tonic_hz']:.1f}Hz  "
              f"dur={result['duration_sec']/60:.1f}min  "
              f"top={','.join(s['svara'] for s in result['dominant_svaras'])}")

    # Save per-track
    with open(out / "per_track_features.json", "w") as f:
        json.dump(per_track, f, indent=2)

    # Aggregate per raga
    by_raga = defaultdict(list)
    for r in per_track:
        by_raga[r["raga"]].append(np.array(r["hist_1cent"]))
    per_raga = {}
    for raga, hists in by_raga.items():
        arr = np.stack(hists, axis=0)
        mean_h = arr.mean(axis=0)
        std_h = arr.std(axis=0)
        per_raga[raga] = {
            "n_tracks": len(hists),
            "mean_hist_1cent": mean_h.tolist(),
            "std_hist_1cent": std_h.tolist(),
            "mean_hist_12svara": histogram_to_12svara(mean_h).tolist(),
        }
    with open(out / "per_raga_histograms.json", "w") as f:
        json.dump(per_raga, f, indent=2)

    # Yaman signature + JSD vs others
    yaman_hist = None
    other_hists = []
    for raga, info in per_raga.items():
        h = np.array(info["mean_hist_1cent"])
        if raga == "Yaman kalyan":
            yaman_hist = h
        else:
            other_hists.append(h)

    if yaman_hist is not None and other_hists:
        mean_other = np.mean(np.stack(other_hists, axis=0), axis=0)
        mean_other = mean_other / mean_other.sum()
        jsd = jensen_shannon(yaman_hist, mean_other)

        signature = {
            "yaman_mean_hist_1cent": yaman_hist.tolist(),
            "yaman_mean_hist_12svara": histogram_to_12svara(yaman_hist).tolist(),
            "other_kalyan_thaat_mean_hist_1cent": mean_other.tolist(),
            "jsd_yaman_vs_other_kalyan_thaat": float(jsd),
            "notes": (
                "Yaman's distinguishing features: strong teevra Ma (M'), "
                "strong Ga (G) and Ni (N), no komal svaras. The JSD measures "
                "how far Yaman's pitch-class distribution sits from the mean "
                "of its Kalyan-thaat siblings. During editing, this JSD should "
                "stay small if the raga identity is preserved."
            ),
        }
        with open(out / "yaman_signature.json", "w") as f:
            json.dump(signature, f, indent=2)
        print(f"\nYaman vs other Kalyan-thaat JSD: {jsd:.4f}")
        print(f"  (lower = more similar; Yaman's teevra Ma is the main differentiator)")

    # Optional plots
    if plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig_dir = out / "plots"
            fig_dir.mkdir(exist_ok=True)

            # Per-raga mean histogram with svara labels
            x = np.arange(1200)
            svara_positions = list(SVARA_CENTS.keys())
            svara_labels = [SVARA_CENTS[c] for c in svara_positions]

            for raga, info in per_raga.items():
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(x, info["mean_hist_1cent"], lw=0.8)
                for c in svara_positions:
                    ax.axvline(c, color="red", lw=0.3, alpha=0.5)
                ax.set_xticks(svara_positions)
                ax.set_xticklabels(svara_labels)
                ax.set_xlabel("Cents from tonic (octave-folded)")
                ax.set_ylabel("Probability")
                ax.set_title(f"Pitch-class histogram: {raga} (n={info['n_tracks']})")
                plt.tight_layout()
                safe = raga.replace(" ", "_").replace("/", "_")
                fig.savefig(fig_dir / f"hist_{safe}.png", dpi=120)
                plt.close(fig)

            # Overlay: Yaman vs mean-of-others
            if yaman_hist is not None and other_hists:
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(x, yaman_hist, label="Yaman kalyan", lw=1.0, color="C3")
                ax.plot(x, mean_other, label="Other Kalyan-thaat (mean)",
                        lw=1.0, color="C0", alpha=0.8)
                for c in svara_positions:
                    ax.axvline(c, color="gray", lw=0.3, alpha=0.4)
                ax.set_xticks(svara_positions)
                ax.set_xticklabels(svara_labels)
                ax.set_xlabel("Cents from tonic (octave-folded)")
                ax.set_ylabel("Probability")
                ax.set_title(f"Yaman vs. other Kalyan-thaat ragas  (JSD={jsd:.3f})")
                ax.legend()
                plt.tight_layout()
                fig.savefig(fig_dir / "yaman_vs_others.png", dpi=120)
                plt.close(fig)

            print(f"  Plots saved → {fig_dir}")
        except ImportError:
            print("  (matplotlib not installed; skipping plots)")

    print(f"\nDone. Features → {out}/")
    print(f"  per_track_features.json")
    print(f"  per_raga_histograms.json")
    print(f"  yaman_signature.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract raga features from Saraga pitch+tonic")
    parser.add_argument("--data_dir", default="data/saraga_kalyan_thaat",
                        help="Directory with extracted Saraga tracks")
    parser.add_argument("--output", default="data/raga_features",
                        help="Output directory for features")
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()
    run(args.data_dir, args.output, plot=not args.no_plots)
