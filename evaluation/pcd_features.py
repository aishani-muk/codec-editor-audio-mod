"""
Extended pitch-based features for raga identification ("PCD+").

Feature blocks (all tonic-normalised, octave-folded where applicable):

  1. Multi-resolution PCD                 — 3 smoothing scales concatenated
     sigmas ∈ {5, 15, 30} cents × 120 bins = 360 dims.
     Koduri et al. JNMR 2012 §4.3.

  2. Pitch-dyad distribution (PDD)        — 2-D histogram of consecutive
     (voiced) pitch-bin pairs, 24×24 = 576 dims. Captures melodic
     direction that a single-frame PCD is blind to. Koduri §5.

  3. Aaroha/avroha-split PCDs             — two 120-bin PCDs restricted to
     frames whose local slope (over ±window_frames) is > +min_delta_cents
     (aaroha) vs < -min_delta_cents (avroha). Catches ragas that use
     different svaras ascending vs descending (Yaman's Ni, Bageshri's Dha).

Total dim: 360 + 576 + 240 = 1176.

All four blocks are computed from the same augmented (pitch_hz, confidence,
tonic) triple so training augmentation stays internally consistent (one
sub-window, one cent-noise draw, one tonic jitter per clip).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d


# ── Feature-dim constants ────────────────────────────────────
N_BINS_PCD = 120                  # 10-cent bins, octave-folded
SIGMAS_CENTS = (5.0, 15.0, 30.0)  # multi-res smoothing
N_BINS_PDD = 24                   # 50-cent bins, octave-folded (both axes)
WINDOW_FRAMES = 20                # ±200 ms at hop 10 ms for slope estimate
MIN_DELTA_CENTS = 30.0            # threshold for aaroha/avroha classification
PCD_PLUS_DIM = (len(SIGMAS_CENTS) * N_BINS_PCD
                + N_BINS_PDD * N_BINS_PDD
                + 2 * N_BINS_PCD)
assert PCD_PLUS_DIM == 1176


# ── Pipeline primitives ──────────────────────────────────────


def _raw_to_cents(
    pitch_hz: np.ndarray,
    conf: np.ndarray,
    tonic_hz: float,
    conf_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Tonic-normalise pitch → unfolded cents (NaN for unvoiced).

    Returns ``(cents_unfolded, conf)``. ``cents_unfolded`` is 1-D same shape
    as pitch; voiced frames carry cent values in (typically) [-2400, +2400],
    unvoiced are NaN. ``conf`` is returned as-is (possibly augmented later).
    """
    cents = np.full_like(pitch_hz, np.nan, dtype=np.float64)
    mask = (pitch_hz > 0) & (conf >= conf_threshold)
    if mask.any():
        cents[mask] = 1200.0 * np.log2(
            pitch_hz[mask].astype(np.float64) / max(tonic_hz, 1e-8) + 1e-12
        )
    return cents, conf


def _augment(
    cents: np.ndarray,
    conf: np.ndarray,
    rng: np.random.Generator,
    tonic_jitter_cents: float = 0.0,
    frame_noise_cents: float = 0.0,
    frame_keep_prob: float = 1.0,
    subwindow_frac: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stochastically perturb (cents, conf) at training time.

    Order: sub-window → frame dropout → per-frame cent noise → global tonic
    shift. Returns fresh arrays (no mutation of inputs).
    """
    cents = cents.copy()
    conf = conf.copy()

    if subwindow_frac < 1.0 and len(cents) > 10:
        frac = rng.uniform(subwindow_frac, 1.0)
        keep_n = max(1, int(len(cents) * frac))
        start = int(rng.integers(0, len(cents) - keep_n + 1))
        cents = cents[start:start + keep_n]
        conf = conf[start:start + keep_n]

    if frame_keep_prob < 1.0:
        drop = rng.random(len(cents)) >= frame_keep_prob
        cents[drop] = np.nan
        conf[drop] = 0.0

    mask = ~np.isnan(cents)
    if frame_noise_cents > 0.0 and mask.any():
        noise = rng.normal(0.0, frame_noise_cents, size=cents.shape)
        cents[mask] = cents[mask] + noise[mask]

    if tonic_jitter_cents > 0.0 and mask.any():
        shift = rng.normal(0.0, tonic_jitter_cents)
        cents[mask] = cents[mask] + shift

    return cents, conf


# ── Feature blocks ───────────────────────────────────────────


def _multires_pcd(
    cents: np.ndarray,
    conf: np.ndarray,
    n_bins: int = N_BINS_PCD,
    sigmas_cents: Tuple[float, ...] = SIGMAS_CENTS,
) -> np.ndarray:
    """3-scale PCD concatenation → (len(sigmas) * n_bins,) float32."""
    mask = ~np.isnan(cents) & (conf > 0)
    total_dim = n_bins * len(sigmas_cents)
    if not mask.any():
        return np.ones(total_dim, dtype=np.float32) / total_dim

    folded = np.mod(cents[mask], 1200.0)
    weights = conf[mask]
    bin_width = 1200.0 / n_bins
    feats = []
    for sigma_c in sigmas_cents:
        hist, _ = np.histogram(folded, bins=n_bins, range=(0.0, 1200.0),
                               weights=weights)
        hist = gaussian_filter1d(hist, sigma=sigma_c / bin_width, mode="wrap")
        s = hist.sum()
        feats.append(hist / s if s > 0 else np.ones(n_bins) / n_bins)
    return np.concatenate(feats).astype(np.float32)


def _pitch_dyad_distribution(
    cents: np.ndarray,
    conf: np.ndarray,
    n_bins: int = N_BINS_PDD,
    dyad_stride: int = 5,
    smooth_sigma_bins: float = 1.0,
) -> np.ndarray:
    """2-D PDD at stride ``dyad_stride`` frames → (n_bins*n_bins,) float32.

    For each pair of voiced frames (t, t+stride), bin both by their
    octave-folded cents and accumulate √(conf_t * conf_{t+stride}).
    ``dyad_stride`` 5 at 10 ms hop ≈ 50 ms between paired frames — captures
    local melodic motion while avoiding self-loop dominance from sustained
    notes.
    """
    total_dim = n_bins * n_bins
    if len(cents) < 2 + dyad_stride:
        return np.ones(total_dim, dtype=np.float32) / total_dim

    folded = np.mod(cents, 1200.0)  # NaN preserved
    c1 = folded[:-dyad_stride]
    c2 = folded[dyad_stride:]
    w1 = conf[:-dyad_stride]
    w2 = conf[dyad_stride:]
    valid = ~np.isnan(c1) & ~np.isnan(c2) & (w1 > 0) & (w2 > 0)
    if valid.sum() < 1:
        return np.ones(total_dim, dtype=np.float32) / total_dim

    bin_width = 1200.0 / n_bins
    b1 = np.clip((c1[valid] / bin_width).astype(int), 0, n_bins - 1)
    b2 = np.clip((c2[valid] / bin_width).astype(int), 0, n_bins - 1)
    w = np.sqrt(np.clip(w1[valid] * w2[valid], 0.0, None))

    hist = np.zeros((n_bins, n_bins), dtype=np.float64)
    np.add.at(hist, (b1, b2), w)
    if smooth_sigma_bins > 0:
        hist = gaussian_filter(hist, sigma=smooth_sigma_bins, mode="wrap")
    s = hist.sum()
    if s <= 0:
        return np.ones(total_dim, dtype=np.float32) / total_dim
    return (hist / s).ravel().astype(np.float32)


def _aaroha_avroha_pcds(
    cents: np.ndarray,
    conf: np.ndarray,
    n_bins: int = N_BINS_PCD,
    window_frames: int = WINDOW_FRAMES,
    min_delta_cents: float = MIN_DELTA_CENTS,
    sigma_bins: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Slope-gated PCDs: (aaroha_pcd, avroha_pcd) each (n_bins,) float32.

    A frame is "aaroha" if ``cents[t+w] - cents[t-w] > +min_delta_cents`` and
    both neighbours are voiced; "avroha" if the slope is < -min_delta_cents.
    Frames with missing neighbours or |slope| ≤ threshold are excluded from
    both histograms.
    """
    mask = ~np.isnan(cents) & (conf > 0)
    if mask.sum() < 2 * window_frames + 1:
        u = np.ones(n_bins, dtype=np.float32) / n_bins
        return u, u.copy()

    T = len(cents)
    slope = np.full(T, np.nan)
    left = cents[:T - 2 * window_frames]
    right = cents[2 * window_frames:]
    centre = np.arange(window_frames, T - window_frames)
    delta = right - left
    ok = ~np.isnan(delta)
    slope[centre[ok]] = delta[ok]

    aaroha_mask = mask & ~np.isnan(slope) & (slope > min_delta_cents)
    avroha_mask = mask & ~np.isnan(slope) & (slope < -min_delta_cents)

    def _pcd(sel: np.ndarray) -> np.ndarray:
        if not sel.any():
            return np.ones(n_bins, dtype=np.float32) / n_bins
        folded = np.mod(cents[sel], 1200.0)
        w = conf[sel]
        hist, _ = np.histogram(folded, bins=n_bins, range=(0.0, 1200.0),
                               weights=w)
        hist = gaussian_filter1d(hist, sigma=sigma_bins, mode="wrap")
        s = hist.sum()
        return (hist / s).astype(np.float32) if s > 0 \
            else np.ones(n_bins, dtype=np.float32) / n_bins

    return _pcd(aaroha_mask), _pcd(avroha_mask)


# ── Public entry point ───────────────────────────────────────


def extract_pcd_plus(
    pitch_hz: np.ndarray,
    conf: np.ndarray,
    tonic_hz: float,
    conf_threshold: float = 0.5,
    aug_params: Optional[Dict[str, float]] = None,
    rng: Optional[np.random.Generator] = None,
    n_bins_pcd: int = N_BINS_PCD,
    sigmas_cents: Tuple[float, ...] = SIGMAS_CENTS,
    n_bins_pdd: int = N_BINS_PDD,
    dyad_stride: int = 5,
    window_frames: int = WINDOW_FRAMES,
    min_delta_cents: float = MIN_DELTA_CENTS,
) -> np.ndarray:
    """Full ``PCD_PLUS_DIM``-dim feature vector.

    Layout in the returned array:
        [0                                 : 360)  multi-res PCD (3 × 120)
        [360                               : 936)  PDD (24 × 24)
        [936                               : 1056) aaroha PCD (120)
        [1056                              : 1176) avroha PCD (120)

    Augmentation is applied ONCE to the ``(cents, conf)`` pair, then shared
    across all four feature blocks so they remain internally consistent.
    Pass ``aug_params=None`` (default) for deterministic inference.
    """
    cents, conf2 = _raw_to_cents(pitch_hz, conf, tonic_hz,
                                 conf_threshold=conf_threshold)
    if aug_params is not None and rng is not None:
        cents, conf2 = _augment(cents, conf2, rng, **aug_params)

    mpcd = _multires_pcd(cents, conf2,
                         n_bins=n_bins_pcd, sigmas_cents=sigmas_cents)
    pdd = _pitch_dyad_distribution(cents, conf2,
                                   n_bins=n_bins_pdd,
                                   dyad_stride=dyad_stride)
    apcd, dpcd = _aaroha_avroha_pcds(cents, conf2,
                                     n_bins=n_bins_pcd,
                                     window_frames=window_frames,
                                     min_delta_cents=min_delta_cents)
    return np.concatenate([mpcd, pdd, apcd, dpcd]).astype(np.float32)
