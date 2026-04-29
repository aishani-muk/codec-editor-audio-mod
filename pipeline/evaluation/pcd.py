"""
Pitch-Class Distribution (PCD) computation and JSD comparison.

Methodology follows:
    Koduri, Gulati, Rao, Serra. "Raga recognition based on pitch distribution
    methods." Journal of New Music Research 41(4):337–350, 2012.

Specifically we implement the pitch-*salience*-weighted variant from §4.2:
each cent-bin is weighted by per-frame confidence (salience proxy), not by
uniform frame count. The distribution is then smoothed with a 30-cent
Gaussian kernel to absorb intonational jitter without erasing fine-grained
microtonal structure (the Gulati-Koduri literature uses 30 c; larger would
collapse shruti-level distinctions).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import jensenshannon


def salience_weighted_pcd(
    pitch_hz: np.ndarray,
    confidence: np.ndarray,
    tonic_hz: float,
    n_bins: int = 1200,
    sigma_cents: float = 30.0,
) -> np.ndarray:
    """Compute a tonic-normalised, octave-folded, salience-weighted PCD.

    Args:
        pitch_hz: (T,) f0 in Hz, 0 for unvoiced frames.
        confidence: (T,) per-frame salience proxy in [0, 1].
        tonic_hz: tonic to normalise against (in Hz).
        n_bins: resolution of the histogram; 1200 = 1 cent per bin.
        sigma_cents: Gaussian kernel width in cents (Koduri §4.2 uses 30).

    Returns:
        (n_bins,) normalised probability mass function.
    """
    mask = (pitch_hz > 0) & (confidence > 0)
    if not mask.any():
        return np.ones(n_bins, dtype=np.float64) / n_bins

    pf = pitch_hz[mask].astype(np.float64)
    wf = confidence[mask].astype(np.float64)

    cents = 1200.0 * np.log2(pf / max(tonic_hz, 1e-8) + 1e-12)
    folded = np.mod(cents, 1200.0)

    hist, _ = np.histogram(folded, bins=n_bins, range=(0.0, 1200.0),
                           weights=wf)

    # 30-cent Gaussian smoothing (bin width is 1 cent ⇒ sigma_bins == sigma_cents).
    hist = gaussian_filter1d(hist, sigma=sigma_cents, mode="wrap")

    total = hist.sum()
    if total <= 0:
        return np.ones(n_bins, dtype=np.float64) / n_bins
    return hist / total


def pcd_jsd(
    pitch_in: np.ndarray,
    conf_in: np.ndarray,
    pitch_out: np.ndarray,
    conf_out: np.ndarray,
    tonic_hz: float,
    n_bins: int = 1200,
    sigma_cents: float = 30.0,
) -> float:
    """Jensen-Shannon divergence between two salience-weighted PCDs.

    Returned value is JSD (the squared JS distance), in nats, clipped to [0, ln2].
    """
    P = salience_weighted_pcd(pitch_in, conf_in, tonic_hz, n_bins, sigma_cents)
    Q = salience_weighted_pcd(pitch_out, conf_out, tonic_hz, n_bins, sigma_cents)
    # scipy's jensenshannon returns the JS *distance* (sqrt of JSD).
    return float(jensenshannon(P, Q) ** 2)
