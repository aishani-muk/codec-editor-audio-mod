"""Pitch + per-frame confidence via PESTO → CREPE → pyin fallback chain.

Returns (pitch_hz, confidence) at the same hop regardless of which backend ran.
"""

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np


def extract_pitch_with_confidence(
    audio: np.ndarray,
    sr: int = 24000,
    hop_ms: float = 10.0,
    conf_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (pitch_hz, confidence) with low-confidence frames zeroed.
    Shape: both arrays are 1-D of length approximately ``len(audio) / (sr * hop_ms / 1000)``.
    """
    # ── 1. PESTO (primary, 2024) ──
    try:
        import torch
        import pesto

        a = torch.from_numpy(audio.astype(np.float32))
        _, pitch, conf, _ = pesto.predict(a, sr=sr, step_size=hop_ms)
        pitch = pitch.squeeze().cpu().numpy().astype(np.float32)
        conf = conf.squeeze().cpu().numpy().astype(np.float32)
        pitch = np.where(conf < conf_threshold, 0.0, pitch)
        return pitch, conf
    except ImportError:
        pass
    except Exception as e:  # torch or pesto internal error — don't abort eval
        warnings.warn(f"PESTO failed ({type(e).__name__}: {e}); falling back to CREPE")

    # ── 2. CREPE (fallback) ──
    try:
        import crepe

        _, pitch, conf, _ = crepe.predict(
            audio, sr, step_size=hop_ms, viterbi=True
        )
        pitch = pitch.astype(np.float32)
        conf = conf.astype(np.float32)
        pitch = np.where(conf < conf_threshold, 0.0, pitch)
        return pitch, conf
    except ImportError:
        pass

    # ── 3. librosa.pyin (final) ──
    import librosa
    hop_len = int(sr * hop_ms / 1000)
    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio, fmin=50, fmax=800, sr=sr, hop_length=hop_len,
    )
    f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)
    # voiced_prob can be None when pyin falls back; use voiced_flag as a bool
    if voiced_prob is None:
        voiced_prob = voiced_flag.astype(np.float32)
    else:
        voiced_prob = voiced_prob.astype(np.float32)
    return f0, voiced_prob
