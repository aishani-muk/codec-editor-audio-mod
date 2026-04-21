"""
Thin wrapper around AMAAI-Lab's Music2Emo (ICLR-MER 2025) for per-clip
valence/arousal/mood prediction.

Why this over a hand-trained DEAM regressor
------------------------------------------
``train_deam_regressor.py`` was designed to train an in-house MLP on DEAM
alone, but DEAM audio is behind a MediaEval access agreement that blocks
automated download. Kang & Herremans 2025 ("Towards Unified Music Emotion
Recognition across Dimensional and Categorical Models", arXiv 2502.03979)
released a **multi-task** checkpoint trained jointly on DEAM + PMEmo +
EmoMusic + MTG-Jamendo that beats the MediaEval 2021 winner and publishes
both the code and the weights. We consume that directly.

Output
------
* ``valence`` on DEAM scale **1-9** (higher = more positive)
* ``arousal`` on DEAM scale **1-9** (higher = more energetic)
* ``moods``: a list of predicted MTG-Jamendo mood tags (binary, threshold 0.5)

The neutral target used by ``shift_toward_neutral`` is
``(valence=5.0, arousal=3.0)`` — mid valence, low arousal — i.e. the calm
contemplative profile this project is trying to steer listeners toward.

Availability
------------
The wrapper is a graceful no-op if ``third_party/Music2Emotion/`` is missing
(``available`` becomes False and callers can skip the metric).

Important note on cwd
---------------------
``Music2emo.predict`` uses *relative* paths (``./temp_out``, ``./output``,
``./inference/data/...``) and ``shutil.rmtree``s them on every call. We
therefore ``chdir`` into the Music2Emotion directory around each call and
restore the caller's cwd in a ``finally``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List


# ── Neutral target shared with legacy deam_regressor.shift_toward_neutral ─

NEUTRAL_VA = (5.0, 3.0)   # (valence, arousal) on DEAM 1-9 scale


def shift_toward_neutral(
    va_in: tuple, va_out: tuple,
    neutral: tuple = NEUTRAL_VA,
) -> Dict[str, float | bool]:
    """Summarise an (in, out) V/A transition against the neutral target."""
    import numpy as np
    d_in = float(np.hypot(va_in[0] - neutral[0], va_in[1] - neutral[1]))
    d_out = float(np.hypot(va_out[0] - neutral[0], va_out[1] - neutral[1]))
    return {
        "valence_in": float(va_in[0]), "arousal_in": float(va_in[1]),
        "valence_out": float(va_out[0]), "arousal_out": float(va_out[1]),
        "delta_valence": float(va_out[0] - va_in[0]),
        "delta_arousal": float(va_out[1] - va_in[1]),
        "dist_to_neutral_in": d_in,
        "dist_to_neutral_out": d_out,
        "moved_toward_neutral": bool(d_out < d_in),
    }


class Music2EmoRegressor:
    """Load once, predict many. ``available`` is False if Music2Emo is missing."""

    MUSIC2EMO_DIR = (
        Path(__file__).resolve().parents[1] / "third_party" / "Music2Emotion"
    )

    def __init__(self) -> None:
        self._model = None
        self._loaded = False
        self._avail = (
            self.MUSIC2EMO_DIR.is_dir()
            and (self.MUSIC2EMO_DIR / "music2emo.py").is_file()
            and (self.MUSIC2EMO_DIR / "saved_models" / "J_all.ckpt").is_file()
        )

    @property
    def available(self) -> bool:
        return self._avail

    # ── Lazy loader ──

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if not self._avail:
            raise RuntimeError(
                f"Music2Emo not found at {self.MUSIC2EMO_DIR}. "
                f"Clone https://github.com/AMAAI-Lab/Music2Emotion there."
            )
        orig_cwd = os.getcwd()
        orig_sys_path = list(sys.path)
        try:
            os.chdir(self.MUSIC2EMO_DIR)
            sys.path.insert(0, str(self.MUSIC2EMO_DIR))
            from music2emo import Music2emo   # type: ignore
            self._model = Music2emo()
        finally:
            os.chdir(orig_cwd)
            # Leave the sys.path entry; `music2emo` module may import more
            # sub-modules lazily on the first predict() call.
            if str(self.MUSIC2EMO_DIR) not in orig_sys_path:
                pass
        self._loaded = True

    # ── Public API ──

    def predict(self, wav_path: str | Path,
                mood_threshold: float = 0.5) -> Dict[str, float | List[str]]:
        """Run Music2Emo on a single WAV/MP3 and return a dict.

        Returns ``{"valence", "arousal", "moods"}``.
        Valence/arousal are on the DEAM 1-9 scale. ``moods`` is a list of
        MTG-Jamendo mood-theme tags scored above ``mood_threshold``.
        """
        self._ensure_loaded()
        abs_path = str(Path(wav_path).expanduser().resolve())
        orig_cwd = os.getcwd()
        try:
            os.chdir(self.MUSIC2EMO_DIR)
            out = self._model.predict(abs_path, threshold=mood_threshold)
        finally:
            os.chdir(orig_cwd)
        return {
            "valence": float(out["valence"]),
            "arousal": float(out["arousal"]),
            "moods": [str(m) for m in out.get("predicted_moods", [])],
        }

    def predict_pair(self, input_wav: str | Path,
                     output_wav: str | Path) -> Dict:
        """Convenience: predict on both sides and summarise the shift."""
        va_in = self.predict(input_wav)
        va_out = self.predict(output_wav)
        summary = shift_toward_neutral(
            (va_in["valence"], va_in["arousal"]),
            (va_out["valence"], va_out["arousal"]),
        )
        summary["moods_in"] = va_in["moods"]
        summary["moods_out"] = va_out["moods"]
        return summary
