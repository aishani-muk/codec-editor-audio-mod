"""
Inference-time wrapper around the DEAM valence/arousal regressor.

Target neutral profile (low arousal, mid valence) is taken at the centre
of the DEAM rating scale (1–9) biased slightly toward positive valence:
``(valence=5.0, arousal=3.0)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class _VARegressor(nn.Module):
    def __init__(self, in_dim: int = 768, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


NEUTRAL_VA = (5.0, 3.0)   # (valence, arousal) on DEAM 1–9 scale


class DEAMPredictor:
    """Load once, predict many. ``None`` if no checkpoint exists."""

    def __init__(self, checkpoint: str = "checkpoints/deam_regressor/model.pt",
                 device: str | None = None):
        self.checkpoint = Path(checkpoint)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if not self.checkpoint.exists():
            self.model = None
            return
        state = torch.load(self.checkpoint, map_location=self.device,
                           weights_only=False)
        self.model = _VARegressor(
            in_dim=state.get("in_dim", 768),
            hidden=state.get("hidden", 256),
        ).to(self.device).eval()
        self.model.load_state_dict(state["state_dict"])
        self.meta = {k: v for k, v in state.items() if k != "state_dict"}

    @property
    def available(self) -> bool:
        return self.model is not None

    @torch.no_grad()
    def predict_embedding(self, mert_embedding: np.ndarray
                          ) -> Tuple[float, float]:
        x = torch.from_numpy(mert_embedding).float().unsqueeze(0).to(self.device)
        y = self.model(x).squeeze(0).cpu().numpy()
        return float(y[0]), float(y[1])


def shift_toward_neutral(
    va_in: Tuple[float, float], va_out: Tuple[float, float],
    neutral: Tuple[float, float] = NEUTRAL_VA,
) -> dict:
    """Summarise an (in, out) V/A transition against the neutral target."""
    d_in = np.hypot(va_in[0] - neutral[0], va_in[1] - neutral[1])
    d_out = np.hypot(va_out[0] - neutral[0], va_out[1] - neutral[1])
    return {
        "valence_in": va_in[0], "arousal_in": va_in[1],
        "valence_out": va_out[0], "arousal_out": va_out[1],
        "delta_valence": va_out[0] - va_in[0],
        "delta_arousal": va_out[1] - va_in[1],
        "dist_to_neutral_in": float(d_in),
        "dist_to_neutral_out": float(d_out),
        "moved_toward_neutral": bool(d_out < d_in),
    }
