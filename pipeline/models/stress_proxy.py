"""
Stress-proxy signal generation, WESAD-based inference, and embedding.

Three modes of producing s(t) ∈ [0,1]:
  1. SyntheticStressProxy  – deterministic ramp/decay profiles for controlled experiments
  2. WESADStressProxy      – trained on WESAD physiological data (ECG/EDA/Resp/Temp)
  3. StressEmbedding       – maps continuous s(t) → dense vector for the codec editor

Control mapping: u(t) = clip(a * s(t) + b, 0, 1)
  u(t) is what the editor sees.  a=1, b=0 by default (identity).
"""

import pickle
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np


# ─────────────────────────────────────────────────────────────
# 1. Control mapping  s(t) → u(t)
# ─────────────────────────────────────────────────────────────

def stress_to_edit_intensity(
    s: np.ndarray, a: float = 1.0, b: float = 0.0
) -> np.ndarray:
    """Map raw stress signal to edit intensity u(t)."""
    return np.clip(a * s + b, 0.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# 2. Synthetic stress-proxy generator
# ─────────────────────────────────────────────────────────────

class SyntheticStressProxy:
    """
    Generate deterministic stress trajectories for training and ablation.

    Profiles:
      "ramp"     – linear onset, sustained plateau
      "pulse"    – ramp up, hold, ramp down
      "episodic" – multiple stress bursts with recovery periods
    """

    def __init__(self, token_rate: float = 40.0):
        self.token_rate = token_rate

    def generate(
        self,
        duration_sec: float,
        peak: float = 0.6,
        onset_sec: float = 5.0,
        ramp_sec: float = 1.0,
        profile: str = "ramp",
        decay_sec: float = 3.0,
        n_episodes: int = 3,
    ) -> np.ndarray:
        """
        Returns:
            s: (n_tokens,) float32 in [0, 1]
        """
        n = int(duration_sec * self.token_rate)
        s = np.zeros(n, dtype=np.float32)

        on = int(onset_sec * self.token_rate)
        ramp = int(ramp_sec * self.token_rate)

        if profile == "ramp":
            if on < n:
                end = min(on + ramp, n)
                s[on:end] = np.linspace(0, peak, end - on)
                if end < n:
                    s[end:] = peak

        elif profile == "pulse":
            up_end = min(on + ramp, n)
            s[on:up_end] = np.linspace(0, peak, up_end - on)
            hold_end = min(up_end + int(2.0 * self.token_rate), n)
            s[up_end:hold_end] = peak
            decay_len = int(decay_sec * self.token_rate)
            down_end = min(hold_end + decay_len, n)
            if hold_end < n:
                s[hold_end:down_end] = np.linspace(
                    peak, 0, down_end - hold_end
                )

        elif profile == "episodic":
            spacing = max(1, (n - on) // n_episodes)
            for ep in range(n_episodes):
                ep_on = on + ep * spacing
                ep_up = min(ep_on + ramp, n)
                s[ep_on:ep_up] = np.linspace(0, peak, ep_up - ep_on)
                hold = min(ep_up + int(1.5 * self.token_rate), n)
                s[ep_up:hold] = peak
                dec = int(decay_sec * self.token_rate)
                ep_down = min(hold + dec, n)
                if hold < n:
                    s[hold:ep_down] = np.linspace(
                        peak, 0, ep_down - hold
                    )
        return s

    def generate_batch(
        self, durations: list[float], peaks: list[float] | None = None,
        profile: str = "ramp", **kwargs,
    ) -> list[np.ndarray]:
        peaks = peaks or [0.6] * len(durations)
        return [
            self.generate(d, p, profile=profile, **kwargs)
            for d, p in zip(durations, peaks)
        ]


# ─────────────────────────────────────────────────────────────
# 3. WESAD-trained stress proxy (real physiological signals)
# ─────────────────────────────────────────────────────────────

class WESADStressProxy:
    """
    Real-time stress inference using a WESAD-trained classifier.

    Trained by models/train_stress_classifier.py on WESAD chest-sensor data
    (ECG, EDA, Resp, Temp).  Outputs calibrated P(stress) ∈ [0,1] per window.

    For the audio pipeline the physiological stream is resampled to the
    token rate by repeating each window's probability across its hop.
    """

    def __init__(self, model_path: str, token_rate: float = 40.0):
        with open(model_path, "rb") as f:
            blob = pickle.load(f)
        self.clf = blob["model"]
        self.feature_names = blob["feature_names"]
        self.window_sec = blob["window_sec"]
        self.token_rate = token_rate

    def predict_window(self, feature_vector: np.ndarray) -> float:
        """Return P(stress) for a single feature window."""
        prob = self.clf.predict_proba(feature_vector.reshape(1, -1))[0, 1]
        return float(prob)

    def predict_stream(
        self, feature_windows: np.ndarray, total_tokens: int,
        hop_sec: float = 2.0,
    ) -> np.ndarray:
        """
        Map a sequence of feature windows to a token-rate stress trajectory.

        Args:
            feature_windows: (n_windows, n_features)
            total_tokens: desired output length in tokens
            hop_sec: seconds between successive feature windows

        Returns:
            s: (total_tokens,) float32 in [0,1]
        """
        probs = self.clf.predict_proba(feature_windows)[:, 1]
        hop_tokens = int(hop_sec * self.token_rate)
        s = np.zeros(total_tokens, dtype=np.float32)
        for i, p in enumerate(probs):
            start = i * hop_tokens
            end = min(start + hop_tokens, total_tokens)
            s[start:end] = p
        return s


# ─────────────────────────────────────────────────────────────
# 4. Stress embedding (torch module for the codec editor)
# ─────────────────────────────────────────────────────────────

class StressEmbedding(nn.Module):
    """
    Embed a per-token stress value u(t) ∈ [0,1] into a dense vector.

    Discretises into buckets → learned embedding → linear projection.
    This gives the transformer a rich, learnable representation of each
    stress level (vs. a raw scalar concatenation).
    """

    def __init__(self, embed_dim: int = 64, n_buckets: int = 64):
        super().__init__()
        self.n_buckets = n_buckets
        self.embed = nn.Embedding(n_buckets, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: (batch, seq_len) float tensor in [0, 1].
        Returns:
            (batch, seq_len, embed_dim) conditioning embeddings.
        """
        bucket_ids = (u * (self.n_buckets - 1)).long().clamp(0, self.n_buckets - 1)
        return self.proj(self.embed(bucket_ids))
