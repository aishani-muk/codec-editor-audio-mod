"""u(t) stress-intensity profile generators.

Every generator returns a 1-D numpy array of length ``n_frames`` in ``[0, 1]``.
The caller decides what ``n_frames`` means (tokens, codec frames, analysis
hops); we sample the profile over ``t = linspace(0, 1, n_frames)`` so all
profiles scale to any clip length.

Available presets
-----------------
  flat        u(t) = peak                                    — constant edit
  ramp_up     u(t) = 0 -> peak   (linear)                    — gradual onset
  ramp_down   u(t) = peak -> 0   (linear)                    — fade-out
  pulse       Gaussian bump centred at t=0.5 with peak ``peak``
              and width ``width`` (fraction of clip).
  arch        triangular bump: 0 -> peak -> 0   (like pulse
              but linear-sided)
  sine        (peak/2) * (1 + sin(2πft - π/2))  (starts at 0,
              oscillates between 0 and peak)
  keyframe    piecewise-linear through N user-supplied (t, u)
              knots with t in [0, 1].

Helpers
-------
  sample_profile_from_name(...)  → single dispatch
  profile_as_overlay(...)        → downsample profile for matplotlib overlay
  serialize_profile(...)         → save profile as .npy for reproducibility
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


PRESET_NAMES = ("flat", "ramp_up", "ramp_down", "pulse", "arch", "sine")


# ── Individual generators ────────────────────────────────────────────────
def flat(n_frames: int, peak: float = 0.5) -> np.ndarray:
    return np.full(n_frames, np.clip(peak, 0.0, 1.0), dtype=np.float32)


def ramp_up(n_frames: int, peak: float = 0.8) -> np.ndarray:
    return np.linspace(0.0, np.clip(peak, 0.0, 1.0),
                       n_frames, dtype=np.float32)


def ramp_down(n_frames: int, peak: float = 0.8) -> np.ndarray:
    return np.linspace(np.clip(peak, 0.0, 1.0), 0.0,
                       n_frames, dtype=np.float32)


def pulse(n_frames: int, peak: float = 0.9,
          width: float = 0.25, centre: float = 0.5) -> np.ndarray:
    """Gaussian bump — models a transient stress spike."""
    t = np.linspace(0.0, 1.0, n_frames, dtype=np.float32)
    sigma = max(1e-3, width / 2.355)   # FWHM -> sigma
    u = np.exp(-0.5 * ((t - centre) / sigma) ** 2) * np.clip(peak, 0.0, 1.0)
    return u.astype(np.float32)


def arch(n_frames: int, peak: float = 0.9,
         centre: float = 0.5) -> np.ndarray:
    """Linear up-then-down triangle centred at ``centre``."""
    t = np.linspace(0.0, 1.0, n_frames, dtype=np.float32)
    # Distance from centre, scaled so the farthest edge is 1.0.
    d_max = max(centre, 1 - centre)
    d = np.abs(t - centre) / max(d_max, 1e-6)
    u = np.clip(peak, 0.0, 1.0) * (1.0 - d)
    return np.clip(u, 0.0, 1.0).astype(np.float32)


def sine(n_frames: int, peak: float = 0.6,
         cycles: float = 2.0) -> np.ndarray:
    """Starts at 0, oscillates between 0 and ``peak``."""
    t = np.linspace(0.0, 1.0, n_frames, dtype=np.float32)
    u = (np.clip(peak, 0.0, 1.0) / 2.0) * (1.0 - np.cos(2 * np.pi * cycles * t))
    return u.astype(np.float32)


def keyframe(n_frames: int,
             knots: Iterable[tuple[float, float]]) -> np.ndarray:
    """Piecewise-linear through (t, u) knots. t values are in [0, 1]."""
    kn = sorted((float(ti), float(ui)) for ti, ui in knots)
    if not kn:
        return np.zeros(n_frames, dtype=np.float32)
    if kn[0][0] > 0.0:
        kn.insert(0, (0.0, kn[0][1]))
    if kn[-1][0] < 1.0:
        kn.append((1.0, kn[-1][1]))
    ts = np.array([t for t, _ in kn], dtype=np.float32)
    us = np.clip([u for _, u in kn], 0.0, 1.0).astype(np.float32)
    t = np.linspace(0.0, 1.0, n_frames, dtype=np.float32)
    return np.interp(t, ts, us).astype(np.float32)


# ── Dispatch ────────────────────────────────────────────────────────────
def sample_profile_from_name(name: str, n_frames: int, peak: float = 0.6,
                             width: float = 0.25, cycles: float = 2.0,
                             knots: Iterable[tuple[float, float]] | None = None
                             ) -> np.ndarray:
    """Return a u(t) array for one of the supported profile names."""
    name = (name or "flat").lower()
    if name == "flat":
        return flat(n_frames, peak)
    if name == "ramp_up":
        return ramp_up(n_frames, peak)
    if name == "ramp_down":
        return ramp_down(n_frames, peak)
    if name == "pulse":
        return pulse(n_frames, peak, width=width)
    if name == "arch":
        return arch(n_frames, peak)
    if name == "sine":
        return sine(n_frames, peak, cycles=cycles)
    if name == "keyframe":
        return keyframe(n_frames, knots or [(0.0, peak), (1.0, peak)])
    raise ValueError(f"Unknown profile {name!r}. "
                     f"Expected one of {PRESET_NAMES + ('keyframe',)}.")


# ── Helpers for the UI ──────────────────────────────────────────────────
@dataclass
class ProfileSpec:
    """User-facing spec for a u(t) profile.  Gradio state passes this
    through the Generate click handler."""
    name: str = "flat"
    peak: float = 0.5
    width: float = 0.25
    cycles: float = 2.0
    knots: tuple[tuple[float, float], ...] = ()

    def to_array(self, n_frames: int) -> np.ndarray:
        return sample_profile_from_name(
            self.name, n_frames,
            peak=self.peak, width=self.width, cycles=self.cycles,
            knots=self.knots if self.name == "keyframe" else None,
        )


def profile_as_overlay(u_arr: np.ndarray, n_points: int = 200
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Downsample u(t) for a smooth matplotlib / Plotly line plot."""
    if u_arr.size == 0:
        return np.zeros(n_points), np.zeros(n_points)
    if u_arr.size <= n_points:
        t = np.linspace(0, 1, u_arr.size, dtype=np.float32)
        return t, u_arr
    idx = np.linspace(0, u_arr.size - 1, n_points).round().astype(int)
    return np.linspace(0, 1, n_points, dtype=np.float32), u_arr[idx]


def serialize_profile(u_arr: np.ndarray, path: str) -> None:
    """Persist a profile next to its output WAV for reproducibility."""
    np.save(path, u_arr)
