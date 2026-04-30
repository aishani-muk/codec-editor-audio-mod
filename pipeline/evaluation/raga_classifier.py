"""Raga-ID classifier inference wrapper.

Auto-selects MERT (768-dim) or PCD (120-dim) features from the checkpoint's
``feature_type`` field. ``available`` is False if the checkpoint is missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


# Must mirror train_raga_classifier.RagaClassifier.
class _RagaClassifier(nn.Module):
    def __init__(self, n_classes: int, in_dim: int = 768, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class RagaPredictor:
    """``available`` is False if the checkpoint is missing."""

    def __init__(self, checkpoint: str = "checkpoints/raga_classifier/model.pt",
                 device: str | None = None):
        self.checkpoint = Path(checkpoint)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if not self.checkpoint.exists():
            self.model = None
            self.vocab = None
            self.inv_vocab = None
            return

        state = torch.load(self.checkpoint, map_location=self.device,
                           weights_only=False)
        self.vocab: Dict[str, int] = state["vocab"]
        self.inv_vocab: Dict[int, str] = {v: k for k, v in self.vocab.items()}

        # Feature type: "mert" (768-dim), "pcd" (N-bin histogram), or
        # "pcd_v2" (1176-dim PCD+ = multi-res PCD + PDD + aaroha/avroha).
        # Default to "mert" for legacy checkpoints.
        self.feature_type: str = state.get("feature_type", "mert")
        if self.feature_type == "pcd":
            in_dim = state.get("n_bins", 120)
        elif self.feature_type == "pcd_v2":
            from .pcd_features import PCD_PLUS_DIM
            in_dim = state.get("in_dim", PCD_PLUS_DIM)
        else:
            in_dim = state.get("in_dim", 768)

        hidden = state.get("hidden", 256)

        if self.feature_type == "pcd_v2":
            # Matches PCDPlusClassifier in train_raga_classifier_pcd_v2.py
            # (named encoder + head so SimCLR-pretrained weights map in).
            class _PCDPlusClassifier(nn.Module):
                def __init__(self, n_classes, in_dim, hidden):
                    super().__init__()
                    self.encoder = nn.Sequential(
                        nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.4),
                        nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.4),
                    )
                    self.head = nn.Linear(hidden, n_classes)

                def forward(self, x):
                    return self.head(self.encoder(x))

            # state_dict keys look like "encoder.net.0.weight" — remap to
            # "encoder.0.weight" since we inline .net here.
            raw = state["state_dict"]
            remapped = {}
            for k, v in raw.items():
                if k.startswith("encoder.net."):
                    remapped[k.replace("encoder.net.", "encoder.")] = v
                else:
                    remapped[k] = v

            self.model = _PCDPlusClassifier(
                n_classes=len(self.vocab),
                in_dim=in_dim,
                hidden=hidden,
            ).to(self.device).eval()
            self.model.load_state_dict(remapped)
        else:
            self.model = _RagaClassifier(
                n_classes=len(self.vocab),
                in_dim=in_dim,
                hidden=hidden,
            ).to(self.device).eval()
            self.model.load_state_dict(state["state_dict"])
        self.meta = {k: v for k, v in state.items()
                     if k not in {"state_dict", "vocab"}}

    @property
    def available(self) -> bool:
        return self.model is not None

    @torch.no_grad()
    def predict(self, feature_vector: np.ndarray) -> Tuple[str, float]:
        """Classify a pre-computed feature vector.
        """
        x = torch.from_numpy(np.asarray(feature_vector)).float() \
            .unsqueeze(0).to(self.device)
        logits = self.model(x).squeeze(0)
        probs = torch.softmax(logits, dim=-1)
        idx = int(probs.argmax().item())
        return self.inv_vocab[idx], float(probs[idx].item())

    def predict_from_audio(
        self,
        audio: Union[str, Path, np.ndarray],
        sr: int | None = None,
        tonic_hz: float = 261.63,
    ) -> Tuple[str, float]:
        """Raw waveform → feature → classify. Returns ``(raga_name, confidence ∈ [0,1])``.

        ``audio`` may be a path or a 1-D ndarray (``sr`` required for ndarrays).
        ``tonic_hz`` only matters for PCD checkpoints.
        """
        if not self.available:
            raise RuntimeError("RagaPredictor: no checkpoint loaded")

        if isinstance(audio, (str, Path)):
            import soundfile as sf
            y, sr_file = sf.read(str(audio), always_2d=False)
            if y.ndim > 1:
                y = y.mean(axis=-1)
            audio = y.astype(np.float32)
            sr = int(sr_file)
        elif sr is None:
            raise ValueError("sr is required when audio is an ndarray")

        if self.feature_type == "pcd":
            feat = self._extract_pcd(audio, sr, tonic_hz)
        elif self.feature_type == "pcd_v2":
            feat = self._extract_pcd_plus(audio, sr, tonic_hz)
        elif self.feature_type == "mert":
            from .mert_features import MERTFeatureExtractor
            if not hasattr(self, "_mert"):
                self._mert = MERTFeatureExtractor(device=self.device)
            feat = self._mert.embed_waveform(audio, sr=sr)
        else:
            raise ValueError(f"Unknown feature_type: {self.feature_type!r}")

        return self.predict(feat)

    # ── PCD+ feature extraction (mirrors train_raga_classifier_pcd_v2) ─

    def _extract_pcd_plus(self, audio: np.ndarray, sr: int,
                          tonic_hz: float) -> np.ndarray:
        """PESTO → extract_pcd_plus (multi-res PCD + PDD + aaroha/avroha PCDs)."""
        from .pitch import extract_pitch_with_confidence
        from .pcd_features import extract_pcd_plus

        hop_ms = float(self.meta.get("hop_ms", 10.0))
        pitch_hz, conf = extract_pitch_with_confidence(
            audio.astype(np.float32), sr=sr, hop_ms=hop_ms,
            conf_threshold=0.5,
        )

        pitch_hz = np.where(conf >= 0.5, pitch_hz, 0.0)
        return extract_pcd_plus(
            pitch_hz, conf, tonic_hz,
            aug_params=None, rng=None,   # deterministic inference
        )

    # ── PCD feature extraction (mirrors train_raga_classifier_pcd.make_pcd) ──

    def _extract_pcd(self, audio: np.ndarray, sr: int,
                     tonic_hz: float) -> np.ndarray:
        """PESTO pitch → tonic-normalised → salience-weighted PCD.

        Must match the training-time pipeline in
        ``train_raga_classifier_pcd.make_pcd`` (no augmentation).
        """
        from scipy.ndimage import gaussian_filter1d
        from .pitch import extract_pitch_with_confidence

        n_bins = int(self.meta.get("n_bins", 120))
        sigma_bins = float(self.meta.get("sigma_bins", 3.0))
        hop_ms = float(self.meta.get("hop_ms", 10.0))

        pitch_hz, conf = extract_pitch_with_confidence(
            audio.astype(np.float32), sr=sr, hop_ms=hop_ms,
        )
        mask = (pitch_hz > 0) & (conf > 0)
        if not mask.any():
            return np.ones(n_bins, dtype=np.float32) / n_bins

        p = pitch_hz[mask]
        w = conf[mask]
        cents = 1200.0 * np.log2(p / max(tonic_hz, 1e-8) + 1e-12)
        cents = np.mod(cents, 1200.0)

        hist, _ = np.histogram(cents, bins=n_bins, range=(0.0, 1200.0),
                               weights=w)
        hist = gaussian_filter1d(hist, sigma=sigma_bins, mode="wrap")
        total = hist.sum()
        if total <= 0:
            return np.ones(n_bins, dtype=np.float32) / n_bins
        return (hist / total).astype(np.float32)
