"""MERT-v1-95M wrapper: 768-dim time-mean pooled embeddings per clip.

Loads the model once; ``embed_wav_file`` and ``embed_batch`` accept any sample
rate and resample to 24 kHz internally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch


class MERTFeatureExtractor:
    """Singleton-friendly MERT wrapper (load once, embed many)"""

    hub_name = "m-a-p/MERT-v1-95M"
    local_dir = "third_party/MERT-v1-95M-local"
    target_sr = 24_000
    embed_dim = 768

    def __init__(self, device: str | None = None, dtype: str = "float32"):
        from pathlib import Path as _P
        from transformers import AutoFeatureExtractor, AutoModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, dtype)

        source = (
            self.local_dir
            if (_P(self.local_dir) / "model.safetensors").is_file()
            else self.hub_name
        )
        self.fe = AutoFeatureExtractor.from_pretrained(
            source, trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            source, trust_remote_code=True,
        ).to(self.device).to(self.dtype).eval()

    # ── Core embedding paths ──

    @torch.no_grad()
    def embed_waveform(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Embed a single 1-D waveform and return a 768-dim vector.

        Waveform is resampled to 24 kHz (MERT's training rate), fed through
        MERT, and the last hidden state is mean-pooled over time.
        """
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0 if waveform.shape[0] < waveform.shape[1]
                                       else 1)
        if sr != self.target_sr:
            import torchaudio
            x = torch.from_numpy(waveform.astype(np.float32)).unsqueeze(0)
            x = torchaudio.functional.resample(x, sr, self.target_sr)
            waveform = x.squeeze(0).numpy()

        inputs = self.fe(waveform, sampling_rate=self.target_sr,
                         return_tensors="pt")
        input_values = inputs["input_values"].to(self.device).to(self.dtype)

        outputs = self.model(input_values, output_hidden_states=False)
        # last_hidden_state: (1, T, 768)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze(0)
        return emb.float().cpu().numpy()

    def embed_wav_file(self, path: str | Path) -> np.ndarray:
        import soundfile as sf
        y, sr = sf.read(str(path), always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=-1)
        return self.embed_waveform(y.astype(np.float32), sr)

    def embed_batch(self, paths: Iterable[str | Path]) -> np.ndarray:
        vecs = [self.embed_wav_file(p) for p in paths]
        return np.stack(vecs, axis=0)
