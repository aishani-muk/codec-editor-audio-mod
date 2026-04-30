""" Given ``(input_wav, model, u_array)`` produces a modulated output WAV plus
evaluation metrics + latency stats. Works with three back-ends:

  - "dsp"         : baselines/dsp_baseline.py  (no ML; chunk-wise DSP with
                    per-chunk u). Always available.
  - "editor"      : proposed codec editor via infer_stream.run_streaming_inference
                    adapted to take an arbitrary u(t) array. Falls back to
                    a friendly "checkpoint not ready" if
                    ``checkpoints/<run>/best/`` is absent.
  - "encodec_mlp" : baselines/encodec_mlp_baseline.py. (Loaded on demand;
                    reports unavailable if the baseline has no checkpoint.)

All three produce a 24 kHz mono WAV. The caller (``demo/app.py``) drives
both the profile selection and the metrics widget.

Usage (programmatic):
    from demo.runner import DemoRunner
    runner = DemoRunner(config_path="configs/proposed.yaml")
    result = runner.run(
        input_wav="path/in.wav",
        u_array=my_ut_array,               # 1-D float32 in [0, 1]
        model="dsp",
        output_wav="path/out.wav",
        compute_metrics=True,
    )
    print(result["metrics"], result["latency"])
"""
from __future__ import annotations

import os, sys, time
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ── Cross-fade helper shared by all back-ends ────────────────────────────
def _crossfade_append(mix: np.ndarray, chunk: np.ndarray,
                      xf_samples: int) -> np.ndarray:
    """Append ``chunk`` onto ``mix`` with a linear cross-fade of
    ``xf_samples`` samples. Used by the DSP back-end to blend per-chunk
    processing boundaries."""
    if mix.size == 0:
        return chunk.astype(np.float32, copy=False)
    xf = min(xf_samples, len(mix), len(chunk))
    if xf <= 0:
        return np.concatenate([mix, chunk])
    head = mix[:-xf]
    tail_mix = mix[-xf:]
    fade_out = np.linspace(1.0, 0.0, xf, dtype=np.float32)
    fade_in = 1.0 - fade_out
    blended = tail_mix * fade_out + chunk[:xf] * fade_in
    return np.concatenate([head, blended, chunk[xf:]]).astype(np.float32,
                                                               copy=False)


def _match_length(y: np.ndarray, n: int) -> np.ndarray:
    if len(y) == n:
        return y
    if len(y) > n:
        return y[:n]
    return np.pad(y, (0, n - len(y)))


# ── DSP back-end (always available) ──────────────────────────────────────
def _run_dsp(y: np.ndarray, sr: int, u_array: np.ndarray,
             chunk_sec: float = 0.5, xfade_sec: float = 0.05) -> np.ndarray:
    """Chunk-wise DSP with per-chunk u(t) sampled at chunk midpoints."""
    from baselines.dsp_baseline import build_dsp_chain

    chunk_n = int(round(chunk_sec * sr))
    xf_n = int(round(xfade_sec * sr))
    n_chunks = int(np.ceil(len(y) / chunk_n))

    if len(u_array) != n_chunks:
        # Resample profile to chunk grid by midpoint-indexing.
        t_chunk = (np.arange(n_chunks) + 0.5) / n_chunks
        t_u = np.linspace(0.0, 1.0, len(u_array), dtype=np.float32)
        u_chunk = np.interp(t_chunk, t_u, u_array).astype(np.float32)
    else:
        u_chunk = u_array.astype(np.float32)

    out = np.zeros(0, dtype=np.float32)
    for i in range(n_chunks):
        chunk = y[i * chunk_n : (i + 1) * chunk_n]
        if chunk.size == 0:
            continue
        board = build_dsp_chain(float(np.clip(u_chunk[i], 0.0, 1.0)), sr)
        processed = board(chunk.reshape(1, -1), sr).squeeze(0)
        processed = _match_length(processed, len(chunk))
        out = _crossfade_append(out, processed, xf_n)
    return _match_length(out, len(y))


# ── Editor back-end (requires a trained checkpoint) ──────────────────────
def _run_editor(input_wav: str, output_wav: str, u_array: np.ndarray,
                config_path: str, checkpoint_dir: str,
                device: str) -> dict:
    """Adapter around infer_stream.run_streaming_inference that takes a
    per-token u(t) rather than a proxy profile name."""
    from infer_stream import run_streaming_inference

    # The existing function accepts ``u_fixed=float``; here we supply a
    # per-token array via a monkey-patch on SyntheticStressProxy so the
    # run still threads through unchanged. Cheaper than refactoring.
    import models.stress_proxy as sp
    orig_generate = sp.SyntheticStressProxy.generate

    arr = u_array.astype(np.float32, copy=False)
    def _patched_generate(self, duration_sec, peak=0.6, onset_sec=5.0,
                          ramp_sec=1.0, profile="ramp"):
        # Resample user-provided u(t) to the proxy's token rate.
        n = int(round(duration_sec * self.token_rate))
        t = np.linspace(0.0, 1.0, n, dtype=np.float32)
        t_u = np.linspace(0.0, 1.0, len(arr), dtype=np.float32)
        return np.interp(t, t_u, arr).astype(np.float32)

    sp.SyntheticStressProxy.generate = _patched_generate
    try:
        run_streaming_inference(
            input_wav=input_wav,
            output_wav=output_wav,
            checkpoint_dir=checkpoint_dir,
            config_path=config_path,
            u_fixed=None,            # triggers the proxy path -> patched fn
            stress_profile="ramp",   # ignored by the patched generate
            peak=0.6,
            device=device,
        )
    finally:
        sp.SyntheticStressProxy.generate = orig_generate
    return {"ok": True}


# ── Main orchestrator ────────────────────────────────────────────────────
class DemoRunner:
    """Load-once-run-many wrapper used by the Gradio UI.

    The heavy components (Music2Emo, raga classifier, editor checkpoint)
    are lazy-loaded on first use so the Gradio server boots in <1 s even
    when the demo-day machine has no GPU.
    """

    def __init__(
        self,
        config_path: str = "configs/proposed.yaml",
        editor_run_name: str = "proposed_v1",
        raga_checkpoint: str | None = None,
        device: str | None = None,
        sr: int = 24000,
    ) -> None:
        self.config_path = config_path
        self.editor_run_name = editor_run_name
        self.editor_checkpoint_dir = str(
            REPO_ROOT / "checkpoints" / editor_run_name / "best"
        )
        self.raga_checkpoint = (
            raga_checkpoint
            or self._autodetect_raga_checkpoint()
        )
        self.device = device or ("cuda" if self._has_cuda() else "cpu")
        self.sr = sr
        self._raga_predictor = None
        self._emo_regressor = None
        self._checked_editor = False
        self._editor_available = False

    # ── availability probes ─────────────────────────────────────────
    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    @staticmethod
    def _autodetect_raga_checkpoint() -> str | None:
        """Prefer the full-Saraga v2 ckpt, then Kalyan v2, then v1."""
        candidates = [
            REPO_ROOT / "checkpoints" / "raga_classifier_pcd_v2_full" / "model.pt",
            REPO_ROOT / "checkpoints" / "raga_classifier_pcd_v2" / "model.pt",
            REPO_ROOT / "checkpoints" / "raga_classifier_pcd" / "model.pt",
        ]
        for c in candidates:
            if c.is_file():
                return str(c)
        return None

    def editor_available(self) -> bool:
        """True if the proposed editor has a loadable checkpoint."""
        if self._checked_editor:
            return self._editor_available
        self._checked_editor = True
        d = Path(self.editor_checkpoint_dir)
        self._editor_available = (
            d.is_dir()
            and (d / "model.pt").is_file()
            and (d / "stress_embed.pt").is_file()
        )
        return self._editor_available

    def model_availability(self) -> dict[str, bool]:
        """Exposed for the UI's 'Model' dropdown so disabled options are
        greyed out with a tooltip."""
        return {
            "dsp": True,
            "editor": self.editor_available(),
            "encodec_mlp": (
                REPO_ROOT / "checkpoints" / "encodec_mlp" / "model.pt"
            ).is_file(),
        }

    # ── Public API ─────────────────────────────────────────────────
    def run(
        self,
        input_wav: str,
        u_array: np.ndarray,
        model: Literal["dsp", "editor", "encodec_mlp"] = "dsp",
        output_wav: str | None = None,
        compute_metrics: bool = True,
        emotion: bool = False,
    ) -> dict:
        """Modulate ``input_wav`` with ``u_array`` and return a result dict.

        Returns
        -------
        dict with keys:
          output_wav  : path to the generated 24 kHz WAV
          model       : which backend ran
          n_samples   : output length in samples
          duration_s  : output length in seconds
          u_array     : the (possibly resampled) u(t) actually used
          metrics     : evaluate.evaluate_pair(...) result (or None)
          latency     : {"total_s", "load_s", "infer_s", "metrics_s",
                         "rtf"}  (rtf = infer_s / duration_s, <1 = realtime)
          ok          : True on success; False if unavailable/errored
          error       : error message when ok=False
        """
        t_total = time.time()
        output_wav = output_wav or self._default_output_name(input_wav, model)
        Path(output_wav).parent.mkdir(parents=True, exist_ok=True)

        # Load input at the target SR.
        t_load = time.time()
        import librosa
        y, _ = librosa.load(input_wav, sr=self.sr, mono=True)
        duration_s = float(len(y) / self.sr)
        load_s = time.time() - t_load

        infer_s = 0.0
        err = None
        try:
            t_infer = time.time()
            if model == "dsp":
                y_out = _run_dsp(y, self.sr, u_array)
                sf.write(output_wav, y_out, self.sr, subtype="PCM_16")
            elif model == "editor":
                if not self.editor_available():
                    raise RuntimeError(
                        f"Editor checkpoint not yet available at "
                        f"{self.editor_checkpoint_dir}. Training is still "
                        f"running; try the DSP baseline for now."
                    )
                _run_editor(
                    input_wav=input_wav,
                    output_wav=output_wav,
                    u_array=u_array,
                    config_path=self.config_path,
                    checkpoint_dir=self.editor_checkpoint_dir,
                    device=self.device,
                )
            elif model == "encodec_mlp":
                raise RuntimeError(
                    "EnCodec+MLP baseline is not wired into the demo "
                    "runner yet."
                )
            else:
                raise ValueError(f"Unknown model {model!r}")
            infer_s = time.time() - t_infer
        except Exception as exc:
            err = repr(exc)

        # Metrics (optional; wraps evaluate.evaluate_pair with our lazy-
        # loaded predictors).
        metrics = None
        metrics_s = 0.0
        if err is None and compute_metrics:
            t_m = time.time()
            try:
                metrics = self._compute_metrics(
                    input_wav, output_wav, with_emotion=emotion
                )
            except Exception as exc:
                metrics = {"error": repr(exc)}
            metrics_s = time.time() - t_m

        total_s = time.time() - t_total
        result = {
            "output_wav": output_wav,
            "model": model,
            "n_samples": int(len(y)),
            "duration_s": duration_s,
            "u_array": u_array.astype(np.float32, copy=False),
            "metrics": metrics,
            "latency": {
                "total_s": round(total_s, 3),
                "load_s": round(load_s, 3),
                "infer_s": round(infer_s, 3),
                "metrics_s": round(metrics_s, 3),
                "rtf": round(infer_s / max(duration_s, 1e-6), 3),
            },
            "ok": err is None,
            "error": err,
        }
        return result

    # ── Metrics: lazy-load raga + Music2Emo, then reuse evaluate.evaluate_pair ──
    def _compute_metrics(self, input_wav: str, output_wav: str,
                         with_emotion: bool) -> dict:
        # Lazy-load predictors on first call.
        if self._raga_predictor is None and self.raga_checkpoint:
            from evaluation.raga_classifier import RagaPredictor
            self._raga_predictor = RagaPredictor(self.raga_checkpoint)
        if with_emotion and self._emo_regressor is None:
            from evaluation.emotion_regressor import Music2EmoRegressor
            self._emo_regressor = Music2EmoRegressor()

        # Resolve tonic (prefer Saraga .ctonic.txt sidecar; fall back to
        # Essentia TonicIndianArtMusic; final fallback = Sa of C4).
        from evaluation.tonic import resolve_tonic
        tonic_hz, _src = resolve_tonic(
            input_wav,
            stem=Path(input_wav).stem,
            tonic_dir=str(REPO_ROOT / "data" / "saraga_kalyan_thaat"),
            default_hz=261.63,
        )

        from evaluate import evaluate_pair
        return evaluate_pair(
            input_wav=input_wav,
            output_wav=output_wav,
            tonic_hz=tonic_hz,
            raga_predictor=self._raga_predictor,
            emotion_regressor=(self._emo_regressor
                               if with_emotion else None),
        )

    # ── Misc utilities ──────────────────────────────────────────────
    @staticmethod
    def _default_output_name(input_wav: str, model: str) -> str:
        stem = Path(input_wav).stem
        return str(
            REPO_ROOT / "results" / "demo_outputs"
            / f"{stem}__{model}__{int(time.time())}.wav"
        )
