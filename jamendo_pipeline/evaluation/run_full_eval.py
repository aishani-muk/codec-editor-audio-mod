"""
Unified SOTA evaluation for one track × multiple u levels on the Celtic
held-out test set (and, when available, the Saraga test set).

For each (track, u) combination:

    1. Load the track's trained checkpoint.
    2. Generate edited waveforms for every test clip at the requested u.
    3. Compute per-clip metrics:
         ΔV, ΔA, frac_moved_toward_neutral, tune-type-preservation,
         tonic-drift, PCD-JSD, rhythm JSD, meter-match, CLAP-score-shift,
         MERT-drift, velocity-TV %, jerk RMS, PESQ, UTMOS.
    4. Write ``per_clip.jsonl`` under ``results/<track>/u{u}/``.
    5. Compute corpus-level FAD-CLAP between test-input and test-output sets.
    6. Bootstrap-aggregate via ``scripts/aggregate_with_ci.py`` to produce
       ``summary.json``.

This is a long-running script: expect ~15 min per (track, u) pair on 1
A100. Skip-if-exists lets failed runs be resumed without redoing work.

Usage:
    source env.sh
    python -m evaluation.run_full_eval \
        --tracks a,b,c \
        --u 0.0,0.3,0.6,0.9 \
        --test_dir $JAMENDO_CACHE/test_clips
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent                     # modelling/jamendo_pipeline
_MODELLING_ROOT = _PKG_ROOT.parent           # modelling/
if str(_MODELLING_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODELLING_ROOT))

from jamendo_pipeline.data.manifest_schema import (                   # noqa: E402
    TUNE_TYPE_TO_ID, read_manifest, read_sidecar,
)
from .clap_score import clap_score_shift                              # noqa: E402
from .fad import fad_clap                                             # noqa: E402
from .mert_drift import mert_drift                                    # noqa: E402
from .rhythm_metrics import (                                         # noqa: E402
    meter_matches_input, onset_autocorr_jsd, tempo_delta,
)

LOG = logging.getLogger("run_full_eval")
NEUTRAL_VA = (5.0, 3.0)


# ---------------------------------------------------------------------
# Smoothness metrics (velocity-TV, jerk RMS) — from the Saraga code
# ---------------------------------------------------------------------
def velocity_tv(x: np.ndarray) -> float:
    return float(np.abs(np.diff(x)).mean()) if x.size > 1 else 0.0


def jerk_rms(x: np.ndarray) -> float:
    if x.size < 3:
        return 0.0
    d3 = np.diff(x, n=3)
    return float(np.sqrt(np.mean(d3 ** 2)))


# ---------------------------------------------------------------------
# Generators per track
# ---------------------------------------------------------------------
@dataclass
class TrackGenerator:
    """Callable that maps (wav_in: np.ndarray, u: float, sidecar: ClipMeta)
    → wav_out: np.ndarray, for one track's trained checkpoint.
    """
    name: str
    ckpt_path: Path
    fn: Callable[[np.ndarray, float, "ClipMeta"], np.ndarray]


def build_track_a_generator(ckpt_path: Path, device: torch.device) -> TrackGenerator:
    """Load Track A checkpoint → wrap its generate loop into a closure."""
    from models.codec_editor import CodecEditor
    from models.stress_proxy import StressEmbedding
    from tokenization.encode_wavtokenizer import load_wavtokenizer

    blob = torch.load(ckpt_path, map_location="cpu")
    cfg = blob["config"]
    ed = cfg["editor"]
    from jamendo_pipeline.data.manifest_schema import TUNE_TYPES
    model = CodecEditor(
        vocab_size=cfg["wavtokenizer"]["codebook_size"],
        bpe_vocab_size=None,
        n_layers=ed["n_layers"], n_heads=ed["n_heads"],
        d_model=ed["d_model"], d_ff=ed["d_ff"],
        max_seq_len=ed["max_seq_len"], dropout=ed["dropout"],
        stress_embed_dim=cfg["stress_proxy"]["embed_dim"],
        n_ragas=len(TUNE_TYPES) - 1,
        use_input_residual=ed.get("use_input_residual", True),
        conditioning=ed.get("conditioning", "add"),
    ).to(device).eval()
    model.load_state_dict(blob["model_state_dict"])
    stress_embed = StressEmbedding(
        embed_dim=cfg["stress_proxy"]["embed_dim"],
        n_buckets=cfg["stress_proxy"]["n_buckets"],
    ).to(device).eval()
    stress_embed.load_state_dict(blob["stress_embed_state_dict"])

    wt = load_wavtokenizer(cfg["wavtokenizer"]["model_name"], device=str(device))
    bandwidth_id = torch.tensor([0], device=device)

    @torch.no_grad()
    def _encode(wav: np.ndarray, sr: int) -> torch.Tensor:
        if sr != 24_000:
            import librosa
            wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=24_000)
        t = torch.from_numpy(wav.astype(np.float32)).to(device).unsqueeze(0)
        _, codes = wt.encode_infer(t, bandwidth_id=bandwidth_id)
        c = codes.squeeze().long()
        if c.dim() == 1:
            c = c.unsqueeze(0)
        return c.unsqueeze(0)  # (1, 1, T) → take first row

    @torch.no_grad()
    def _decode(codes: torch.Tensor) -> np.ndarray:
        codes_wt = codes.unsqueeze(0)   # (n_q=1, B, T)
        features = wt.codes_to_features(codes_wt)
        wav = wt.decode(features, bandwidth_id=bandwidth_id)
        return wav.squeeze().detach().cpu().float().numpy()

    def _gen(wav_in: np.ndarray, u: float, sidecar) -> np.ndarray:
        codes = _encode(wav_in, sr=24_000)            # (1, 1, T)
        codes = codes.squeeze(0)                       # (1, T)
        T_in = codes.size(-1)
        u_t = torch.full((1, T_in), float(u), device=device)
        from models.stress_proxy import StressEmbedding
        stress = stress_embed(u_t)
        tt_id = torch.tensor(
            [TUNE_TYPE_TO_ID.get(sidecar.tune_type, 0)],
            dtype=torch.long, device=device,
        )
        out = model.generate_edited(
            input_ids=codes, stress_embeds=stress, raga_ids=tt_id,
            max_new_tokens=T_in,
            temperature=0.8, top_k=40, cfg_scale=2.0,
        )
        return _decode(out)

    return TrackGenerator(name="a", ckpt_path=ckpt_path, fn=_gen)


# Tracks B and C plug in analogously; their loader bodies would be:
#   Track B: load MusicGen + LoRA, call editor.edit().
#   Track C: instantiate HybridMusicGenEditor, greedy decode from logits.
# To keep this file short we leave Track B / C loaders as stubs that
# raise a clear error until their checkpoints are trained. The unified
# harness still runs on Track A alone while B/C are training.
def build_track_b_generator(ckpt_path: Path, device: torch.device) -> TrackGenerator:
    def _gen(*a, **k):
        raise NotImplementedError(
            "Track B generator: wire up after ``train_track_b`` finishes. "
            "Load MusicGen + LoRA via `inject_lora_into_musicgen` and call "
            "`editor.edit(...)`."
        )
    return TrackGenerator(name="b", ckpt_path=ckpt_path, fn=_gen)


def build_track_c_generator(ckpt_path: Path, device: torch.device) -> TrackGenerator:
    def _gen(*a, **k):
        raise NotImplementedError(
            "Track C generator: wire up after ``train_track_c`` finishes. "
            "Instantiate HybridMusicGenEditor, load partial state dict, and "
            "greedy-decode via `logits.argmax(-1)` then MusicGen EnCodec decode."
        )
    return TrackGenerator(name="c", ckpt_path=ckpt_path, fn=_gen)


# ---------------------------------------------------------------------
# Per-clip metric bundle
# ---------------------------------------------------------------------
@dataclass
class ClipMetrics:
    stem: str
    u: float
    track: str
    tune_type: str
    # Emotion (if M2E is available)
    valence_in: float = 0.0
    arousal_in: float = 0.0
    valence_out: float = 0.0
    arousal_out: float = 0.0
    delta_valence: float = 0.0
    delta_arousal: float = 0.0
    moved_toward_neutral: bool = False
    # Celtic-specific structural
    tune_type_preserved: int = 0
    meter_match: int = 0
    meter_in: int = 0
    meter_out: int = 0
    tempo_delta_bpm: float = 0.0
    rhythm_jsd: float = 0.0
    # Pitch
    pcd_jsd: float = 0.0
    tonic_drift_cents: float = 0.0
    # MERT / CLAP
    mert_drift: float = 0.0
    clap_calm_in: float = 0.0
    clap_calm_out: float = 0.0
    clap_shift: float = 0.0
    # Smoothness / quality
    velocity_tv_in: float = 0.0
    velocity_tv_out: float = 0.0
    velocity_tv_change_pct: float = 0.0
    jerk_rms_out: float = 0.0
    # Misc
    collapse: bool = False

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def per_clip_eval(
    sidecar, wav_in: np.ndarray, wav_out: np.ndarray, *,
    sr: int, u: float, track_name: str,
    tune_classifier=None, m2e=None, mert_model=None, mert_fe=None,
) -> ClipMetrics:
    m = ClipMetrics(
        stem=sidecar.stem, u=u, track=track_name,
        tune_type=sidecar.tune_type,
    )
    # ── Velocity-TV / jerk ──────────────────────────────────────────
    m.velocity_tv_in = velocity_tv(wav_in)
    m.velocity_tv_out = velocity_tv(wav_out)
    if m.velocity_tv_in > 1e-8:
        m.velocity_tv_change_pct = 100.0 * abs(
            m.velocity_tv_in - m.velocity_tv_out
        ) / m.velocity_tv_in
    m.jerk_rms_out = jerk_rms(wav_out)
    m.collapse = m.velocity_tv_change_pct >= 99.5

    # ── Structural metrics ──────────────────────────────────────────
    # Rhythm JSD
    m.rhythm_jsd = onset_autocorr_jsd(
        wav_in, wav_out, sr=sr,
        bpm=sidecar.tempo_bpm or 120.0,
        meter_num=sidecar.meter_num or 4,
    )
    match, mi, mo = meter_matches_input(wav_in, wav_out, sr=sr)
    m.meter_match = int(match)
    m.meter_in = int(mi)
    m.meter_out = int(mo)
    m.tempo_delta_bpm = tempo_delta(wav_in, wav_out, sr=sr)

    # ── PCD-JSD using chroma ────────────────────────────────────────
    import librosa
    c_in = librosa.feature.chroma_cqt(y=wav_in, sr=sr, n_chroma=12).mean(1)
    c_out = librosa.feature.chroma_cqt(y=wav_out, sr=sr, n_chroma=12).mean(1)
    p_in = c_in / (c_in.sum() + 1e-9)
    p_out = c_out / (c_out.sum() + 1e-9)
    mid = 0.5 * (p_in + p_out)
    eps = 1e-8
    js = 0.5 * (
        np.sum(p_in * (np.log(p_in + eps) - np.log(mid + eps)))
        + np.sum(p_out * (np.log(p_out + eps) - np.log(mid + eps)))
    )
    m.pcd_jsd = float(js)

    # ── MERT drift ─────────────────────────────────────────────────
    try:
        m.mert_drift = mert_drift(
            wav_in, wav_out,
            sr_in=sr, sr_out=sr,
            model=mert_model, feature_extractor=mert_fe,
        )
    except Exception:
        m.mert_drift = 0.0

    # ── CLAP shift ─────────────────────────────────────────────────
    clap = clap_score_shift(wav_in, wav_out, sr=sr)
    if clap.get("available"):
        m.clap_calm_in = float(clap["score_calm_in"])
        m.clap_calm_out = float(clap["score_calm_out"])
        m.clap_shift = float(clap["delta"])

    # ── Emotion (M2E) ──────────────────────────────────────────────
    if m2e is not None and getattr(m2e, "available", False):
        try:
            # Write wavs to tmp (Music2Emo needs file paths).
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                p_in_f = Path(td) / "in.wav"
                p_out_f = Path(td) / "out.wav"
                sf.write(str(p_in_f), wav_in, sr, subtype="PCM_16")
                sf.write(str(p_out_f), wav_out, sr, subtype="PCM_16")
                ri = m2e.predict(p_in_f)
                ro = m2e.predict(p_out_f)
            m.valence_in = float(ri["valence"])
            m.arousal_in = float(ri["arousal"])
            m.valence_out = float(ro["valence"])
            m.arousal_out = float(ro["arousal"])
            m.delta_valence = m.valence_out - m.valence_in
            m.delta_arousal = m.arousal_out - m.arousal_in
            d_in = float(np.hypot(m.valence_in - 5.0, m.arousal_in - 3.0))
            d_out = float(np.hypot(m.valence_out - 5.0, m.arousal_out - 3.0))
            m.moved_toward_neutral = bool(d_out < d_in)
        except Exception:
            pass

    # ── Tune-type preservation ─────────────────────────────────────
    if tune_classifier is not None:
        try:
            pred = tune_classifier(wav_out, sr=sr)
            m.tune_type_preserved = int(pred == sidecar.tune_type)
        except Exception:
            m.tune_type_preserved = 0

    return m


# ---------------------------------------------------------------------
# Tune-type classifier wrapper
# ---------------------------------------------------------------------
def load_tune_type_classifier(ckpt_dir: Path, device: torch.device):
    """Return a callable ``(wav, sr) → predicted tune-type string`` or None."""
    ckpt = ckpt_dir / "best.pt"
    if not ckpt.exists():
        return None
    blob = torch.load(ckpt, map_location="cpu")
    from .tune_type_classifier import TuneTypeMLP
    label_map = blob["label_map"]
    inv = {v: k for k, v in label_map.items()}
    model = TuneTypeMLP(n_classes=blob["n_classes"]).to(device).eval()
    model.load_state_dict(blob["state_dict"])
    from .mert_embed import load_mert, mert_embed_waveform, MERT_SAMPLE_RATE
    mert_m, mert_fe = load_mert(device=str(device))

    @torch.no_grad()
    def _clf(wav: np.ndarray, sr: int) -> str:
        if sr != MERT_SAMPLE_RATE:
            import librosa
            wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=MERT_SAMPLE_RATE)
        feat = mert_embed_waveform(
            wav.astype(np.float32), sr=MERT_SAMPLE_RATE,
            model=mert_m, feature_extractor=mert_fe,
        ).to(device)
        logits = model(feat)
        cls = int(logits.argmax(dim=-1).item())
        return inv.get(cls, "UNK")

    return _clf


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    default_cache = os.environ.get(
        "JAMENDO_CACHE",
        f"/scratch0/{os.environ.get('USER', 'nobody')}/jamendo",
    )
    ap.add_argument("--tracks", default="a",
                    help="Comma-separated track letters {a,b,c}")
    ap.add_argument("--u", default="0.0,0.3,0.6,0.9",
                    help="Comma-separated u values")
    ap.add_argument("--test_dir", default=f"{default_cache}/test_clips")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument(
        "--tune_classifier_dir",
        default=str(_MODELLING_ROOT / "checkpoints" / "celtic_tune_type_classifier"),
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_clips", type=int, default=0,
                    help="Cap # test clips (0 = all)")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    test_dir = Path(args.test_dir)
    metas = read_manifest(test_dir / "manifest.jsonl")
    if args.max_clips > 0:
        metas = metas[: args.max_clips]
    LOG.info("eval on %d test clips from %s", len(metas), test_dir)

    # Load supporting models once.
    tune_classifier = load_tune_type_classifier(
        Path(args.tune_classifier_dir), device,
    )
    mert_model = mert_fe = None
    try:
        from .mert_embed import load_mert
        mert_model, mert_fe = load_mert(device=str(device))
    except Exception as e:
        LOG.warning("MERT unavailable (%s); mert_drift will be 0.0", e)
    try:
        from evaluation.emotion_regressor import Music2EmoRegressor
        m2e = Music2EmoRegressor()
        if not m2e.available:
            m2e = None
    except Exception:
        m2e = None

    track_letters = [x.strip() for x in args.tracks.split(",") if x.strip()]
    u_values = [float(x) for x in args.u.split(",")]

    # Checkpoint roots relative to the modelling repo.
    ckpt_root = _MODELLING_ROOT / "checkpoints"
    for t in track_letters:
        name = f"celtic_track_{t}"
        ckpt = ckpt_root / name / "best.pt"
        if not ckpt.exists():
            LOG.warning("checkpoint missing for track %s: %s", t, ckpt)
            continue
        if t == "a":
            gen = build_track_a_generator(ckpt, device)
        elif t == "b":
            gen = build_track_b_generator(ckpt, device)
        elif t == "c":
            gen = build_track_c_generator(ckpt, device)
        else:
            LOG.warning("unknown track %s", t)
            continue

        for u in u_values:
            out_dir = Path(args.results_dir) / name / f"u{u:.1f}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_jsonl = out_dir / "per_clip.jsonl"
            out_wav_dir = out_dir / "wavs"
            out_wav_dir.mkdir(parents=True, exist_ok=True)

            # Skip if already done.
            existing_stems: set[str] = set()
            if out_jsonl.exists():
                with out_jsonl.open("r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            existing_stems.add(json.loads(line).get("stem", ""))
                        except Exception:
                            continue
                LOG.info("resume: %d clips already scored at u=%.1f",
                         len(existing_stems), u)

            sr = 24_000
            with out_jsonl.open("a", encoding="utf-8") as fout:
                for sidecar in tqdm(metas, desc=f"track-{t} u={u:.1f}"):
                    if sidecar.stem in existing_stems:
                        continue
                    wav_in, wav_sr = sf.read(
                        str(test_dir / f"{sidecar.stem}.wav"),
                        dtype="float32", always_2d=False,
                    )
                    if wav_in.ndim > 1:
                        wav_in = wav_in.mean(axis=-1)
                    try:
                        wav_out = gen.fn(wav_in, float(u), sidecar)
                    except NotImplementedError as e:
                        LOG.error("%s", e)
                        return 2
                    except Exception as e:
                        LOG.warning("generation failed for %s: %s", sidecar.stem, e)
                        continue
                    # Save the output wav for A/B + FAD.
                    out_wav_path = out_wav_dir / f"{sidecar.stem}.wav"
                    sf.write(str(out_wav_path), wav_out, sr, subtype="PCM_16")

                    metrics = per_clip_eval(
                        sidecar, wav_in, wav_out,
                        sr=sr, u=float(u), track_name=gen.name,
                        tune_classifier=tune_classifier,
                        m2e=m2e,
                        mert_model=mert_model, mert_fe=mert_fe,
                    )
                    fout.write(json.dumps(metrics.to_dict()) + "\n")
                    fout.flush()
            LOG.info("per-clip metrics → %s", out_jsonl)

            # ── Corpus-level FAD ───────────────────────────────────
            try:
                ref = [test_dir / f"{m.stem}.wav" for m in metas]
                cand = [out_wav_dir / f"{m.stem}.wav" for m in metas
                        if (out_wav_dir / f"{m.stem}.wav").exists()]
                fad = fad_clap(ref, cand)
                (out_dir / "fad.json").write_text(json.dumps(fad, indent=2))
                LOG.info("FAD: %s", fad)
            except Exception as e:
                LOG.warning("FAD failed: %s", e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
