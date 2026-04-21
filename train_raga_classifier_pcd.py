"""
PCD-based raga-identity classifier (recording-invariant features).

Rationale
---------
The MERT-based classifier (``train_raga_classifier.py``) hits ~100% train
accuracy but only ~27% val accuracy on held-out recordings — it memorises
production / timbre fingerprints, not raga grammar.

PCDs (tonic-normalised, octave-folded, salience-weighted pitch-class
distributions) are recording-invariant by construction. They are the
standard feature used by the raga-recognition literature since
Koduri et al., JNMR 2012.

Feature pipeline per 10 s clip
-----------------------------
    wav → PESTO (f0, confidence) → tonic-normalise (cents) → octave-fold
        → cache raw (cents, weights) per clip
    training: re-histogram on the fly with random augmentation
    val: fixed histogram, no augmentation

Augmentation (training only)
---------------------------
  * Global tonic-estimate jitter         N(0, 20 cents)
  * Per-frame pitch-estimate noise       N(0, 5  cents)
  * Random frame dropout                 Bernoulli(keep=0.7)
  * Random contiguous sub-window         uniform in [0.7, 1.0]
  * Gaussian-smoothing sigma jitter      base ± 1 bin

Usage
-----
    python train_raga_classifier_pcd.py \\
        --saraga_dir data/saraga_kalyan_thaat \\
        --output     checkpoints/raga_classifier_pcd \\
        --clip_seconds 10.0 --epochs 40
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Ensure local imports resolve regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation.tonic import resolve_tonic  # noqa: E402


_RAGA_PREFIX_RE = re.compile(r"^(?:Raag|Raga)\s*", re.IGNORECASE)


# ──────────────────── Label extraction ────────────────────


def canonicalise_raga(dir_name: str) -> str:
    """Turn a Saraga parent-dir like 'Raag Hameer' into 'Hameer'."""
    name = _RAGA_PREFIX_RE.sub("", dir_name).strip()
    return re.sub(r"\s+", " ", name) or "UNK"


# ──────────────────── Clip indexing (duplicated from train_raga_classifier.py) ─


@dataclass(frozen=True)
class Clip:
    wav_path: str
    recording_id: str     # '<album>/<raga_dir>'
    raga: str
    start_sec: float
    end_sec: float


def index_clips(saraga_dir: Path, clip_seconds: float,
                skip_stems: List[str] | None = None) -> List[Clip]:
    clips: List[Clip] = []
    skip_stems = skip_stems or []
    for wav_path in sorted(saraga_dir.rglob("*.wav")):
        if any(wav_path.stem == s for s in skip_stems):
            continue
        parts = wav_path.parts
        if len(parts) < 3:
            continue
        raga_dir = parts[-2]
        album = parts[-3]
        recording_id = f"{album}/{raga_dir}"
        raga = canonicalise_raga(raga_dir)
        info = sf.info(str(wav_path))
        duration = info.frames / info.samplerate
        n_clips = int(duration // clip_seconds)
        for i in range(n_clips):
            clips.append(Clip(
                wav_path=str(wav_path),
                recording_id=recording_id,
                raga=raga,
                start_sec=i * clip_seconds,
                end_sec=(i + 1) * clip_seconds,
            ))
    return clips


def recording_level_split(clips: List[Clip],
                          rng: np.random.Generator) -> Tuple[List[Clip], List[Clip]]:
    """Hold out one full recording per multi-recording raga for val."""
    per_raga: Dict[str, List[str]] = {}
    for c in clips:
        per_raga.setdefault(c.raga, []).append(c.recording_id)
    val_recs: set[str] = set()
    for raga, recs in per_raga.items():
        uniq = sorted(set(recs))
        if len(uniq) >= 2:
            val_recs.add(uniq[int(rng.integers(0, len(uniq)))])
    train, val = [], []
    for c in clips:
        (val if c.recording_id in val_recs else train).append(c)
    return train, val


# ──────────────────── Tonic resolution per recording ─────────


def resolve_tonics(clips: List[Clip], saraga_dir: Path) -> Dict[str, Tuple[float, str]]:
    """One tonic per recording_id, via Saraga .ctonic.txt → Essentia → default."""
    tonics: Dict[str, Tuple[float, str]] = {}
    by_rec: Dict[str, str] = {}
    for c in clips:
        by_rec.setdefault(c.recording_id, c.wav_path)
    for rec_id, wav_path in tqdm(by_rec.items(), desc="Tonics"):
        stem = Path(wav_path).stem
        hz, src = resolve_tonic(wav_path, stem=stem, tonic_dir=str(saraga_dir))
        tonics[rec_id] = (hz, src)
    return tonics


# ──────────────────── PESTO pitch extraction (cached) ─────────


def extract_raw_frames(
    clips: List[Clip],
    tonics: Dict[str, Tuple[float, str]],
    cache_path: Path,
    device: str = "cuda",
    hop_ms: float = 10.0,
    conf_threshold: float = 0.5,
) -> List[Dict[str, np.ndarray]]:
    """For each clip, run PESTO once and store (cents, weights) tonic-normalised.

    Cents are already octave-folded (mod 1200).
    """
    if cache_path.exists():
        print(f"  using cached {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    import pesto

    raw: List[Dict[str, np.ndarray]] = []
    current_path = None
    y = None
    sr_audio = None
    for c in tqdm(clips, desc=f"PESTO ({cache_path.name})"):
        if c.wav_path != current_path:
            y, sr_audio = sf.read(c.wav_path, always_2d=False)
            if y.ndim > 1:
                y = y.mean(axis=-1)
            y = y.astype(np.float32)
            current_path = c.wav_path
        i0 = int(c.start_sec * sr_audio)
        i1 = int(c.end_sec * sr_audio)
        clip_audio = y[i0:i1]
        if len(clip_audio) < sr_audio * 0.5:
            raw.append({"cents": np.array([], dtype=np.float32),
                        "weights": np.array([], dtype=np.float32)})
            continue

        a = torch.from_numpy(clip_audio).to(device)
        try:
            _, pitch, conf, _ = pesto.predict(
                a, sr=sr_audio, step_size=hop_ms,
            )
        except Exception as e:
            print(f"  PESTO failed on {c.wav_path}@{c.start_sec:.1f}s: {e}")
            raw.append({"cents": np.array([], dtype=np.float32),
                        "weights": np.array([], dtype=np.float32)})
            continue
        pitch_np = pitch.squeeze().detach().cpu().numpy().astype(np.float32)
        conf_np = conf.squeeze().detach().cpu().numpy().astype(np.float32)

        mask = (pitch_np > 0) & (conf_np >= conf_threshold)
        if not mask.any():
            raw.append({"cents": np.array([], dtype=np.float32),
                        "weights": np.array([], dtype=np.float32)})
            continue
        p = pitch_np[mask]
        w = conf_np[mask]
        tonic_hz = tonics[c.recording_id][0]
        cents = 1200.0 * np.log2(p / max(tonic_hz, 1e-8) + 1e-12)
        cents = np.mod(cents, 1200.0).astype(np.float32)
        raw.append({"cents": cents, "weights": w.astype(np.float32)})

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(raw, f)
    return raw


# ──────────────────── PCD histogram (with optional augmentation) ─


def make_pcd(
    cents: np.ndarray,
    weights: np.ndarray,
    n_bins: int = 120,
    sigma_bins: float = 3.0,
    tonic_cent_jitter: float = 0.0,
    frame_cent_noise_std: float = 0.0,
    frame_keep_prob: float = 1.0,
    subwindow_frac: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Tonic-normalised salience-weighted PCD. Augmentations applied iff rng given."""
    if len(cents) == 0:
        return np.ones(n_bins, dtype=np.float32) / n_bins

    c = cents.astype(np.float32, copy=False)
    w = weights.astype(np.float32, copy=False)

    if rng is not None:
        if subwindow_frac < 1.0 and len(c) > 10:
            frac = rng.uniform(subwindow_frac, 1.0)
            keep = max(1, int(len(c) * frac))
            start = rng.integers(0, len(c) - keep + 1)
            c = c[start:start + keep]
            w = w[start:start + keep]
        if frame_keep_prob < 1.0:
            mask = rng.random(len(c)) < frame_keep_prob
            if mask.any():
                c, w = c[mask], w[mask]
        if frame_cent_noise_std > 0.0:
            c = c + rng.normal(0.0, frame_cent_noise_std, size=c.shape).astype(np.float32)
        if tonic_cent_jitter > 0.0:
            c = c + rng.normal(0.0, tonic_cent_jitter)

    c = np.mod(c, 1200.0)
    hist, _ = np.histogram(c, bins=n_bins, range=(0.0, 1200.0), weights=w)
    if rng is not None and sigma_bins > 0:
        sigma_bins = max(0.5, sigma_bins + rng.uniform(-1.0, 1.0))
    hist = gaussian_filter1d(hist, sigma=sigma_bins, mode="wrap")
    total = hist.sum()
    if total <= 0:
        return np.ones(n_bins, dtype=np.float32) / n_bins
    return (hist / total).astype(np.float32)


# ──────────────────── Dataset / Model ─────────


class PCDTrainDataset(Dataset):
    """Re-histograms on the fly with augmentation."""

    def __init__(self, raw: List[Dict[str, np.ndarray]], labels: np.ndarray,
                 n_bins: int = 120, sigma_bins: float = 3.0, seed: int = 0,
                 augment: bool = True):
        self.raw = raw
        self.labels = labels
        self.n_bins = n_bins
        self.sigma_bins = sigma_bins
        self.augment = augment
        self._base_seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, i):
        if self.augment:
            rng = np.random.default_rng(self._base_seed + self._epoch * 104729 + i)
            pcd = make_pcd(
                self.raw[i]["cents"], self.raw[i]["weights"],
                n_bins=self.n_bins, sigma_bins=self.sigma_bins,
                tonic_cent_jitter=20.0,
                frame_cent_noise_std=5.0,
                frame_keep_prob=0.7,
                subwindow_frac=0.7,
                rng=rng,
            )
        else:
            pcd = make_pcd(self.raw[i]["cents"], self.raw[i]["weights"],
                           n_bins=self.n_bins, sigma_bins=self.sigma_bins)
        return torch.from_numpy(pcd), int(self.labels[i])


class RagaPCDClassifier(nn.Module):
    def __init__(self, n_classes: int, n_bins: int = 120, hidden: int = 128,
                 dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_bins, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ──────────────────── Training ─────────


def train(args: argparse.Namespace) -> None:
    saraga_dir = Path(args.saraga_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    clips = index_clips(saraga_dir, args.clip_seconds,
                        skip_stems=args.skip_stems)
    if not clips:
        raise RuntimeError(f"No WAVs found under {saraga_dir}")
    print(f"Indexed {len(clips)} clips from "
          f"{len({c.recording_id for c in clips})} recordings, "
          f"{len({c.raga for c in clips})} ragas.")

    rng = np.random.default_rng(args.seed)
    train_clips, val_clips = recording_level_split(clips, rng)
    print(f"Split: train={len(train_clips)} val={len(val_clips)} "
          f"({len({c.recording_id for c in val_clips})} val recordings)")

    # Tonics (cheap — one .ctonic.txt lookup per recording).
    tonics = resolve_tonics(clips, saraga_dir)
    src_counts: Dict[str, int] = {}
    for _, src in tonics.values():
        src_counts[src] = src_counts.get(src, 0) + 1
    print(f"Tonic sources: {src_counts}")

    # Raga vocab (0 = UNK, matches train.py).
    ragas = sorted({c.raga for c in clips})
    if "UNK" in ragas:
        ragas.remove("UNK")
    vocab: Dict[str, int] = {"UNK": 0}
    for i, r in enumerate(ragas, start=1):
        vocab[r] = i
    n_classes = len(vocab)
    print(f"Ragas ({n_classes}): {vocab}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Per-clip raw (cents, weights) via PESTO, cached.
    print("\n== Train raw PESTO frames ==")
    raw_train = extract_raw_frames(
        train_clips, tonics, out_dir / "pcd_raw_train.pkl",
        device=device, hop_ms=args.hop_ms,
    )
    y_train = np.array([vocab[c.raga] for c in train_clips], dtype=np.int64)

    print("\n== Val raw PESTO frames ==")
    raw_val = extract_raw_frames(
        val_clips, tonics, out_dir / "pcd_raw_val.pkl",
        device=device, hop_ms=args.hop_ms,
    ) if val_clips else []
    y_val = np.array([vocab[c.raga] for c in val_clips], dtype=np.int64)

    # Sanity: voiced-frame counts per clip.
    train_frame_counts = np.array([len(r["cents"]) for r in raw_train])
    val_frame_counts = np.array([len(r["cents"]) for r in raw_val])
    print(f"Voiced frames / clip  train: mean={train_frame_counts.mean():.0f} "
          f"med={np.median(train_frame_counts):.0f} "
          f"empty={(train_frame_counts == 0).sum()}/{len(train_frame_counts)}")
    if len(val_frame_counts):
        print(f"Voiced frames / clip  val  : mean={val_frame_counts.mean():.0f} "
              f"med={np.median(val_frame_counts):.0f} "
              f"empty={(val_frame_counts == 0).sum()}/{len(val_frame_counts)}")

    train_ds = PCDTrainDataset(raw_train, y_train, n_bins=args.n_bins,
                               sigma_bins=args.sigma_bins,
                               seed=args.seed, augment=True)
    val_ds = PCDTrainDataset(raw_val, y_val, n_bins=args.n_bins,
                             sigma_bins=args.sigma_bins,
                             seed=args.seed + 1, augment=False) if raw_val else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0) if val_ds else None

    model = RagaPCDClassifier(n_classes=n_classes, n_bins=args.n_bins,
                              hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    best_key = -1.0
    best_state = None

    for ep in range(args.epochs):
        train_ds.set_epoch(ep)

        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(x)
            tr_correct += (logits.argmax(-1) == y).sum().item()
            tr_total += len(x)
        tr_loss /= max(tr_total, 1)
        tr_acc = tr_correct / max(tr_total, 1)

        val_loss = val_acc = float("nan")
        if val_loader is not None:
            model.eval()
            vl, vc, vt = 0.0, 0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    l = loss_fn(logits, y)
                    vl += l.item() * len(x)
                    vc += (logits.argmax(-1) == y).sum().item()
                    vt += len(x)
            val_loss = vl / max(vt, 1)
            val_acc = vc / max(vt, 1)

        history.append({"epoch": ep, "tr_loss": tr_loss, "tr_acc": tr_acc,
                        "val_loss": val_loss, "val_acc": val_acc})
        print(f"ep {ep:3d}  tr_loss={tr_loss:.3f}  tr_acc={tr_acc:.3f}  "
              f"val_loss={val_loss:.3f}  val_acc={val_acc:.3f}")

        key = val_acc if val_loader is not None else tr_acc
        if key > best_key:
            best_key = key
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            torch.save({
                "state_dict": best_state,
                "vocab": vocab,
                "feature_type": "pcd",
                "n_bins": args.n_bins,
                "hidden": args.hidden,
                "sigma_bins": args.sigma_bins,
                "clip_seconds": args.clip_seconds,
                "hop_ms": args.hop_ms,
                "epoch": ep, "val_acc": val_acc, "tr_acc": tr_acc,
            }, out_dir / "model.pt")

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nbest = {best_key:.3f} → {out_dir/'model.pt'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--saraga_dir",   default="data/saraga_kalyan_thaat")
    p.add_argument("--output",       default="checkpoints/raga_classifier_pcd")
    p.add_argument("--clip_seconds", type=float, default=10.0)
    p.add_argument("--hop_ms",       type=float, default=10.0)
    p.add_argument("--n_bins",       type=int,   default=120)
    p.add_argument("--sigma_bins",   type=float, default=3.0)
    p.add_argument("--hidden",       type=int,   default=128)
    p.add_argument("--dropout",      type=float, default=0.4)
    p.add_argument("--epochs",       type=int,   default=40)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--skip_stems",   nargs="*", default=[])
    args = p.parse_args()
    train(args)
