"""
Train a MERT-based raga-identity classifier on the Saraga Kalyan-thaat subset.

Pipeline:
  1. Walk ``data/saraga_kalyan_thaat`` for ``*.wav`` files.
  2. Slice each recording into non-overlapping 10-second clips.
  3. Label each clip with its parent dir's canonical raga name
     (same normalisation as ``train.extract_raga``).
  4. Split at the **recording level** (not clip level) so the val set
     contains recordings unseen in training.
  5. Extract MERT-v1-95M mean-pooled features per clip (cached to .npy).
  6. Train a 768 → 256 → |R| MLP with cross-entropy.
  7. Save to ``checkpoints/raga_classifier/model.pt`` with the raga vocab.

Because the Saraga Kalyan-thaat subset has only 15 source recordings
over 12 ragas, single-recording ragas contribute only to the training
side (we can't measure generalisation for those). Reported val accuracy
will therefore reflect the three multi-recording ragas (Bihag, Hameer,
Yaman). That's still useful: for *evaluation* of the codec editor we
ask whether the editor preserves the raga prediction between input and
output — a measurement that does not require generalisation to unseen
ragas, only discrimination within the training distribution.

Usage:
    python train_raga_classifier.py \\
        --saraga_dir data/saraga_kalyan_thaat \\
        --output checkpoints/raga_classifier \\
        --epochs 40 --clip_seconds 10.0
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from evaluation.mert_features import MERTFeatureExtractor

_RAGA_PREFIX_RE = re.compile(r"^(?:Raag|Raga)\s*", re.IGNORECASE)


# ──────────────────── Label extraction ────────────────────


def canonicalise_raga(dir_name: str) -> str:
    """Turn a Saraga parent-dir like 'Raag Hameer' into 'Hameer'."""
    name = _RAGA_PREFIX_RE.sub("", dir_name).strip()
    return re.sub(r"\s+", " ", name) or "UNK"


# ──────────────────── Clip indexing ────────────────────


@dataclass(frozen=True)
class Clip:
    wav_path: str
    recording_id: str     # full source album path
    raga: str
    start_sec: float
    end_sec: float


def index_clips(saraga_dir: Path, clip_seconds: float,
                skip_stems: List[str] | None = None) -> List[Clip]:
    """Enumerate non-overlapping clips across all WAVs under ``saraga_dir``."""
    clips: List[Clip] = []
    skip_stems = skip_stems or []
    for wav_path in sorted(saraga_dir.rglob("*.wav")):
        if any(wav_path.stem == s for s in skip_stems):
            continue
        parts = wav_path.parts
        # .../saraga_kalyan_thaat/<album>/<raga_dir>/<raga_dir>.wav
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


def recording_level_split(clips: List[Clip], val_frac: float,
                          rng: np.random.Generator) -> Tuple[List[Clip], List[Clip]]:
    """Hold out one full recording per multi-recording raga for val.

    For ragas with only one recording, all clips go into train (val coverage
    of that raga is impossible without a second recording).
    """
    per_raga: Dict[str, List[str]] = {}
    for c in clips:
        per_raga.setdefault(c.raga, []).append(c.recording_id)

    val_recs: set[str] = set()
    for raga, recs in per_raga.items():
        uniq = sorted(set(recs))
        if len(uniq) >= 2:
            # Choose one recording for val deterministically.
            val_recs.add(uniq[int(rng.integers(0, len(uniq)))])

    train, val = [], []
    for c in clips:
        (val if c.recording_id in val_recs else train).append(c)
    return train, val


# ──────────────────── Feature caching ────────────────────


def extract_features(clips: List[Clip], cache_path: Path,
                     mert: MERTFeatureExtractor) -> np.ndarray:
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True).item()
        if len(cached["features"]) == len(clips):
            return cached["features"]
    feats = np.zeros((len(clips), MERTFeatureExtractor.embed_dim),
                     dtype=np.float32)
    current_path = None
    y = None
    sr = None
    for i, c in enumerate(tqdm(clips, desc="MERT features")):
        if c.wav_path != current_path:
            y, sr = sf.read(c.wav_path, always_2d=False)
            if y.ndim > 1:
                y = y.mean(axis=-1)
            y = y.astype(np.float32)
            current_path = c.wav_path
        i0 = int(c.start_sec * sr)
        i1 = int(c.end_sec * sr)
        clip = y[i0:i1]
        feats[i] = mert.embed_waveform(clip, sr=sr)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, {"features": feats}, allow_pickle=True)
    return feats


# ──────────────────── Model ────────────────────


class RagaClassifier(nn.Module):
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


class ClipDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i], self.labels[i]


# ──────────────────── Training ────────────────────


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
    train_clips, val_clips = recording_level_split(clips, args.val_frac, rng)
    print(f"Split: train={len(train_clips)} val={len(val_clips)} "
          f"({len({c.recording_id for c in val_clips})} val recordings)")

    # Build deterministic raga vocab — 0 is reserved for UNK (matches train.py).
    ragas = sorted({c.raga for c in clips})
    ragas.remove("UNK") if "UNK" in ragas else None
    vocab: Dict[str, int] = {"UNK": 0}
    for i, r in enumerate(ragas, start=1):
        vocab[r] = i
    n_classes = len(vocab)
    print(f"Ragas ({n_classes}): {vocab}")

    mert = MERTFeatureExtractor()

    # Cache features separately for train and val to avoid mixing.
    print("\n== Train features ==")
    X_train = extract_features(train_clips, out_dir / "feat_train.npy", mert)
    y_train = np.array([vocab[c.raga] for c in train_clips], dtype=np.int64)

    print("\n== Val features ==")
    X_val = extract_features(val_clips, out_dir / "feat_val.npy", mert) \
        if val_clips else np.zeros((0, MERTFeatureExtractor.embed_dim), np.float32)
    y_val = np.array([vocab[c.raga] for c in val_clips], dtype=np.int64)

    train_ds = ClipDataset(X_train, y_train)
    val_ds = ClipDataset(X_val, y_val) if len(X_val) else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False) if val_ds else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RagaClassifier(n_classes=n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    best_acc = -1.0
    for ep in range(args.epochs):
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
        tr_loss /= tr_total
        tr_acc = tr_correct / tr_total

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

        # Save best (train_acc if no val, else val_acc).
        key = val_acc if val_loader is not None else tr_acc
        if key > best_acc:
            best_acc = key
            torch.save({
                "state_dict": model.state_dict(),
                "vocab": vocab,
                "in_dim": 768, "hidden": 256,
                "clip_seconds": args.clip_seconds,
                "mert_name": MERTFeatureExtractor.hub_name,
                "epoch": ep, "val_acc": val_acc, "tr_acc": tr_acc,
            }, out_dir / "model.pt")

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nbest = {best_acc:.3f} → {out_dir/'model.pt'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--saraga_dir", default="data/saraga_kalyan_thaat")
    p.add_argument("--output",     default="checkpoints/raga_classifier")
    p.add_argument("--clip_seconds", type=float, default=10.0)
    p.add_argument("--epochs",       type=int,   default=40)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--val_frac",     type=float, default=0.2)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--skip_stems", nargs="*", default=[],
                   help="Optional WAV stems to exclude entirely.")
    args = p.parse_args()
    train(args)
