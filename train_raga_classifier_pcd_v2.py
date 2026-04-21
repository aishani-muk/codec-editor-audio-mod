"""
PCD+ raga classifier (v2) — adds feature- and training-side upgrades on
top of ``train_raga_classifier_pcd.py``.

Feature upgrades (all from ``evaluation.pcd_features.extract_pcd_plus``)
----------------------------------------------------------------------
  * Longer clips with overlap:  30 s clips, 10 s hop
    (vs v1: 10 s non-overlapping). Captures enough audio for a pakad.
  * Multi-resolution PCD:       3 sigmas (5, 15, 30 cents) ⇒ 360 dims.
  * Pitch-dyad distribution:    24 × 24 ⇒ 576 dims, captures melodic
                                direction the 1-D PCD is blind to.
  * Aaroha/avroha split PCDs:   slope-gated ⇒ 2 × 120 = 240 dims.
  Total feature dim: 1176.

Training upgrades
-----------------
  * Mixup in PCD+ space (alpha=0.2, applied on 50% of batches).
  * Label smoothing 0.1 in CrossEntropyLoss.
  * Optional SimCLR pre-training (``--pretrain_epochs > 0``):
    two-view NT-Xent on augmented feature pairs warms up the classifier's
    first layer; the projection head is discarded before supervised
    fine-tuning.

Usage
-----
    python train_raga_classifier_pcd_v2.py \\
        --saraga_dir data/saraga_kalyan_thaat \\
        --output     checkpoints/raga_classifier_pcd_v2 \\
        --clip_seconds 30.0 --hop_seconds 10.0 \\
        --pretrain_epochs 20 --epochs 50
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation.tonic import resolve_tonic  # noqa: E402
from evaluation.pcd_features import extract_pcd_plus, PCD_PLUS_DIM  # noqa: E402

# Reuse indexing helpers from v1. Import is cheap (no MERT model load there).
from train_raga_classifier_pcd import (  # noqa: E402
    canonicalise_raga,
    recording_level_split,
)


# ──────────────────── Clip indexing (with overlap) ─────────


@dataclass(frozen=True)
class Clip:
    wav_path: str
    recording_id: str
    raga: str
    start_sec: float
    end_sec: float


def index_clips_overlap(saraga_dir: Path, clip_seconds: float,
                        hop_seconds: float,
                        skip_stems: List[str] | None = None) -> List[Clip]:
    """Sliding window indexing. Drops trailing partial window."""
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
        if duration < clip_seconds:
            continue
        # Number of windows: floor((duration - clip) / hop) + 1
        n = int((duration - clip_seconds) // hop_seconds) + 1
        for i in range(n):
            start = i * hop_seconds
            clips.append(Clip(
                wav_path=str(wav_path),
                recording_id=recording_id,
                raga=raga,
                start_sec=start,
                end_sec=start + clip_seconds,
            ))
    return clips


# ──────────────────── Tonic cache ─────────


def resolve_tonics(clips: List[Clip], saraga_dir: Path
                   ) -> Dict[str, Tuple[float, str]]:
    tonics: Dict[str, Tuple[float, str]] = {}
    by_rec: Dict[str, str] = {}
    for c in clips:
        by_rec.setdefault(c.recording_id, c.wav_path)
    for rec_id, wav_path in tqdm(by_rec.items(), desc="Tonics"):
        stem = Path(wav_path).stem
        hz, src = resolve_tonic(wav_path, stem=stem, tonic_dir=str(saraga_dir))
        tonics[rec_id] = (hz, src)
    return tonics


# ──────────────────── PESTO pitch cache (raw frames) ─────────


def extract_raw_frames(
    clips: List[Clip],
    tonics: Dict[str, Tuple[float, str]],
    cache_path: Path,
    device: str = "cuda",
    hop_ms: float = 10.0,
    conf_threshold: float = 0.5,
) -> List[Dict[str, np.ndarray]]:
    """Per clip, store the FULL (pitch_hz, conf) time series + tonic.

    Unlike v1 (which stored octave-folded cents), we keep the raw pitch
    stream so that time-sensitive features (PDD, aaroha/avroha) can be
    computed at training time with fresh augmentations.
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
        tonic_hz = tonics[c.recording_id][0]
        if len(clip_audio) < sr_audio * 0.5:
            raw.append({
                "pitch_hz": np.zeros(0, dtype=np.float32),
                "conf": np.zeros(0, dtype=np.float32),
                "tonic_hz": float(tonic_hz),
            })
            continue
        a = torch.from_numpy(clip_audio).to(device)
        try:
            _, pitch, conf, _ = pesto.predict(a, sr=sr_audio, step_size=hop_ms)
        except Exception as e:
            print(f"  PESTO failed on {c.wav_path}@{c.start_sec:.1f}s: {e}")
            raw.append({
                "pitch_hz": np.zeros(0, dtype=np.float32),
                "conf": np.zeros(0, dtype=np.float32),
                "tonic_hz": float(tonic_hz),
            })
            continue
        pitch_np = pitch.squeeze().detach().cpu().numpy().astype(np.float32)
        conf_np = conf.squeeze().detach().cpu().numpy().astype(np.float32)
        # Zero out sub-threshold frames' pitch so downstream masking is
        # uniform; keep conf for salience weighting.
        pitch_np = np.where(conf_np >= conf_threshold, pitch_np, 0.0)
        raw.append({
            "pitch_hz": pitch_np,
            "conf": conf_np,
            "tonic_hz": float(tonic_hz),
        })

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(raw, f)
    return raw


# ──────────────────── Datasets ─────────


# Augmentation hyper-parameters shared between supervised + SimCLR loops.
TRAIN_AUG = dict(
    tonic_jitter_cents=20.0,
    frame_noise_cents=5.0,
    frame_keep_prob=0.8,
    subwindow_frac=0.7,
)


class PCDPlusDataset(Dataset):
    """Computes PCD+ features on the fly. With ``augment=True`` the aug
    parameters above are applied every __getitem__."""

    def __init__(self, raw: List[Dict[str, np.ndarray]], labels: np.ndarray,
                 seed: int = 0, augment: bool = True):
        self.raw = raw
        self.labels = labels
        self.augment = augment
        self._base_seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self):
        return len(self.raw)

    def _get_feat(self, i: int, rng: np.random.Generator | None) -> np.ndarray:
        r = self.raw[i]
        return extract_pcd_plus(
            r["pitch_hz"], r["conf"], r["tonic_hz"],
            aug_params=TRAIN_AUG if rng is not None else None,
            rng=rng,
        )

    def __getitem__(self, i):
        if self.augment:
            rng = np.random.default_rng(
                self._base_seed + self._epoch * 104729 + i)
            feat = self._get_feat(i, rng)
        else:
            feat = self._get_feat(i, None)
        return torch.from_numpy(feat), int(self.labels[i])


class PCDPlusTwoViewDataset(Dataset):
    """Two-augmented-views dataset for SimCLR. Returns (v1, v2) per clip."""

    def __init__(self, raw: List[Dict[str, np.ndarray]], seed: int = 0):
        self.raw = raw
        self._base_seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, i):
        r = self.raw[i]
        base = self._base_seed + self._epoch * 104729 + i * 2
        rng1 = np.random.default_rng(base)
        rng2 = np.random.default_rng(base + 1)
        v1 = extract_pcd_plus(r["pitch_hz"], r["conf"], r["tonic_hz"],
                              aug_params=TRAIN_AUG, rng=rng1)
        v2 = extract_pcd_plus(r["pitch_hz"], r["conf"], r["tonic_hz"],
                              aug_params=TRAIN_AUG, rng=rng2)
        return torch.from_numpy(v1), torch.from_numpy(v2)


# ──────────────────── Model ─────────


class PCDPlusEncoder(nn.Module):
    """First two layers: 1176 → hidden → hidden. Output is a representation
    (used as classifier backbone AND SimCLR encoder)."""

    def __init__(self, in_dim: int = PCD_PLUS_DIM, hidden: int = 256,
                 dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class PCDPlusClassifier(nn.Module):
    def __init__(self, n_classes: int, in_dim: int = PCD_PLUS_DIM,
                 hidden: int = 256, dropout: float = 0.4):
        super().__init__()
        self.encoder = PCDPlusEncoder(in_dim=in_dim, hidden=hidden,
                                      dropout=dropout)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x):
        h = self.encoder(x)
        return self.head(h)


class SimCLRProjection(nn.Module):
    """Small 2-layer projection head, discarded after pre-training."""

    def __init__(self, in_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ──────────────────── Training primitives ─────────


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor,
                 temperature: float = 0.2) -> torch.Tensor:
    """SimCLR NT-Xent over a batch of paired views."""
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)                # (2B, D)
    z = F.normalize(z, dim=1)
    sim = z @ z.T / temperature                   # (2B, 2B)
    sim.fill_diagonal_(float("-inf"))
    # Positive for row i is row (i + B) mod 2B.
    targets = torch.arange(2 * B, device=z.device)
    targets = (targets + B) % (2 * B)
    return F.cross_entropy(sim, targets)


def mixup_batch(x: torch.Tensor, y: torch.Tensor,
                alpha: float, rng: torch.Generator
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Sample λ from Beta(α, α), blend half the batch with a shuffled copy."""
    B = x.size(0)
    lam = float(np.random.default_rng().beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)   # canonical Mixup: always blend with dominant side
    idx = torch.randperm(B, device=x.device, generator=rng)
    x_mix = lam * x + (1.0 - lam) * x[idx]
    return x_mix, y, y[idx], lam


# ──────────────────── Pre-train + fine-tune ─────────


def pretrain_simclr(
    raw_train: List[Dict[str, np.ndarray]],
    model: PCDPlusClassifier,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    temperature: float,
    seed: int,
) -> None:
    """Warm up ``model.encoder`` via NT-Xent on two-view augmentations."""
    ds = PCDPlusTwoViewDataset(raw_train, seed=seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0,
                        drop_last=True)
    proj = SimCLRProjection(in_dim=model.encoder.net[0].out_features,
                            out_dim=128).to(device)
    params = list(model.encoder.parameters()) + list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    model.encoder.train()
    proj.train()
    for ep in range(epochs):
        ds.set_epoch(ep)
        loss_sum, n = 0.0, 0
        for v1, v2 in loader:
            v1 = v1.to(device)
            v2 = v2.to(device)
            z1 = proj(model.encoder(v1))
            z2 = proj(model.encoder(v2))
            loss = nt_xent_loss(z1, z2, temperature=temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item() * v1.size(0)
            n += v1.size(0)
        print(f"  [simclr] ep {ep:3d}  loss={loss_sum / max(n, 1):.4f}")


def train(args: argparse.Namespace) -> None:
    saraga_dir = Path(args.saraga_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    clips = index_clips_overlap(
        saraga_dir,
        clip_seconds=args.clip_seconds,
        hop_seconds=args.hop_seconds,
        skip_stems=args.skip_stems,
    )
    if not clips:
        raise RuntimeError(f"No clips under {saraga_dir}")
    print(f"Indexed {len(clips)} clips "
          f"(clip={args.clip_seconds}s, hop={args.hop_seconds}s) "
          f"from {len({c.recording_id for c in clips})} recordings, "
          f"{len({c.raga for c in clips})} ragas.")

    rng = np.random.default_rng(args.seed)
    train_clips, val_clips = recording_level_split(clips, rng)
    print(f"Split: train={len(train_clips)} val={len(val_clips)} "
          f"({len({c.recording_id for c in val_clips})} val recordings)")

    tonics = resolve_tonics(clips, saraga_dir)
    src_counts: Dict[str, int] = {}
    for _, src in tonics.values():
        src_counts[src] = src_counts.get(src, 0) + 1
    print(f"Tonic sources: {src_counts}")

    # Raga vocab (UNK = 0 for consistency with v1).
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

    print("\n== Train raw PESTO frames ==")
    raw_train = extract_raw_frames(
        train_clips, tonics, out_dir / "pcd_raw_v2_train.pkl",
        device=device, hop_ms=args.hop_ms,
    )
    y_train = np.array([vocab[c.raga] for c in train_clips], dtype=np.int64)

    print("\n== Val raw PESTO frames ==")
    raw_val = extract_raw_frames(
        val_clips, tonics, out_dir / "pcd_raw_v2_val.pkl",
        device=device, hop_ms=args.hop_ms,
    ) if val_clips else []
    y_val = np.array([vocab[c.raga] for c in val_clips], dtype=np.int64)

    # Sanity: voiced-frame counts.
    tf = np.array([(r["pitch_hz"] > 0).sum() for r in raw_train])
    vf = np.array([(r["pitch_hz"] > 0).sum() for r in raw_val])
    print(f"Voiced frames / clip  train: mean={tf.mean():.0f} med={np.median(tf):.0f} "
          f"empty={(tf == 0).sum()}/{len(tf)}")
    if len(vf):
        print(f"Voiced frames / clip  val  : mean={vf.mean():.0f} med={np.median(vf):.0f} "
              f"empty={(vf == 0).sum()}/{len(vf)}")

    train_ds = PCDPlusDataset(raw_train, y_train, seed=args.seed, augment=True)
    val_ds = PCDPlusDataset(raw_val, y_val, seed=args.seed + 1, augment=False) \
        if raw_val else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0) if val_ds else None

    model = PCDPlusClassifier(n_classes=n_classes, in_dim=PCD_PLUS_DIM,
                              hidden=args.hidden, dropout=args.dropout
                              ).to(device)
    print(f"Model: {PCD_PLUS_DIM} → {args.hidden} → {args.hidden} → {n_classes}  "
          f"({sum(p.numel() for p in model.parameters())/1e3:.1f}k params)")

    # Optional SimCLR pre-training.
    if args.pretrain_epochs > 0:
        print(f"\n== SimCLR pre-training ({args.pretrain_epochs} epochs) ==")
        pretrain_simclr(
            raw_train, model, device,
            epochs=args.pretrain_epochs,
            batch_size=max(args.batch_size, 64),
            lr=args.pretrain_lr,
            weight_decay=args.weight_decay,
            temperature=args.simclr_temperature,
            seed=args.seed + 7,
        )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    gen = torch.Generator(device=device).manual_seed(args.seed + 11)

    print(f"\n== Supervised training ({args.epochs} epochs) ==")
    history = []
    best_key = -1.0
    best_state = None

    for ep in range(args.epochs):
        train_ds.set_epoch(ep)

        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            if args.mixup_alpha > 0 and (ep % 2 == 0):
                x_mix, y_a, y_b, lam = mixup_batch(x, y, args.mixup_alpha, gen)
                logits = model(x_mix)
                loss = lam * loss_fn(logits, y_a) + (1 - lam) * loss_fn(logits, y_b)
                # For accuracy bookkeeping, use the dominant-side label.
                tr_correct += (logits.argmax(-1) == y_a).sum().item()
            else:
                logits = model(x)
                loss = loss_fn(logits, y)
                tr_correct += (logits.argmax(-1) == y).sum().item()

            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * x.size(0)
            tr_total += x.size(0)
        tr_loss /= max(tr_total, 1)
        tr_acc = tr_correct / max(tr_total, 1)

        val_loss = val_acc = float("nan")
        if val_loader is not None:
            model.eval()
            vl, vc, vt = 0.0, 0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    y = y.to(device)
                    logits = model(x)
                    l = loss_fn(logits, y)
                    vl += l.item() * x.size(0)
                    vc += (logits.argmax(-1) == y).sum().item()
                    vt += x.size(0)
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
                "feature_type": "pcd_v2",
                "in_dim": PCD_PLUS_DIM,
                "hidden": args.hidden,
                "clip_seconds": args.clip_seconds,
                "hop_seconds": args.hop_seconds,
                "hop_ms": args.hop_ms,
                "label_smoothing": args.label_smoothing,
                "mixup_alpha": args.mixup_alpha,
                "pretrain_epochs": args.pretrain_epochs,
                "epoch": ep, "val_acc": val_acc, "tr_acc": tr_acc,
            }, out_dir / "model.pt")

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nbest = {best_key:.3f} → {out_dir/'model.pt'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--saraga_dir",        default="data/saraga_kalyan_thaat")
    p.add_argument("--output",            default="checkpoints/raga_classifier_pcd_v2")
    p.add_argument("--clip_seconds",      type=float, default=30.0)
    p.add_argument("--hop_seconds",       type=float, default=10.0)
    p.add_argument("--hop_ms",            type=float, default=10.0)
    p.add_argument("--hidden",            type=int,   default=256)
    p.add_argument("--dropout",           type=float, default=0.4)
    p.add_argument("--epochs",            type=int,   default=50)
    p.add_argument("--batch_size",        type=int,   default=32)
    p.add_argument("--lr",                type=float, default=1e-3)
    p.add_argument("--weight_decay",      type=float, default=1e-2)
    p.add_argument("--label_smoothing",   type=float, default=0.1)
    p.add_argument("--mixup_alpha",       type=float, default=0.2,
                   help="Beta(α,α) for Mixup. 0 disables.")
    p.add_argument("--pretrain_epochs",   type=int,   default=20,
                   help="SimCLR pre-training epochs. 0 disables.")
    p.add_argument("--pretrain_lr",       type=float, default=1e-3)
    p.add_argument("--simclr_temperature", type=float, default=0.2)
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--skip_stems",        nargs="*", default=[])
    args = p.parse_args()
    train(args)
