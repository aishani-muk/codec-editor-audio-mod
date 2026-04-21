"""
Train a MERT-based valence/arousal regressor on DEAM.

Pipeline:
  1. Load DEAM's static per-song V/A annotations (annotations_averaged_songs.csv).
  2. Extract MERT-v1-95M features for every song (mean-pooled last hidden state).
  3. Train a 2-layer MLP 768 → 256 → 2 with MSE loss for ~30 epochs.
  4. Save the regressor to checkpoints/deam_regressor/model.pt.

At eval time, ``evaluation/deam_regressor.py::predict_va(wav)`` returns a
(valence, arousal) pair on the DEAM scale (1–9) — downstream code computes
``delta = out - in`` and a boolean ``moved_toward_neutral``.

References:
  * Aljanaki, Yang, Soleymani. "Developing a benchmark for emotional
    analysis of music." PLOS ONE 12(3), 2017.
  * Li et al., "MERT: Acoustic Music Understanding Model with Large-Scale
    Self-supervised Training," ICLR 2024.

Usage:
    python train_deam_regressor.py \\
        --deam_dir data/deam \\
        --output checkpoints/deam_regressor \\
        --epochs 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from evaluation.mert_features import MERTFeatureExtractor


# ──────────────────── Data loading ────────────────────


def locate_deam_paths(deam_dir: Path) -> Tuple[Path, Path]:
    """Find DEAM's static annotation CSV and its audio dir.

    Handles common extraction layouts:
      * ``deam_dir/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs.csv``
      * ``deam_dir/annotations_averaged_songs.csv``
    """
    # CSV.
    csv_candidates = list(deam_dir.rglob("*annotations_averaged_songs*.csv")) \
        + list(deam_dir.rglob("*static_annotations_averaged*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(
            f"No DEAM annotation CSV under {deam_dir}. "
            f"Download instructions: python data/download_deam.py --output {deam_dir}"
        )
    csv_path = csv_candidates[0]

    # Audio dir.
    audio_dir = deam_dir / "audio"
    if not audio_dir.is_dir() or not any(audio_dir.iterdir()):
        # Search deeper.
        cands = [p.parent for p in deam_dir.rglob("MEMD_audio_*.mp3")]
        if cands:
            audio_dir = cands[0]
        else:
            raise FileNotFoundError(
                f"No DEAM audio under {deam_dir}/audio/. See "
                f"data/download_deam.py for instructions."
            )

    return csv_path, audio_dir


def build_song_index(csv_path: Path, audio_dir: Path) -> pd.DataFrame:
    """Read DEAM CSV + verify audio exists for each song_id."""
    df = pd.read_csv(csv_path)
    # DEAM CSV columns vary; normalise.
    df.columns = [c.strip().lower() for c in df.columns]
    col_id = next(c for c in df.columns if "song" in c and "id" in c)
    col_val = next(c for c in df.columns if c.startswith("valence") and "mean" in c)
    col_aro = next(c for c in df.columns if c.startswith("arousal") and "mean" in c)
    df = df.rename(columns={col_id: "song_id",
                            col_val: "valence", col_aro: "arousal"})

    # Resolve audio paths.
    paths = []
    for sid in df["song_id"]:
        # Try both MEMD_audio_{id}.mp3 and {id}.mp3.
        for stem in (f"MEMD_audio_{sid}", str(sid)):
            for ext in (".mp3", ".wav"):
                p = audio_dir / f"{stem}{ext}"
                if p.exists():
                    paths.append(str(p))
                    break
            else:
                continue
            break
        else:
            paths.append(None)
    df["audio_path"] = paths
    df = df.dropna(subset=["audio_path", "valence", "arousal"])
    df = df[["song_id", "audio_path", "valence", "arousal"]].reset_index(drop=True)
    return df


# ──────────────────── Feature caching ────────────────────


def extract_or_load_features(df: pd.DataFrame, cache_path: Path,
                             mert: MERTFeatureExtractor) -> np.ndarray:
    """Compute MERT features per audio, cache to ``cache_path``."""
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True).item()
        if cached["song_ids"] == df["song_id"].tolist():
            print(f"  loaded MERT features from {cache_path}")
            return cached["features"]

    print(f"  extracting MERT features for {len(df)} songs ...")
    feats = np.zeros((len(df), MERTFeatureExtractor.embed_dim), dtype=np.float32)
    failures: List[int] = []
    for i, row in enumerate(tqdm(df.itertuples(index=False), total=len(df))):
        try:
            feats[i] = mert.embed_wav_file(row.audio_path)
        except Exception as e:  # skip unreadable files
            failures.append(i)
            print(f"    skip {row.audio_path}: {type(e).__name__}: {e}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, {"song_ids": df["song_id"].tolist(), "features": feats,
                          "failures": failures})
    print(f"  cached MERT features to {cache_path} "
          f"({len(failures)} skipped)")
    return feats


# ──────────────────── Model & training ────────────────────


class VARegressor(nn.Module):
    """MERT-768 → 256 → 2 (valence, arousal). Outputs on DEAM scale 1–9."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DEAMDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.from_numpy(features).float()
        self.targets = torch.from_numpy(targets).float()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i], self.targets[i]


def r2_score(pred: np.ndarray, true: np.ndarray) -> float:
    ss_res = ((pred - true) ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum()
    return 1.0 - float(ss_res / (ss_tot + 1e-8))


def train(args: argparse.Namespace) -> None:
    deam_dir = Path(args.deam_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path, audio_dir = locate_deam_paths(deam_dir)
    print(f"DEAM CSV:    {csv_path}")
    print(f"DEAM audio:  {audio_dir}")

    df = build_song_index(csv_path, audio_dir)
    print(f"usable songs with audio: {len(df)}")
    if len(df) < 100:
        raise RuntimeError(
            f"Only {len(df)} songs have locatable audio — refusing to train "
            f"a regressor on so few. Check your DEAM download."
        )

    mert = MERTFeatureExtractor()
    features = extract_or_load_features(
        df, out_dir / "mert_features.npy", mert
    )
    targets = df[["valence", "arousal"]].to_numpy(dtype=np.float32)

    full_ds = DEAMDataset(features, targets)
    n_val = max(1, int(len(full_ds) * 0.15))
    gen = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(full_ds, [len(full_ds) - n_val, n_val],
                                    generator=gen)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VARegressor().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    history = []
    for epoch in range(args.epochs):
        model.train()
        tr_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(x)
        tr_loss /= len(train_ds)
        sched.step()

        # Val.
        model.eval()
        preds, ys = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                preds.append(model(x).cpu().numpy())
                ys.append(y.numpy())
        pred_np = np.concatenate(preds)
        y_np = np.concatenate(ys)
        val_mse = float(((pred_np - y_np) ** 2).mean())
        r2_v = r2_score(pred_np[:, 0], y_np[:, 0])
        r2_a = r2_score(pred_np[:, 1], y_np[:, 1])
        history.append({"epoch": epoch, "tr_mse": tr_loss,
                        "val_mse": val_mse, "r2_val": r2_v, "r2_aro": r2_a})
        print(f"epoch {epoch:3d}  tr_mse={tr_loss:.3f}  "
              f"val_mse={val_mse:.3f}  R²_val={r2_v:.3f}  R²_aro={r2_a:.3f}")

        if val_mse < best_val:
            best_val = val_mse
            torch.save({
                "state_dict": model.state_dict(),
                "in_dim": 768, "hidden": 256,
                "mert_name": MERTFeatureExtractor.hub_name,
                "epoch": epoch, "val_mse": val_mse,
                "r2_valence": r2_v, "r2_arousal": r2_a,
            }, out_dir / "model.pt")

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nbest val MSE = {best_val:.3f} — saved to {out_dir/'model.pt'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--deam_dir", default="data/deam")
    p.add_argument("--output",   default="checkpoints/deam_regressor")
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch_size", type=int,   default=64)
    p.add_argument("--lr",         type=float, default=1e-3)
    args = p.parse_args()
    train(args)
