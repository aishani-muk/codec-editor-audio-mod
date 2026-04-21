"""
2x2 live dashboard rendered from ``training_log.jsonl``.

Usage:
    python scripts/plot_run.py --log checkpoints/<run>/training_log.jsonl \
        --vocab 4096

Panels:
    (top-left)    cross-entropy nats, train + val, with log(V) dashed ceiling
    (top-right)   info-fraction %, train + val (0=random, 100=perfect)
    (bottom-left) top-1 and top-5 token accuracy %, train + val
    (bottom-right) learning-rate schedule

The training loop invokes this automatically at every eval step; the PNG
is overwritten in place so any viewer with file-watch (VSCode preview,
``eog``, ``feh --reload 5``) hot-reloads.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_log(path: Path):
    train, val = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "val_loss" in e:
                val.append(e)
            elif "loss" in e:
                train.append(e)
    return train, val


def _series(rows, key):
    xs, ys = [], []
    for r in rows:
        if key in r and r[key] is not None:
            xs.append(r["step"])
            ys.append(r[key])
    return xs, ys


def plot_run(log_path: Path, output_path: Path | None, vocab_size: int) -> None:
    train, val = _read_log(log_path)
    if not train and not val:
        print(f"no log rows in {log_path}")
        return

    ceiling = math.log(max(vocab_size, 2))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)

    # ── top-left: CE ───────────────────────────────────────────────────────
    ax = axes[0, 0]
    x, y = _series(train, "train_ce_nats")
    if x:
        ax.plot(x, y, label="train", color="C0", lw=1.0, alpha=0.8)
    else:
        x, y = _series(train, "loss")
        if x:
            ax.plot(x, y, label="train (raw loss)", color="C0", lw=1.0, alpha=0.6)
    x, y = _series(val, "val_ce_nats")
    if not x:
        x, y = _series(val, "val_loss")
    if x:
        ax.plot(x, y, label="val", color="C3", lw=1.6, marker="o", markersize=4)
    ax.axhline(ceiling, ls="--", color="grey", lw=0.8,
               label=f"log V = {ceiling:.2f}")
    ax.set_ylabel("Cross-entropy (nats)")
    ax.set_title("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # ── top-right: info fraction % ─────────────────────────────────────────
    ax = axes[0, 1]
    x, y = _series(train, "train_info_frac_pct")
    if x:
        ax.plot(x, y, label="train", color="C0", lw=1.0, alpha=0.8)
    x, y = _series(val, "val_info_frac_pct")
    if x:
        ax.plot(x, y, label="val", color="C3", lw=1.6, marker="o", markersize=4)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Info captured (%)")
    ax.set_title("Information fraction (0 = random, 100 = perfect)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # ── bottom-left: top-k accuracy ────────────────────────────────────────
    ax = axes[1, 0]
    x, y = _series(train, "train_top1_acc_pct")
    if x:
        ax.plot(x, y, color="C0", ls="-",  lw=1.0, alpha=0.8, label="train top1")
    x, y = _series(train, "train_top5_acc_pct")
    if x:
        ax.plot(x, y, color="C0", ls="--", lw=1.0, alpha=0.8, label="train top5")
    x, y = _series(val, "val_top1_acc_pct")
    if x:
        ax.plot(x, y, color="C3", ls="-",  lw=1.6, marker="o", markersize=4,
                label="val top1")
    x, y = _series(val, "val_top5_acc_pct")
    if x:
        ax.plot(x, y, color="C3", ls="--", lw=1.6, marker="s", markersize=4,
                label="val top5")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Top-k token accuracy")
    ax.set_xlabel("Optimizer step")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # ── bottom-right: LR ───────────────────────────────────────────────────
    ax = axes[1, 1]
    x, y = _series(train, "lr")
    if x:
        ax.plot(x, y, color="C2", lw=1.0)
    ax.set_ylabel("Learning rate")
    ax.set_title("LR schedule")
    ax.set_xlabel("Optimizer step")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"{log_path.parent.name}   (vocab = {vocab_size})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if output_path is None:
        output_path = log_path.with_name("loss_curves.png")
    fig.savefig(output_path, dpi=130)
    plt.close(fig)

    # Summary lines.
    val_ce = _series(val, "val_ce_nats") or _series(val, "val_loss")
    if val_ce[1]:
        best = min(val_ce[1])
        last = val_ce[1][-1]
        print(
            f"{log_path.parent.name}: "
            f"val_ce last={last:.4f} best={best:.4f} "
            f"(info_frac_best={100 * (1 - best / ceiling):.2f}%)  "
            f"-> {output_path}"
        )
    else:
        print(f"{log_path.parent.name}: no val metrics yet  ->  {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True, type=Path)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--vocab", type=int, required=True,
                   help="Effective output vocab (e.g. 4096 for raw WavTokenizer, "
                        "8192 for BPE).")
    args = p.parse_args()
    plot_run(args.log, args.output, args.vocab)
