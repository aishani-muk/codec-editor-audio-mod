"""
Raw-codec language-model pretraining for the CodecEditor.

Trains the SAME GPT-2 backbone that the full editor uses, but as a plain
next-token predictor on unpaired speech-75 target-side codec tokens.
No raga/stress/residual conditioning — just ``tokens[:-1] -> tokens[1:]``.

Purpose: give the token-embedding table and the lower transformer layers
a reasonable acoustic-prior before they are fine-tuned on the (input,
target, lambda) editing task. On a tiny paired-edit dataset, this prior
dramatically improves the generalisation ceiling.

Output: a ``pretrain.pt`` checkpoint in ``checkpoints/<run_name>/`` that
train.py can consume via ``--pretrain_ckpt``.

Usage:
    python pretrain_codec_lm.py --config configs/pretrain_codec_lm.yaml \\
                                --run_name pretrain_codec_lm_v1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import yaml

from models.codec_editor import CodecEditor

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAVE_TB = True
except ImportError:
    SummaryWriter = None
    _HAVE_TB = False


# =============================================================================
# Config helpers (reused from train.py; duplicated here to keep the script
# standalone).
# =============================================================================

def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_config_with_inheritance(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if "_base_" in cfg:
        base_path = str(Path(config_path).parent / cfg["_base_"])
        base_cfg = _load_config_with_inheritance(base_path)
        cfg = _deep_merge(base_cfg,
                          {k: v for k, v in cfg.items() if k != "_base_"})
    return cfg


# =============================================================================
# Dataset: unconditional codec LM on target-side tokens
# =============================================================================

class CodecLMDataset(Dataset):
    """Flat dataset of codec-token sequences.

    Scans every ``.npy`` file in ``tokens_dir`` (target side). Stores
    paths only; loads and random-crops to ``crop_tokens`` at ``__getitem__``
    time so we can do many effective passes per clip.
    """

    def __init__(self, tokens_dir: str, crop_tokens: int, rng_seed: int = 42):
        self.paths = sorted(Path(tokens_dir).glob("*.npy"))
        self.crop_tokens = int(crop_tokens)
        self.rng = np.random.default_rng(rng_seed)
        print(f"CodecLMDataset: {len(self.paths)} token files "
              f"(crop={self.crop_tokens})")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        arr = np.load(self.paths[idx]).astype(np.int64)
        if arr.ndim == 2:
            arr = arr.reshape(-1)   # (n_q, T) -> (T,)
        if len(arr) > self.crop_tokens:
            start = int(self.rng.integers(0, len(arr) - self.crop_tokens + 1))
            arr = arr[start:start + self.crop_tokens]
        return {
            "input_ids": torch.from_numpy(arr),
            "length": torch.tensor(len(arr), dtype=torch.long),
        }


def codec_lm_collate(batch):
    """Right-pad to the max length in the batch; build attention mask."""
    B = len(batch)
    max_len = max(int(b["length"]) for b in batch)
    input_ids = torch.zeros(B, max_len, dtype=torch.long)
    attn = torch.zeros(B, max_len, dtype=torch.long)
    for i, b in enumerate(batch):
        n = int(b["length"])
        input_ids[i, :n] = b["input_ids"]
        attn[i, :n] = 1
    return {"input_ids": input_ids, "attention_mask": attn}


# =============================================================================
# Pretraining loop
# =============================================================================

def pretrain(config_path: str, run_name: str):
    cfg = _load_config_with_inheritance(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Pretraining on: {device}")

    ckpt_dir = Path(cfg["checkpoints"]["dir"]) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    # ── Model (same arch as editor, but we don't exercise the residual /
    #       raga / stress paths at all during pretraining). ──
    ed = cfg["editor"]
    model = CodecEditor(
        vocab_size=cfg["wavtokenizer"]["codebook_size"],
        bpe_vocab_size=(cfg["bpe"]["vocab_size"]
                        if cfg["bpe"]["enabled"] else None),
        n_layers=ed["n_layers"],
        n_heads=ed["n_heads"],
        d_model=ed["d_model"],
        d_ff=ed["d_ff"],
        max_seq_len=ed["max_seq_len"],
        dropout=ed["dropout"],
        stress_embed_dim=cfg["stress_proxy"]["embed_dim"],
        n_ragas=0,                  # no raga conditioning during pretraining
        use_input_residual=False,   # unconditional next-token LM only
    ).to(device)
    print(f"Pretrain editor parameters: "
          f"{sum(p.numel() for p in model.parameters()):,}")

    vocab_size = (cfg["bpe"]["vocab_size"] if cfg["bpe"]["enabled"]
                  else cfg["wavtokenizer"]["codebook_size"])
    print(f"Effective vocab: {vocab_size} "
          f"(log = {math.log(vocab_size):.3f} nats)")

    # ── Dataset: target-side tokens only. ──
    data_cfg = cfg["data"]
    subdir = data_cfg.get("tokens_subdir") or (
        "bpe" if cfg["bpe"]["enabled"] else "wavtok"
    )
    tokens_dir = Path(data_cfg["tokens_dir"]) / f"target_{subdir}"
    if not tokens_dir.exists():
        raise FileNotFoundError(
            f"Tokens directory not found: {tokens_dir}. "
            "Run the tokenization step first."
        )

    crop_tokens = int(ed.get("train_crop_tokens") or 150)
    full_ds = CodecLMDataset(str(tokens_dir), crop_tokens=crop_tokens,
                             rng_seed=cfg.get("seed", 42))

    if len(full_ds) == 0:
        raise RuntimeError(f"No token files found in {tokens_dir}")

    # Source-level-ish 90/10 split: since we only have filenames, shuffle
    # deterministically and split. Leakage is less concerning here because
    # the objective is just a codec prior.
    all_idx = list(range(len(full_ds)))
    rng = random.Random(cfg.get("seed", 42))
    rng.shuffle(all_idx)
    split = int(len(all_idx) * 0.9)
    train_idx, val_idx = all_idx[:split], all_idx[split:]
    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)
    print(f"Split: {len(train_idx)} train, {len(val_idx)} val")

    t = cfg["training"]
    train_loader = DataLoader(
        train_ds, batch_size=t["batch_size"], shuffle=True,
        collate_fn=codec_lm_collate,
        num_workers=t["dataloader_num_workers"],
    )
    val_loader = DataLoader(
        val_ds, batch_size=t["batch_size"], shuffle=False,
        collate_fn=codec_lm_collate,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=t["learning_rate"],
        weight_decay=t["weight_decay"],
    )
    try:
        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=t.get("warmup_steps", 0),
            num_training_steps=t["max_steps"],
        )
    except ImportError:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t["max_steps"]
        )

    scaler = (torch.amp.GradScaler("cuda")
              if t.get("fp16") and device == "cuda" else None)

    tb_writer = SummaryWriter(log_dir=str(ckpt_dir / "tb")) if _HAVE_TB else None
    log_path = ckpt_dir / "pretrain_log.jsonl"

    global_step = 0
    best_val = float("inf")
    print(f"\nPretraining for {t['max_steps']} steps (warmup {t.get('warmup_steps', 0)})...")

    model.train()
    while global_step < t["max_steps"]:
        for batch in train_loader:
            if global_step >= t["max_steps"]:
                break

            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = input_ids.clone()
            labels[attn == 0] = -100  # pad positions ignored

            optimizer.zero_grad()
            if scaler:
                with torch.amp.autocast("cuda"):
                    # Bypass CodecEditor's conditioning machinery entirely
                    # by going straight through the underlying GPT-2.
                    out = model.transformer(
                        input_ids=input_ids, attention_mask=attn,
                        labels=labels,
                    )
                    loss = out.loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model.transformer(
                    input_ids=input_ids, attention_mask=attn, labels=labels,
                )
                loss = out.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            global_step += 1

            if global_step % 100 == 0:
                ce = float(loss.detach().item())
                ppl = math.exp(min(ce, 50.0))
                info = max(0.0, 100.0 * (1.0 - ce / math.log(vocab_size)))
                print(f"  Step {global_step}  lr={scheduler.get_last_lr()[0]:.2e}  "
                      f"ce={ce:.4f}  ppl={ppl:.1f}  info={info:.2f}%")
                with open(log_path, "a") as f:
                    f.write(json.dumps({
                        "step": global_step, "train_ce": ce,
                        "train_ppl": ppl, "train_info_frac_pct": info,
                        "lr": scheduler.get_last_lr()[0],
                    }) + "\n")
                if tb_writer:
                    tb_writer.add_scalar("pretrain/train_ce", ce, global_step)
                    tb_writer.add_scalar("pretrain/info_frac_pct", info,
                                         global_step)

            # Val + save-best
            if global_step % t["eval_steps"] == 0:
                model.eval()
                val_ce_sum, val_n = 0.0, 0
                with torch.no_grad():
                    for vb in val_loader:
                        vi = vb["input_ids"].to(device)
                        va = vb["attention_mask"].to(device)
                        vl = vi.clone()
                        vl[va == 0] = -100
                        vo = model.transformer(
                            input_ids=vi, attention_mask=va, labels=vl,
                        )
                        mask = vl.ne(-100)
                        n_tok = int(mask.sum())
                        val_ce_sum += float(vo.loss.item()) * n_tok
                        val_n += n_tok
                val_ce = val_ce_sum / max(val_n, 1)
                val_info = max(0.0, 100.0 * (1.0 - val_ce / math.log(vocab_size)))
                print(f"  VAL: ce={val_ce:.4f}  info={val_info:.2f}%  "
                      f"n_tok={val_n}")
                with open(log_path, "a") as f:
                    f.write(json.dumps({
                        "step": global_step, "val_ce": val_ce,
                        "val_info_frac_pct": val_info,
                    }) + "\n")
                if tb_writer:
                    tb_writer.add_scalar("pretrain/val_ce", val_ce, global_step)
                    tb_writer.add_scalar("pretrain/val_info_frac_pct",
                                         val_info, global_step)
                if val_ce < best_val:
                    best_val = val_ce
                    torch.save(model.state_dict(), ckpt_dir / "pretrain.pt")
                    print(f"  New best pretrain checkpoint (val_ce={val_ce:.4f})")
                model.train()

    if tb_writer:
        tb_writer.close()
    print(f"\nPretraining complete. Best val_ce={best_val:.4f}. "
          f"Weights: {ckpt_dir / 'pretrain.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain codec LM")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_name", required=True)
    args = parser.parse_args()
    pretrain(args.config, args.run_name)
