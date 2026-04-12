"""
Main training loop for the conditional codec-to-codec editor.

Trains the GPT-2-based editor on synthetic paired edits:
  input_tokens (from WavTokenizer+BPE) + stress_embedding → edited_tokens

Uses HuggingFace Trainer with:
  - Periodic checkpointing (every N steps)
  - Best-model saving (by validation JSD)
  - Logging to TensorBoard / W&B
  - FP16 mixed precision on GPU

Usage:
    python train.py --config configs/proposed.yaml --run_name proposed_v1

On NEXUS cluster:
    sbatch --gres=gpu:1 --mem=32G --time=12:00:00 \\
      --wrap="python train.py --config configs/proposed.yaml --run_name proposed_v1"
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import yaml

from models.codec_editor import CodecEditor
from models.stress_proxy import SyntheticStressProxy, StressEmbedding


class PairedTokenDataset(Dataset):
    """
    Dataset of (input_tokens, target_tokens, lambda) triples.

    Loads BPE-encoded token sequences from the paired_edits directory.
    """

    def __init__(self, pairs_dir: str, tokens_dir: str, max_seq_len: int = 2048):
        self.max_seq_len = max_seq_len
        self.pairs = []

        input_token_dir = Path(tokens_dir) / "input_bpe"
        target_token_dir = Path(tokens_dir) / "target_bpe"
        meta_dir = Path(pairs_dir) / "meta"

        if not input_token_dir.exists():
            print(f"WARNING: {input_token_dir} not found. Run tokenization first.")
            return

        for meta_path in sorted(meta_dir.glob("*.npz")):
            stem = meta_path.stem
            input_path = input_token_dir / f"{stem}.npy"
            target_path = target_token_dir / f"{stem}.npy"

            if input_path.exists() and target_path.exists():
                self.pairs.append({
                    "input": str(input_path),
                    "target": str(target_path),
                    "meta": str(meta_path),
                })

        print(f"Loaded {len(self.pairs)} training pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        input_tokens = np.load(pair["input"]).astype(np.int64)
        target_tokens = np.load(pair["target"]).astype(np.int64)
        meta = np.load(pair["meta"])
        lam = float(meta["lambda_val"])

        # Truncate to max_seq_len
        input_tokens = input_tokens[:self.max_seq_len]
        target_tokens = target_tokens[:self.max_seq_len]

        return {
            "input_ids": torch.from_numpy(input_tokens),
            "target_ids": torch.from_numpy(target_tokens),
            "u": torch.tensor(lam, dtype=torch.float32),
        }


def collate_fn(batch):
    """Pad sequences to the same length within a batch."""
    max_in = max(b["input_ids"].shape[0] for b in batch)
    max_tgt = max(b["target_ids"].shape[0] for b in batch)

    input_ids = torch.zeros(len(batch), max_in, dtype=torch.long)
    target_ids = torch.zeros(len(batch), max_tgt, dtype=torch.long)
    u_vals = torch.zeros(len(batch))

    for i, b in enumerate(batch):
        input_ids[i, :b["input_ids"].shape[0]] = b["input_ids"]
        target_ids[i, :b["target_ids"].shape[0]] = b["target_ids"]
        u_vals[i] = b["u"]

    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "u": u_vals,
    }


def train(config_path: str, run_name: str):
    """Main training function."""
    # Load config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Load base config if specified
    if "_base_" in cfg:
        base_path = Path(config_path).parent / cfg["_base_"]
        with open(base_path) as f:
            base_cfg = yaml.safe_load(f)
        base_cfg.update({k: v for k, v in cfg.items() if k != "_base_"})
        cfg = base_cfg

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    # Setup checkpoint directory
    ckpt_dir = Path(cfg["checkpoints"]["dir"]) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save frozen config
    with open(ckpt_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    # Initialize model
    editor_cfg = cfg["editor"]
    bpe_cfg = cfg["bpe"]
    model = CodecEditor(
        vocab_size=cfg["wavtokenizer"]["codebook_size"],
        bpe_vocab_size=bpe_cfg["vocab_size"] if bpe_cfg["enabled"] else None,
        n_layers=editor_cfg["n_layers"],
        n_heads=editor_cfg["n_heads"],
        d_model=editor_cfg["d_model"],
        d_ff=editor_cfg["d_ff"],
        max_seq_len=editor_cfg["max_seq_len"],
        dropout=editor_cfg["dropout"],
        stress_embed_dim=cfg["stress_proxy"]["embed_dim"],
    ).to(device)

    stress_embed = StressEmbedding(
        embed_dim=cfg["stress_proxy"]["embed_dim"]
    ).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_stress = sum(p.numel() for p in stress_embed.parameters())
    print(f"Editor parameters: {n_params:,}")
    print(f"Stress embed parameters: {n_stress:,}")

    # Dataset
    dataset = PairedTokenDataset(
        pairs_dir=cfg["data"]["pairs_dir"],
        tokens_dir=cfg["data"]["tokens_dir"],
        max_seq_len=editor_cfg["max_seq_len"],
    )

    if len(dataset) == 0:
        print("\nNo training data found. Please run the data preparation pipeline:")
        print("  1. python data/download_saraga.py")
        print("  2. python data/prepare_pairs.py")
        print("  3. python tokenize/encode_wavtokenizer.py (for input and target)")
        print("  4. python tokenize/train_bpe.py")
        print("  5. python tokenize/apply_bpe.py")
        return

    # Split into train/val
    train_size = int(len(dataset) * cfg["data"]["train_split"])
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_cfg = cfg["training"]
    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"],
        shuffle=True, collate_fn=collate_fn,
        num_workers=train_cfg["dataloader_num_workers"],
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg["batch_size"],
        shuffle=False, collate_fn=collate_fn,
    )

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(stress_embed.parameters()),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["max_steps"],
    )

    # Training loop
    log_path = ckpt_dir / "training_log.jsonl"
    global_step = 0
    best_val_loss = float("inf")

    scaler = torch.amp.GradScaler("cuda") if train_cfg["fp16"] and device == "cuda" else None

    print(f"\nStarting training for {train_cfg['max_steps']} steps...")
    model.train()
    stress_embed.train()

    while global_step < train_cfg["max_steps"]:
        for batch in train_loader:
            if global_step >= train_cfg["max_steps"]:
                break

            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            u = batch["u"].to(device)

            # Generate stress embeddings
            B, T_in = input_ids.shape
            u_expanded = u.unsqueeze(1).expand(B, T_in)
            stress_emb = stress_embed(u_expanded)

            # Forward
            optimizer.zero_grad()
            if scaler:
                with torch.amp.autocast("cuda"):
                    outputs = model(input_ids, target_ids, stress_emb)
                    loss = outputs["loss"]
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(stress_embed.parameters()),
                    max_norm=1.0,
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids, target_ids, stress_emb)
                loss = outputs["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(stress_embed.parameters()),
                    max_norm=1.0,
                )
                optimizer.step()

            scheduler.step()
            global_step += 1

            # Logging
            if global_step % 100 == 0:
                log_entry = {
                    "step": global_step,
                    "loss": loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                }
                with open(log_path, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
                print(f"  Step {global_step}: loss={loss.item():.4f}")

            # Checkpoint
            if global_step % train_cfg["save_steps"] == 0:
                step_dir = ckpt_dir / f"step_{global_step}"
                step_dir.mkdir(exist_ok=True)
                torch.save(model.state_dict(), step_dir / "model.pt")
                torch.save(stress_embed.state_dict(), step_dir / "stress_embed.pt")
                torch.save(optimizer.state_dict(), step_dir / "optimizer.pt")
                print(f"  Saved checkpoint: {step_dir}")

            # Validation
            if global_step % train_cfg["eval_steps"] == 0:
                model.eval()
                stress_embed.eval()
                val_losses = []
                with torch.no_grad():
                    for vb in val_loader:
                        vi = vb["input_ids"].to(device)
                        vt = vb["target_ids"].to(device)
                        vu = vb["u"].to(device)
                        B_v, T_v = vi.shape
                        vu_exp = vu.unsqueeze(1).expand(B_v, T_v)
                        vs = stress_embed(vu_exp)
                        vo = model(vi, vt, vs)
                        val_losses.append(vo["loss"].item())

                val_loss = np.mean(val_losses) if val_losses else float("inf")
                print(f"  Validation loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_dir = ckpt_dir / "best"
                    best_dir.mkdir(exist_ok=True)
                    torch.save(model.state_dict(), best_dir / "model.pt")
                    torch.save(stress_embed.state_dict(),
                               best_dir / "stress_embed.pt")
                    print(f"  New best model saved (loss={val_loss:.4f})")

                model.train()
                stress_embed.train()

    print(f"\nTraining complete. Checkpoints in: {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train codec-to-codec editor")
    parser.add_argument("--config", required=True, help="Config YAML path")
    parser.add_argument("--run_name", required=True, help="Run name for checkpointing")
    args = parser.parse_args()

    train(args.config, args.run_name)
