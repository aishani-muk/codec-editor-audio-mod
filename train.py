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
import math
import os
import random
import re
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import yaml

from models.codec_editor import CodecEditor
from models.stress_proxy import SyntheticStressProxy, StressEmbedding

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAVE_TB = True
except ImportError:  # pragma: no cover
    SummaryWriter = None
    _HAVE_TB = False


_RAGA_PREFIX_RE = re.compile(r"^(?:Raag|Raga)_?", re.IGNORECASE)


def extract_raga(track_stem: str) -> str:
    """Pull a canonical raga name from a metadata ``track_stem``.

    Saraga track stems we see in practice:
      ``<performance>__Raag_<Name>__Raag_<Name>``  (most)
      ``<performance>__<Name>__<Name>``            (a couple without Raag_)
      ``<performance>__<Name>``                    (single-segment fallback)

    Strategy:
      - Split on ``__``.
      - Prefer the penultimate segment; if it equals the last one (common
        "Raag_X__Raag_X" convention) use either.
      - Strip leading ``Raag_``/``Raga_`` (case-insensitive).
      - Collapse runs of underscores, then return.
      - Return ``"UNK"`` only if the input is empty or whitespace-only.
    """
    if not track_stem or not track_stem.strip():
        return "UNK"
    parts = [p for p in track_stem.split("__") if p]
    if not parts:
        return "UNK"
    if len(parts) >= 2 and parts[-1] == parts[-2]:
        candidate = parts[-1]
    elif len(parts) >= 2:
        candidate = parts[-2]
    else:
        candidate = parts[-1]
    candidate = _RAGA_PREFIX_RE.sub("", candidate)
    candidate = re.sub(r"_+", "_", candidate).strip("_")
    return candidate or "UNK"


def build_raga_vocab(dataset: "PairedTokenDataset") -> dict[str, int]:
    """Return a deterministic ``raga_name -> id`` mapping (0 = UNK)."""
    names = set()
    for pair in dataset.pairs:
        names.add(pair["raga"])
    names.discard("UNK")
    vocab = {"UNK": 0}
    for i, name in enumerate(sorted(names), start=1):
        vocab[name] = i
    return vocab


class PairedTokenDataset(Dataset):
    """
    Dataset of (input_tokens, target_tokens, lambda) triples.

    Loads BPE-encoded token sequences from the paired_edits directory.
    """

    def __init__(self, pairs_dir: str, tokens_dir: str,
                 max_seq_len: int = 2048, use_bpe: bool = True,
                 tokens_subdir: str | None = None,
                 raga_vocab: dict[str, int] | None = None):
        self.max_seq_len = max_seq_len
        self.use_bpe = use_bpe
        self.raga_vocab = raga_vocab  # may be set after init via attach_raga_vocab
        self.pairs = []

        subdir = tokens_subdir or ("bpe" if use_bpe else "wavtok")
        input_token_dir = Path(tokens_dir) / f"input_{subdir}"
        target_token_dir = Path(tokens_dir) / f"target_{subdir}"
        meta_dir = Path(pairs_dir) / "meta"
        manifest_path = Path(pairs_dir) / "manifest.jsonl"

        if not input_token_dir.exists():
            print(f"WARNING: {input_token_dir} not found. Run tokenization first.")
            return

        if manifest_path.exists():
            with open(manifest_path) as f:
                stems = [json.loads(line)["stem"] for line in f if line.strip()]
        else:
            stems = sorted(p.stem for p in meta_dir.glob("*.npz"))

        for stem in stems:
            input_path = input_token_dir / f"{stem}.npy"
            target_path = target_token_dir / f"{stem}.npy"
            meta_path = meta_dir / f"{stem}.npz"
            if input_path.exists() and target_path.exists() and meta_path.exists():
                meta = np.load(meta_path, allow_pickle=True)
                stem_str = str(meta.get("track_stem", meta.get("source_file", stem)))
                self.pairs.append({
                    "input": str(input_path),
                    "target": str(target_path),
                    "meta": str(meta_path),
                    "track_stem": stem_str,
                    "raga": extract_raga(stem_str),
                })

        print(f"Loaded {len(self.pairs)} training pairs "
              f"(tokens subdir: {subdir})")

    def attach_raga_vocab(self, raga_vocab: dict[str, int]) -> None:
        self.raga_vocab = raga_vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        input_tokens = np.load(pair["input"]).astype(np.int64)
        target_tokens = np.load(pair["target"]).astype(np.int64)
        meta = np.load(pair["meta"], allow_pickle=True)
        lam = float(meta["lambda_val"])

        # Raw WavTokenizer codes are stored as (n_q, T); BPE outputs are (T,).
        # Collapse to 1-D regardless so downstream code is shape-uniform.
        if input_tokens.ndim == 2:
            input_tokens = input_tokens.reshape(-1)
        if target_tokens.ndim == 2:
            target_tokens = target_tokens.reshape(-1)

        # Truncate to max_seq_len
        input_tokens = input_tokens[:self.max_seq_len]
        target_tokens = target_tokens[:self.max_seq_len]

        raga_id = 0
        if self.raga_vocab is not None:
            raga_id = self.raga_vocab.get(pair["raga"], 0)

        return {
            "input_ids": torch.from_numpy(input_tokens),
            "target_ids": torch.from_numpy(target_tokens),
            "u": torch.tensor(lam, dtype=torch.float32),
            "raga_id": torch.tensor(raga_id, dtype=torch.long),
            "input_len": torch.tensor(len(input_tokens), dtype=torch.long),
            "target_len": torch.tensor(len(target_tokens), dtype=torch.long),
        }


def make_collate_fn(crop_tokens: int | None = None,
                    crop_rng_seed: int | None = None):
    """Build a collate function.

    If ``crop_tokens`` is an int and the clip is longer, each example is
    randomly cropped to that many tokens *before* padding. Input and target
    share the same time offset so they stay pair-aligned — this is how we
    teach the editor the same short windows it will see during streaming
    inference (``editor.window_sec * tokens_per_sec``).

    ``crop_rng_seed`` is seeded once per worker for reproducibility.
    """
    rng = np.random.default_rng(crop_rng_seed)

    def collate_fn(batch):
        if crop_tokens is not None:
            new_batch = []
            for b in batch:
                n = int(b["input_len"])
                m = int(b["target_len"])
                aligned = min(n, m)
                if aligned > crop_tokens:
                    start = int(rng.integers(0, aligned - crop_tokens + 1))
                    bc = dict(b)
                    bc["input_ids"] = b["input_ids"][start:start + crop_tokens]
                    bc["target_ids"] = b["target_ids"][start:start + crop_tokens]
                    bc["input_len"] = torch.tensor(crop_tokens,
                                                   dtype=torch.long)
                    bc["target_len"] = torch.tensor(crop_tokens,
                                                    dtype=torch.long)
                    new_batch.append(bc)
                else:
                    new_batch.append(b)
            batch = new_batch

        B = len(batch)
        max_in = max(int(b["input_len"]) for b in batch)
        max_tgt = max(int(b["target_len"]) for b in batch)

        input_ids = torch.zeros(B, max_in, dtype=torch.long)
        target_ids = torch.zeros(B, max_tgt, dtype=torch.long)
        input_mask = torch.zeros(B, max_in, dtype=torch.long)
        target_mask = torch.zeros(B, max_tgt, dtype=torch.long)
        u_vals = torch.zeros(B)
        raga_ids = torch.zeros(B, dtype=torch.long)

        for i, b in enumerate(batch):
            n_in = int(b["input_len"])
            n_out = int(b["target_len"])
            input_ids[i, :n_in] = b["input_ids"]
            target_ids[i, :n_out] = b["target_ids"]
            input_mask[i, :n_in] = 1
            target_mask[i, :n_out] = 1
            u_vals[i] = b["u"]
            raga_ids[i] = b["raga_id"]

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "u": u_vals,
            "raga_ids": raga_ids,
            "input_mask": input_mask,
            "target_mask": target_mask,
            "attention_mask": torch.cat([input_mask, target_mask], dim=1),
        }

    return collate_fn


# Back-compat: plain uncropped collate.
collate_fn = make_collate_fn(crop_tokens=None)


# =============================================================================
# Metric helpers
# =============================================================================

def _ce_and_topk(logits: torch.Tensor, labels: torch.Tensor,
                 vocab_size: int, ks=(1, 5)) -> dict[str, float]:
    """Given raw (B, T, V) logits and (B, T) labels (``-100`` = ignore),
    return cross-entropy (nats), perplexity, info-fraction %, top-k accuracy %.

    Shifting follows GPT-2 convention: logits[:, :-1] predicts labels[:, 1:].
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    mask = shift_labels.ne(-100)
    n_valid = int(mask.sum())

    if n_valid == 0:
        return {"ce_nats": float("nan"), "ppl": float("nan"),
                "info_frac_pct": float("nan"),
                **{f"top{k}_acc_pct": float("nan") for k in ks}, "n": 0}

    flat_logits = shift_logits.reshape(-1, shift_logits.size(-1))  # (N, V)
    flat_labels = shift_labels.reshape(-1)                         # (N,)
    keep = flat_labels.ne(-100)
    flat_logits = flat_logits[keep]
    flat_labels = flat_labels[keep]

    ce = torch.nn.functional.cross_entropy(flat_logits, flat_labels,
                                           reduction="mean")
    ce_val = float(ce.item())

    # Top-k accuracy.
    topk_acc = {}
    maxk = max(ks)
    _, pred = flat_logits.topk(maxk, dim=-1)     # (N, maxk)
    correct = pred.eq(flat_labels.unsqueeze(-1)) # (N, maxk)
    for k in ks:
        topk_acc[f"top{k}_acc_pct"] = float(
            correct[:, :k].any(dim=-1).float().mean().item() * 100.0
        )

    log_vocab = math.log(max(vocab_size, 2))
    info_frac = max(0.0, 100.0 * (1.0 - ce_val / log_vocab))

    return {
        "ce_nats": ce_val,
        "ppl": math.exp(min(ce_val, 50.0)),
        "info_frac_pct": info_frac,
        "n": int(keep.sum()),
        **topk_acc,
    }


def _token_hist_jsd_loss(logits: torch.Tensor, target_ids: torch.Tensor,
                         target_mask: torch.Tensor, T_in: int,
                         vocab_size: int) -> torch.Tensor:
    """Batched JSD between the predicted and empirical token-histograms
    over the target window. Acts as a tractable proxy for the
    pitch-class distribution preservation criterion (Koduri et al.) —
    each WavTokenizer code roughly represents one ~13 ms audio frame,
    so the distribution of codes over a window correlates with the
    distribution of acoustic events, which dominates raga fingerprinting.

    Differentiable end-to-end; O(B·T·V) memory but V≈4–8 k so fine.

    Returns a scalar tensor (mean over batch) on the logits' device.
    """
    # Logits are the un-shifted (B, T_total, V). Predictions for target[t]
    # live at position (T_in - 1) + t, i.e. the indices T_in-1 ... T_in-1+T_out-1.
    T_out = target_ids.size(1)
    if T_out == 0:
        return logits.new_zeros(())

    tgt_logits = logits[:, T_in - 1:T_in - 1 + T_out, :]      # (B, T_out, V)
    pred_dist = torch.softmax(tgt_logits.float(), dim=-1)     # (B, T_out, V)
    w = target_mask.float().unsqueeze(-1)                     # (B, T_out, 1)
    w_sum = w.sum(dim=1).clamp(min=1.0)                       # (B, 1)
    pred_hist = (pred_dist * w).sum(dim=1) / w_sum            # (B, V)

    # Empirical target histogram (one-hot average, masked).
    tgt_oh = torch.nn.functional.one_hot(
        target_ids.clamp(min=0, max=vocab_size - 1),
        num_classes=vocab_size,
    ).float()                                                 # (B, T_out, V)
    tgt_hist = (tgt_oh * w).sum(dim=1) / w_sum                # (B, V)

    eps = 1e-10
    m = 0.5 * (pred_hist + tgt_hist)
    log_m = torch.log(m.clamp(min=eps))
    kl_p = (pred_hist * (torch.log(pred_hist.clamp(min=eps)) - log_m)).sum(-1)
    kl_q = (tgt_hist * (torch.log(tgt_hist.clamp(min=eps)) - log_m)).sum(-1)
    jsd = 0.5 * (kl_p + kl_q)                                 # (B,)
    return jsd.mean()


def _format_metrics(tag: str, m: dict[str, float]) -> str:
    return (
        f"{tag}: ce={m['ce_nats']:.4f}  ppl={m['ppl']:.1f}  "
        f"info={m['info_frac_pct']:.2f}%  "
        f"top1={m['top1_acc_pct']:.2f}%  top5={m['top5_acc_pct']:.2f}%"
    )


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base``. Nested dicts are merged
    key-by-key; leaf values in ``override`` replace those in ``base``.
    """
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_config_with_inheritance(config_path: str) -> dict:
    """Recursively resolve ``_base_`` pointers with a deep merge: a child
    may override individual sub-keys without wiping its parent's whole
    sub-dict. Supports arbitrary inheritance chains.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if "_base_" in cfg:
        base_path = str(Path(config_path).parent / cfg["_base_"])
        base_cfg = _load_config_with_inheritance(base_path)
        cfg = _deep_merge(base_cfg, {k: v for k, v in cfg.items() if k != "_base_"})
    return cfg


def _load_pretrain_weights(model: torch.nn.Module,
                           pretrain_ckpt: str, device: str) -> None:
    """Load a ``pretrain_codec_lm.py`` checkpoint into the editor.

    The pretraining script saved a plain ``model.state_dict()`` that
    shares the same keys we need (``transformer.*``, ``stress_proj``,
    ``role_embed``). Raga and input-residual branches are ignored from
    the pretrain file (they're untrained there) and remain with their
    fresh random init.
    """
    pretrain_sd = torch.load(pretrain_ckpt, map_location=device)
    tgt_sd = model.state_dict()
    loaded, skipped = 0, 0
    for k, v in pretrain_sd.items():
        if k in tgt_sd and tgt_sd[k].shape == v.shape:
            tgt_sd[k] = v
            loaded += 1
        else:
            skipped += 1
    missing_keys = [k for k in tgt_sd if k not in pretrain_sd]
    model.load_state_dict(tgt_sd)
    print(f"Loaded pretrain weights from {pretrain_ckpt}: "
          f"{loaded} tensors loaded, {skipped} shape/name mismatched, "
          f"{len(missing_keys)} editor-specific tensors kept random "
          f"(raga, residual).")


def train(config_path: str, run_name: str, pretrain_ckpt: str | None = None):
    """Main training function."""
    cfg = _load_config_with_inheritance(config_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    # Setup checkpoint directory
    ckpt_dir = Path(cfg["checkpoints"]["dir"]) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save frozen config
    with open(ckpt_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    # ── Dataset first: we need the raga vocab before instantiating the model ──
    editor_cfg = cfg["editor"]
    bpe_cfg = cfg["bpe"]
    data_cfg = cfg["data"]

    # The model concatenates [input | target] before the GPT-2 block, so each
    # side must be <= max_seq_len // 2 to keep the concatenation within the
    # learned positional-embedding range.
    dataset = PairedTokenDataset(
        pairs_dir=data_cfg["pairs_dir"],
        tokens_dir=data_cfg["tokens_dir"],
        max_seq_len=editor_cfg["max_seq_len"] // 2,
        use_bpe=bpe_cfg["enabled"],
        tokens_subdir=data_cfg.get("tokens_subdir"),
    )

    if len(dataset) == 0:
        print("\nNo training data found. Please run the data preparation pipeline.")
        return

    use_raga_label = bool(editor_cfg.get("use_raga_label", True))
    if use_raga_label:
        raga_vocab = build_raga_vocab(dataset)
        dataset.attach_raga_vocab(raga_vocab)
        n_ragas = len(raga_vocab) - 1  # exclude UNK from the count
        print(f"Raga vocabulary ({n_ragas} ragas + UNK): {raga_vocab}")
    else:
        raga_vocab = None
        n_ragas = 0

    # ── Initialize model ──
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
        n_ragas=n_ragas,
        use_input_residual=bool(editor_cfg.get("use_input_residual", True)),
    ).to(device)

    if pretrain_ckpt:
        _load_pretrain_weights(model, pretrain_ckpt, device)

    effective_vocab = (
        bpe_cfg["vocab_size"] if bpe_cfg["enabled"]
        else cfg["wavtokenizer"]["codebook_size"]
    )
    print(f"Effective output vocab: {effective_vocab} "
          f"(log = {math.log(effective_vocab):.3f} nats)")

    stress_embed = StressEmbedding(
        embed_dim=cfg["stress_proxy"]["embed_dim"]
    ).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_stress = sum(p.numel() for p in stress_embed.parameters())
    print(f"Editor parameters: {n_params:,}")
    print(f"Stress embed parameters: {n_stress:,}")

    # ── Source-level train/val split ──
    # Group pairs by track_stem (stored in the meta .npz) so that all clips +
    # all λ values of the same recording land on the same side of the split.
    # Prevents leakage: random-split would put (same song, different λ) on
    # both sides, giving artificially-low val loss.
    idx_by_track = defaultdict(list)
    for i, pair in enumerate(dataset.pairs):
        meta = np.load(pair["meta"])
        stem = str(meta.get("track_stem", meta.get("source_file", f"unknown_{i}")))
        idx_by_track[stem].append(i)

    held_out = set(cfg["data"].get("held_out_tracks") or [])
    shuf_tracks = sorted(t for t in idx_by_track if not any(h in t for h in held_out))
    rng = random.Random(cfg.get("seed", 42))
    rng.shuffle(shuf_tracks)
    n_train_tracks = max(1, int(len(shuf_tracks) * cfg["data"]["train_split"]))
    train_tracks = set(shuf_tracks[:n_train_tracks])
    val_tracks = set(shuf_tracks[n_train_tracks:]) | {
        t for t in idx_by_track if any(h in t for h in held_out)
    }

    train_idx = [i for t in train_tracks for i in idx_by_track[t]]
    val_idx = [i for t in val_tracks for i in idx_by_track[t]]
    # Edge case: only one track → fall back to random split within
    if not val_idx:
        print("WARNING: only one source track; falling back to random split.")
        all_idx = list(range(len(dataset)))
        rng.shuffle(all_idx)
        split = int(len(all_idx) * cfg["data"]["train_split"])
        train_idx, val_idx = all_idx[:split], all_idx[split:]

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    print(f"Source-level split: {len(train_tracks)} train tracks ({len(train_idx)} pairs), "
          f"{len(val_tracks)} val tracks ({len(val_idx)} pairs)")

    train_cfg = cfg["training"]

    # Windowed-crop training: force train examples to the streaming-inference
    # window length so training-time and deployment-time distributions match.
    # Disabled (None) if editor.train_crop_tokens is not set or 0.
    crop_tokens = editor_cfg.get("train_crop_tokens")
    if crop_tokens in (None, 0, False):
        crop_tokens = None
        print("Training on full clips (no windowed cropping).")
    else:
        crop_tokens = int(crop_tokens)
        print(f"Training with random windowed crops of {crop_tokens} tokens "
              f"(~{crop_tokens / cfg['wavtokenizer']['tokens_per_sec']:.2f}s)")

    train_collate = make_collate_fn(crop_tokens=crop_tokens,
                                    crop_rng_seed=cfg.get("seed", 42))
    val_collate = make_collate_fn(crop_tokens=None)

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"],
        shuffle=True, collate_fn=train_collate,
        num_workers=train_cfg["dataloader_num_workers"],
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg["batch_size"],
        shuffle=False, collate_fn=val_collate,
    )

    # Optimizer & scheduler (linear warmup → cosine decay)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(stress_embed.parameters()),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    try:
        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=train_cfg.get("warmup_steps", 0),
            num_training_steps=train_cfg["max_steps"],
        )
    except ImportError:
        # Fallback (warmup gets ignored)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_cfg["max_steps"],
        )
        print("WARNING: transformers not installed; using CosineAnnealingLR (no warmup).")

    # Gradient accumulation
    grad_accum = max(1, train_cfg.get("gradient_accumulation_steps", 1))

    # Auxiliary token-histogram JSD loss weight (soft pitch-class-distribution
    # preservation regulariser). 0 disables it entirely.
    aux_jsd_weight = float(train_cfg.get("aux_jsd_weight", 0.0))
    if aux_jsd_weight > 0:
        print(f"Auxiliary token-histogram JSD loss enabled "
              f"(weight={aux_jsd_weight})")

    # Early stopping
    early_stop_patience = train_cfg.get("early_stop_patience", 5)  # n eval intervals

    # Training loop
    log_path = ckpt_dir / "training_log.jsonl"
    global_step = 0           # counts optimizer steps (not micro-batches)
    micro_step = 0            # counts forward passes
    best_val_loss = float("inf")
    patience_counter = 0

    scaler = torch.amp.GradScaler("cuda") if train_cfg["fp16"] and device == "cuda" else None
    clip_params = list(model.parameters()) + list(stress_embed.parameters())

    # TensorBoard writer (optional).
    tb_writer = None
    if _HAVE_TB:
        tb_writer = SummaryWriter(log_dir=str(ckpt_dir / "tb"))
        print(f"TensorBoard logs: {ckpt_dir / 'tb'}")

    # Running window of the most recent train-batch metric dicts (~100 batches).
    train_metric_buf: deque = deque(maxlen=100)

    print(f"\nStarting training for {train_cfg['max_steps']} optimizer steps "
          f"(grad_accum={grad_accum}, warmup={train_cfg.get('warmup_steps', 0)}, "
          f"early_stop_patience={early_stop_patience})...")
    model.train()
    stress_embed.train()
    optimizer.zero_grad()
    should_stop = False

    while global_step < train_cfg["max_steps"] and not should_stop:
        for batch in train_loader:
            if global_step >= train_cfg["max_steps"] or should_stop:
                break

            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            u = batch["u"].to(device)
            raga_ids_b = batch["raga_ids"].to(device) if use_raga_label else None
            attn_mask = batch["attention_mask"].to(device)
            tgt_mask = batch["target_mask"].to(device)

            # Generate stress embeddings
            B, T_in = input_ids.shape
            u_expanded = u.unsqueeze(1).expand(B, T_in)
            stress_emb = stress_embed(u_expanded)

            # Forward + backward (scaled by 1/grad_accum)
            if scaler:
                with torch.amp.autocast("cuda"):
                    outputs = model(
                        input_ids, target_ids, stress_emb,
                        attention_mask=attn_mask,
                        raga_ids=raga_ids_b,
                        target_mask=tgt_mask,
                    )
                    base_loss = outputs["loss"]
                    aux_loss = (
                        _token_hist_jsd_loss(
                            outputs["logits"], target_ids, tgt_mask,
                            T_in=T_in, vocab_size=effective_vocab,
                        ) if aux_jsd_weight > 0 else base_loss.new_zeros(())
                    )
                    loss = (base_loss + aux_jsd_weight * aux_loss) / grad_accum
                scaler.scale(loss).backward()
            else:
                outputs = model(
                    input_ids, target_ids, stress_emb,
                    attention_mask=attn_mask,
                    raga_ids=raga_ids_b,
                    target_mask=tgt_mask,
                )
                base_loss = outputs["loss"]
                aux_loss = (
                    _token_hist_jsd_loss(
                        outputs["logits"], target_ids, tgt_mask,
                        T_in=T_in, vocab_size=effective_vocab,
                    ) if aux_jsd_weight > 0 else base_loss.new_zeros(())
                )
                loss = (base_loss + aux_jsd_weight * aux_loss) / grad_accum
                loss.backward()

            # Track 5 metrics on a running window of the most-recent train batches.
            with torch.no_grad():
                full_labels = torch.cat([
                    torch.full_like(input_ids, -100),
                    target_ids.clone(),
                ], dim=1)
                full_labels[:, T_in:][tgt_mask == 0] = -100
                step_metrics = _ce_and_topk(
                    outputs["logits"].detach().float(), full_labels,
                    vocab_size=effective_vocab,
                )
            train_metric_buf.append(step_metrics)

            micro_step += 1

            # Only step the optimizer once every grad_accum micro-batches
            if micro_step % grad_accum != 0:
                continue

            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(clip_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(clip_params, max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

            # Logging (report un-scaled loss + rolling-mean metrics)
            if global_step % 100 == 0:
                def _rm(key):
                    vals = [m[key] for m in train_metric_buf if not math.isnan(m[key])]
                    return float(np.mean(vals)) if vals else float("nan")
                train_ce = _rm("ce_nats")
                train_ppl = _rm("ppl")
                train_info = _rm("info_frac_pct")
                train_t1 = _rm("top1_acc_pct")
                train_t5 = _rm("top5_acc_pct")
                log_entry = {
                    "step": global_step,
                    "loss": loss.item() * grad_accum,
                    "base_loss": float(base_loss.detach().item()),
                    "aux_jsd_loss": float(aux_loss.detach().item()),
                    "lr": scheduler.get_last_lr()[0],
                    "train_ce_nats": train_ce,
                    "train_ppl": train_ppl,
                    "train_info_frac_pct": train_info,
                    "train_top1_acc_pct": train_t1,
                    "train_top5_acc_pct": train_t5,
                }
                with open(log_path, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
                if tb_writer is not None:
                    tb_writer.add_scalar("train/ce_nats", train_ce, global_step)
                    tb_writer.add_scalar("train/ppl", train_ppl, global_step)
                    tb_writer.add_scalar("train/info_frac_pct", train_info, global_step)
                    tb_writer.add_scalar("train/top1_acc_pct", train_t1, global_step)
                    tb_writer.add_scalar("train/top5_acc_pct", train_t5, global_step)
                    tb_writer.add_scalar("train/base_loss", log_entry["base_loss"], global_step)
                    tb_writer.add_scalar("train/aux_jsd_loss", log_entry["aux_jsd_loss"], global_step)
                    tb_writer.add_scalar("lr", log_entry["lr"], global_step)
                print(
                    f"  Step {global_step}  lr={log_entry['lr']:.2e}  "
                    + _format_metrics("train(rm100)",
                        {"ce_nats": train_ce, "ppl": train_ppl,
                         "info_frac_pct": train_info,
                         "top1_acc_pct": train_t1, "top5_acc_pct": train_t5})
                )

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
                val_cum = {"ce_sum": 0.0, "n": 0,
                           "top1_correct": 0, "top5_correct": 0}
                with torch.no_grad():
                    for vb in val_loader:
                        vi = vb["input_ids"].to(device)
                        vt = vb["target_ids"].to(device)
                        vu = vb["u"].to(device)
                        v_raga = vb["raga_ids"].to(device) if use_raga_label else None
                        v_attn = vb["attention_mask"].to(device)
                        v_tmask = vb["target_mask"].to(device)
                        B_v, T_v = vi.shape
                        vu_exp = vu.unsqueeze(1).expand(B_v, T_v)
                        vs = stress_embed(vu_exp)
                        vo = model(vi, vt, vs,
                                   attention_mask=v_attn,
                                   raga_ids=v_raga,
                                   target_mask=v_tmask)
                        # Accumulate token-level stats (not averaged per batch).
                        full_labels_v = torch.cat([
                            torch.full_like(vi, -100), vt.clone()
                        ], dim=1)
                        full_labels_v[:, T_v:][v_tmask == 0] = -100
                        shift_logits = vo["logits"][..., :-1, :].contiguous()
                        shift_labels = full_labels_v[..., 1:].contiguous()
                        flat_l = shift_logits.reshape(-1, shift_logits.size(-1))
                        flat_y = shift_labels.reshape(-1)
                        keep = flat_y.ne(-100)
                        if keep.any():
                            fl = flat_l[keep]
                            fy = flat_y[keep]
                            ce_batch = torch.nn.functional.cross_entropy(
                                fl, fy, reduction="sum")
                            val_cum["ce_sum"] += float(ce_batch.item())
                            val_cum["n"] += int(keep.sum().item())
                            _, pred = fl.topk(5, dim=-1)
                            correct = pred.eq(fy.unsqueeze(-1))
                            val_cum["top1_correct"] += int(correct[:, :1].any(dim=-1).sum().item())
                            val_cum["top5_correct"] += int(correct[:, :5].any(dim=-1).sum().item())

                n_tok = max(val_cum["n"], 1)
                val_ce = val_cum["ce_sum"] / n_tok
                val_metrics = {
                    "ce_nats": val_ce,
                    "ppl": math.exp(min(val_ce, 50.0)),
                    "info_frac_pct": max(0.0, 100.0 * (1.0 - val_ce / math.log(effective_vocab))),
                    "top1_acc_pct": 100.0 * val_cum["top1_correct"] / n_tok,
                    "top5_acc_pct": 100.0 * val_cum["top5_correct"] / n_tok,
                }
                val_loss = val_ce
                print(f"  {_format_metrics('VAL', val_metrics)}")

                entry = {"step": global_step, "val_loss": val_loss,
                         **{f"val_{k}": v for k, v in val_metrics.items()}}
                with open(log_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
                if tb_writer is not None:
                    for k, v in val_metrics.items():
                        tb_writer.add_scalar(f"val/{k}", v, global_step)

                # Auto-render the 2x2 loss/metrics dashboard PNG.
                try:
                    subprocess.run(
                        [sys.executable, str(Path("scripts") / "plot_run.py"),
                         "--log", str(log_path),
                         "--vocab", str(effective_vocab)],
                        check=False, timeout=30,
                    )
                except Exception as e:  # non-fatal
                    print(f"  (plot_run.py failed: {e})")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_dir = ckpt_dir / "best"
                    best_dir.mkdir(exist_ok=True)
                    torch.save(model.state_dict(), best_dir / "model.pt")
                    torch.save(stress_embed.state_dict(),
                               best_dir / "stress_embed.pt")
                    with open(best_dir / "info.json", "w") as f:
                        json.dump({"step": global_step,
                                   "val_loss": val_loss,
                                   **val_metrics}, f)
                    print(f"  New best model saved (val_loss={val_loss:.4f})")
                else:
                    patience_counter += 1
                    print(f"  No val-loss improvement ({patience_counter}/{early_stop_patience})")
                    if patience_counter >= early_stop_patience:
                        print(f"  Early stopping triggered at step {global_step} "
                              f"(best val_loss={best_val_loss:.4f})")
                        should_stop = True

                model.train()
                stress_embed.train()

    if tb_writer is not None:
        tb_writer.close()
    print(f"\nTraining complete. Checkpoints in: {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train codec-to-codec editor")
    parser.add_argument("--config", required=True, help="Config YAML path")
    parser.add_argument("--run_name", required=True, help="Run name for checkpointing")
    parser.add_argument("--pretrain_ckpt", default=None,
                        help="Optional pretrain_codec_lm.py checkpoint to "
                             "initialise the editor from (path to .pt)")
    args = parser.parse_args()

    train(args.config, args.run_name, pretrain_ckpt=args.pretrain_ckpt)
