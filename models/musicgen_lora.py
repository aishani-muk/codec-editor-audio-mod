"""LoRA adapter + MusicGen-small editor wrapper (rescue-v3 Day 3 Track B).

What this file provides
-----------------------
* ``LoRALinear``         — minimal-deps (torch-only) low-rank adapter
                           that wraps a frozen ``nn.Linear`` with a
                           trainable ``α/r · A·B`` side path.
* ``inject_lora_into_musicgen`` — walks a MusicGen decoder and replaces
                           the q/k/v/o projections of every self- and
                           cross-attention module with ``LoRALinear``s.
                           All other parameters are frozen. Returns the
                           total trainable-param count.
* ``make_raga_u_prompt``  — builds the natural-language prompt we
                           repurpose MusicGen's text conditioning for,
                           e.g. ``"raga yaman, calm, u=0.6"``.
* ``MusicGenEditor``      — wrapper exposing the editor-style forward
                           signature used by ``train_lora_musicgen.py``.
                           Builds ``decoder_input_ids`` = concat of
                           input + target audio codes, teacher-forces
                           the decoder, and returns a loss that only
                           counts target-window positions.

Design notes
------------
* **No peft dependency.** The full LoRA implementation here is ~50 lines;
  adding another package to the venv isn't worth it for a single
  ablation track.
* **Audio conditioning via decoder prefix.** MusicGen's official
  melody-conditioning input is a separate melody-encoder hook we don't
  have on the HF port. The simplest thing that works with HF's
  ``MusicgenForConditionalGeneration`` is to prepend the input-audio
  Encodec codes to the decoder input and mask them from the CE loss.
  The decoder then learns "produce target given (text, input-as-prefix)".
* **No delay-pattern gymnastics in the loss.** HF's
  ``MusicgenForConditionalGeneration.forward`` handles the delay
  pattern internally when you pass ``decoder_input_ids`` and ``labels``;
  we do the same and let it do its thing.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """Wrap a frozen ``nn.Linear`` with an additive low-rank adapter.

    Output = ``frozen_linear(x) + (α/r) · dropout(x A^T) B^T``.

    Shapes: ``A: (r, in)``, ``B: (out, r)``.
    ``A`` is Kaiming-uniform, ``B`` is zero → the adapter starts as an
    exact no-op.
    """

    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / max(r, 1)
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        self.lora_drop = (
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x)
        h = self.lora_drop(x)
        # x @ A.T → (..., r) ; @ B.T → (..., out)
        h = F.linear(h, self.lora_A)
        h = F.linear(h, self.lora_B) * self.scaling
        return base + h


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------


_DEFAULT_TARGET_MODULES: tuple[str, ...] = ("q_proj", "k_proj", "v_proj",
                                             "out_proj")


def _iter_named_children(module: nn.Module, prefix: str = ""):
    """Depth-first walk yielding ``(full_name, parent_module, attr_name, child)``."""
    for name, child in module.named_children():
        full = f"{prefix}.{name}" if prefix else name
        yield full, module, name, child
        yield from _iter_named_children(child, full)


def inject_lora_into_musicgen(
    model: nn.Module,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Sequence[str] = _DEFAULT_TARGET_MODULES,
    include_cross_attn: bool = True,
    include_self_attn: bool = True,
) -> dict:
    """Walk ``model.decoder`` (or the model itself if no ``.decoder``)
    and replace attention-projection ``nn.Linear`` layers with
    ``LoRALinear``s.

    All non-LoRA parameters are frozen. Other branches of the MusicGen
    model (text encoder, audio encoder) are also frozen.

    Returns a dict summary with counts of replaced modules and total
    trainable params.
    """
    # Freeze everything first.
    for p in model.parameters():
        p.requires_grad_(False)

    # Focus the traversal on the decoder to avoid touching Encodec/T5.
    decoder = getattr(model, "decoder", model)

    # MusicGen wraps its core autoregressive LM in a chain:
    #   model.decoder (MusicgenForCausalLM) -> .model (MusicgenModel)
    #     -> .decoder (MusicgenDecoder) -> .layers
    # Traverse from the LM wrapper inward so we cover both self_attn
    # and encoder_attn uniformly.
    n_replaced = 0
    n_trainable = 0
    for full, parent, attr_name, child in _iter_named_children(decoder):
        if not isinstance(child, nn.Linear):
            continue
        if attr_name not in target_modules:
            continue
        # Only touch attention layers to stay within the ~3 M param budget.
        # Attention modules expose exactly these four linear children.
        if not (include_self_attn and "self_attn" in full) and not (
                include_cross_attn and "encoder_attn" in full):
            continue
        lora = LoRALinear(child, r=r, alpha=alpha, dropout=dropout)
        setattr(parent, attr_name, lora)
        n_replaced += 1

    for p in model.parameters():
        if p.requires_grad:
            n_trainable += p.numel()

    return {
        "n_lora_modules": n_replaced,
        "n_trainable_params": n_trainable,
        "r": r, "alpha": alpha, "dropout": dropout,
        "target_modules": list(target_modules),
        "include_self_attn": include_self_attn,
        "include_cross_attn": include_cross_attn,
    }


def lora_state_dict(model: nn.Module) -> dict:
    """Return only the LoRA adapter tensors for compact checkpointing."""
    out: dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad and (
                name.endswith("lora_A") or name.endswith("lora_B")):
            out[name] = param.detach().cpu().clone()
    return out


def load_lora_state_dict(model: nn.Module, sd: dict) -> int:
    """Load a LoRA-only state dict produced by ``lora_state_dict``.
    Returns the number of tensors loaded.
    """
    own = dict(model.named_parameters())
    n = 0
    for k, v in sd.items():
        if k not in own:
            continue
        own[k].data.copy_(v.to(own[k].device))
        n += 1
    return n


# ---------------------------------------------------------------------------
# Prompt & editor wrapper
# ---------------------------------------------------------------------------


def _raga_slug(raga: str) -> str:
    return raga.replace("_", " ").strip().lower() or "unknown"


def _u_to_emotion(u: float) -> str:
    # Natural-language gloss of u that T5 was pretrained on. We keep it
    # short and consistent so the text encoder returns stable
    # representations across examples.
    if u < 0.15:
        return "neutral"
    if u < 0.45:
        return "slightly calmer"
    if u < 0.75:
        return "calm"
    return "very calm"


def make_raga_u_prompt(raga: str, u: float,
                       emotion_target: str | None = None) -> str:
    """Compose the tiny natural-language prompt we feed to T5.

    Example: ``"raga yaman, calm, u=0.60"``.
    ``emotion_target`` lets the caller override the default u-derived
    gloss (e.g. to ``"pleasant"`` for ΔV>0 targets).
    """
    gloss = emotion_target or _u_to_emotion(float(u))
    return f"raga {_raga_slug(raga)}, {gloss}, u={float(u):.2f}"


class MusicGenEditor(nn.Module):
    """Thin wrapper that makes a ``MusicgenForConditionalGeneration``
    behave like our editor (input tokens + u → target tokens).

    Usage
    -----
        mg = MusicgenForConditionalGeneration.from_pretrained(...)
        processor = AutoProcessor.from_pretrained(...)
        inject_lora_into_musicgen(mg, r=16, alpha=32)
        editor = MusicGenEditor(mg, processor)

        # forward(input_codes=(B,n_q,T_in),
        #         target_codes=(B,n_q,T_out),
        #         prompts=list[str]) → dict(loss, logits, target_mask)
    """

    def __init__(self, mg_model: nn.Module, processor, max_text_len: int = 32):
        super().__init__()
        self.mg = mg_model
        self.processor = processor
        self.max_text_len = max_text_len
        # ``num_codebooks`` lives on the decoder config in transformers
        # >=5.x; the EncodecConfig no longer exposes it. MusicGen-small
        # always uses 4 codebooks, but we look it up defensively.
        self.n_codebooks = int(
            getattr(self.mg.audio_encoder.config, "num_codebooks", None)
            or getattr(self.mg.decoder.config, "num_codebooks", None)
            or 4
        )

    # ----- text side -----

    def _encode_prompts(self, prompts: Sequence[str], device: torch.device):
        tok = self.processor.tokenizer(
            list(prompts), padding=True, truncation=True,
            max_length=self.max_text_len, return_tensors="pt",
        )
        input_ids = tok["input_ids"].to(device)
        attn_mask = tok["attention_mask"].to(device)
        return input_ids, attn_mask

    # ----- audio side -----

    @staticmethod
    def _concat_audio_codes(
        input_codes: torch.Tensor, target_codes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Concatenate input + target along the time axis and build a
        label tensor with ``-100`` on the input positions so CE ignores
        them. Shapes: each input is ``(B, n_q, T)``.
        """
        B, n_q, T_in = input_codes.shape
        _, _, T_out = target_codes.shape
        dec = torch.cat([input_codes, target_codes], dim=-1)          # (B,n_q,T_in+T_out)
        labels = dec.clone()
        labels[..., :T_in] = -100
        return dec, labels

    # ----- forward -----

    def forward(
        self,
        input_codes: torch.Tensor,
        target_codes: torch.Tensor,
        prompts: Sequence[str],
    ) -> dict:
        """Teacher-forced forward. Returns a dict with:
            loss                : scalar CE on target-window positions.
            logits              : (B, n_q, T_in+T_out, V)   raw logits.
            target_start        : int (=T_in), index where the target
                                  window begins in the decoder stream.

        Notes on the transformers >= 5.x MusicGen contract:
        * `out.logits` has shape ``(B*n_q, T_total, V)`` — codebooks are
          unrolled into the batch dim.
        * Passing ``labels=`` triggers a buggy internal CE that crashes
          on a shape mismatch (1000 vs 4 in the unrolled batch). We
          therefore compute the masked LM loss ourselves and only
          consume the returned logits.
        """
        device = input_codes.device
        text_ids, text_mask = self._encode_prompts(prompts, device)
        dec_ids, _labels_unused = self._concat_audio_codes(input_codes, target_codes)
        B, n_q, T_total = dec_ids.shape
        T_in = input_codes.shape[-1]

        out = self.mg(
            input_ids=text_ids,
            attention_mask=text_mask,
            decoder_input_ids=dec_ids,
        )
        # Reshape (B*n_q, T_total, V) -> (B, n_q, T_total, V) for callers
        # that still expect the legacy 4-D layout (entropy probe, etc.).
        logits = out.logits
        V = logits.shape[-1]
        logits_4d = logits.view(B, n_q, T_total, V)

        # Standard LM shift: position p predicts dec_ids[:, :, p+1].
        # We only train on positions whose target token lies in the
        # target window, i.e. p in [T_in-1, T_total-1).
        preds = logits_4d[:, :, :-1, :]                # (B, n_q, T_total-1, V)
        tgts  = dec_ids[:, :, 1:].long()               # (B, n_q, T_total-1)
        mask  = torch.zeros_like(tgts, dtype=torch.bool)
        mask[:, :, T_in - 1:] = True

        ce = torch.nn.functional.cross_entropy(
            preds.reshape(-1, V),
            tgts.reshape(-1),
            reduction="none",
        ).reshape_as(tgts)
        denom = mask.float().sum().clamp_min(1.0)
        loss  = (ce * mask.float()).sum() / denom

        return {
            "loss": loss,
            "logits": logits_4d,
            "target_start": T_in,
        }

    # ----- inference helpers -----

    @torch.no_grad()
    def edit(
        self,
        input_codes: torch.Tensor,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 250,
    ) -> torch.Tensor:
        """Greedy/stochastic generation of a target token stream given
        an input-code prefix + text prompt. Returns ``(n_q, T_new)``.

        Intended for offline eval; streaming would build on
        ``MusicgenForConditionalGeneration.generate`` with a prefix.
        """
        device = input_codes.device
        if input_codes.dim() == 2:
            input_codes = input_codes.unsqueeze(0)
        B, n_q, T_in = input_codes.shape
        max_new_tokens = max_new_tokens or T_in
        text_ids, text_mask = self._encode_prompts([prompt], device)

        # HF wants (B*n_q, T)
        flat_codes = input_codes.reshape(B * n_q, T_in)
        gen = self.mg.generate(
            input_ids=text_ids,
            attention_mask=text_mask,
            decoder_input_ids=flat_codes,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=max(temperature, 1e-4),
            top_k=top_k,
        )
        new_codes = gen[:, T_in:].reshape(B, n_q, -1)
        return new_codes.squeeze(0)
