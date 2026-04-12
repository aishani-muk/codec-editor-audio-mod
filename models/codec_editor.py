"""
Conditional codec-to-codec transformer editor.

Takes input codec tokens (optionally BPE-compressed) + a stress-proxy embedding
and outputs edited codec tokens. Based on a GPT-2-style decoder-only transformer
with cross-attention to the conditioning signal.

Architecture:
    input_tokens + positional_embed + stress_embed → transformer layers → output_logits
"""

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel


class CodecEditor(nn.Module):
    """
    Conditional codec-to-codec editor built on GPT-2.

    The model receives:
        - input_ids: (B, T) token IDs from WavTokenizer (+ optional BPE)
        - stress_embeds: (B, T, embed_dim) from StressEmbedding
    and produces logits over the output token vocabulary.
    """

    def __init__(
        self,
        vocab_size: int = 4096,
        bpe_vocab_size: int | None = 8192,
        n_layers: int = 6,
        n_heads: int = 8,
        d_model: int = 512,
        d_ff: int = 2048,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        stress_embed_dim: int = 64,
    ):
        super().__init__()
        # Use BPE vocab if enabled, otherwise raw codec vocab
        effective_vocab = bpe_vocab_size if bpe_vocab_size else vocab_size

        self.config = GPT2Config(
            vocab_size=effective_vocab,
            n_positions=max_seq_len,
            n_embd=d_model,
            n_layer=n_layers,
            n_head=n_heads,
            n_inner=d_ff,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
        )
        self.transformer = GPT2LMHeadModel(self.config)

        # Project stress embedding to match model dimension
        self.stress_proj = nn.Linear(stress_embed_dim, d_model)

        # Learnable "task" embedding to distinguish input vs. target sections
        self.role_embed = nn.Embedding(2, d_model)  # 0=input, 1=target

    def forward(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor | None = None,
        stress_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> dict:
        """
        Forward pass for training (teacher-forced) or inference.

        For training: concatenate [input_ids | target_ids] and predict target_ids.
        For inference: pass input_ids and generate autoregressively.

        Args:
            input_ids: (B, T_in) source token IDs.
            target_ids: (B, T_out) target token IDs (training only).
            stress_embeds: (B, T_in, stress_embed_dim) conditioning.
            attention_mask: (B, T_total) optional mask.

        Returns:
            dict with 'loss' (if target_ids given) and 'logits'.
        """
        B, T_in = input_ids.shape
        device = input_ids.device

        if target_ids is not None:
            # Training: concat input + target
            T_out = target_ids.shape[1]
            full_ids = torch.cat([input_ids, target_ids], dim=1)  # (B, T_in+T_out)

            # Role embeddings
            role_ids = torch.cat([
                torch.zeros(B, T_in, dtype=torch.long, device=device),
                torch.ones(B, T_out, dtype=torch.long, device=device),
            ], dim=1)
            role_emb = self.role_embed(role_ids)

            # Stress conditioning (zero-padded for target portion)
            if stress_embeds is not None:
                stress_cond = self.stress_proj(stress_embeds)  # (B, T_in, d_model)
                pad = torch.zeros(B, T_out, stress_cond.shape[-1],
                                  device=device, dtype=stress_cond.dtype)
                stress_cond = torch.cat([stress_cond, pad], dim=1)
            else:
                stress_cond = 0.0

            # Get token embeddings and add conditioning
            inputs_embeds = (
                self.transformer.transformer.wte(full_ids)
                + role_emb
                + stress_cond
            )

            # Labels: -100 for input portion (don't compute loss), target_ids for rest
            labels = torch.cat([
                torch.full((B, T_in), -100, dtype=torch.long, device=device),
                target_ids,
            ], dim=1)

            outputs = self.transformer(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )
            return {"loss": outputs.loss, "logits": outputs.logits}

        else:
            # Inference: just encode the input portion
            role_emb = self.role_embed(
                torch.zeros(B, T_in, dtype=torch.long, device=device)
            )
            if stress_embeds is not None:
                stress_cond = self.stress_proj(stress_embeds)
            else:
                stress_cond = 0.0

            inputs_embeds = (
                self.transformer.transformer.wte(input_ids)
                + role_emb
                + stress_cond
            )
            outputs = self.transformer(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )
            return {"logits": outputs.logits}

    @torch.no_grad()
    def generate_edited(
        self,
        input_ids: torch.Tensor,
        stress_embeds: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 40,
    ) -> torch.Tensor:
        """
        Autoregressively generate edited tokens given input tokens + stress.

        Args:
            input_ids: (B, T_in) source tokens.
            stress_embeds: (B, T_in, stress_embed_dim).
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k sampling.

        Returns:
            (B, T_out) generated edited token IDs.
        """
        B, T_in = input_ids.shape
        device = input_ids.device

        # Build initial input embeddings
        role_emb_in = self.role_embed(
            torch.zeros(B, T_in, dtype=torch.long, device=device)
        )
        stress_cond = self.stress_proj(stress_embeds)
        input_embeds = (
            self.transformer.transformer.wte(input_ids)
            + role_emb_in
            + stress_cond
        )

        # Start with a BOS-like token (use token 0 as start)
        generated = torch.zeros(B, 1, dtype=torch.long, device=device)
        all_embeds = input_embeds

        for _ in range(max_new_tokens):
            # Embed the latest generated token with role=1 (target)
            new_emb = (
                self.transformer.transformer.wte(generated[:, -1:])
                + self.role_embed(torch.ones(B, 1, dtype=torch.long, device=device))
            )
            all_embeds = torch.cat([all_embeds, new_emb], dim=1)

            outputs = self.transformer(inputs_embeds=all_embeds)
            next_logits = outputs.logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                topk_vals, _ = torch.topk(next_logits, top_k)
                threshold = topk_vals[:, -1].unsqueeze(-1)
                next_logits[next_logits < threshold] = float("-inf")

            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

        return generated[:, 1:]  # Remove the start token
