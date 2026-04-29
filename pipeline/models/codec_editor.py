"""
Conditional codec-to-codec transformer editor.

Architecture (input-aligned residual; May-2026 redesign):

    The decoder-only transformer sees the concatenation [input | target]. For
    every target position t we ADDITIONALLY add the time-aligned input-token
    embedding ``wte(input[clamp(t, T_in-1)])`` to the target-embedding stream.

    This gives the model direct positional access to the source token it is
    editing, so the λ=0 "identity edit" case is trivially solvable and the
    network only has to learn the *residual* for larger λ. Empirically this
    is a much better inductive bias for codec-to-codec editing than the
    original "look back across the T_in boundary via self-attention" scheme
    (see docs/redesign_apr2026.md).

    A learned raga-label embedding (one per clip, broadcast over every
    position) and the stress-proxy embedding (per input position) are added
    to both the input and target embedding streams.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel


class CodecEditor(nn.Module):
    """Conditional codec-to-codec editor built on GPT-2."""

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
        n_ragas: int = 0,
        use_input_residual: bool = True,
        conditioning: str = "add",
    ):
        super().__init__()

        # Effective vocab: BPE output if used, else raw codec vocab.
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

        # Projection for stress-proxy vector (per input position).
        self.stress_proj = nn.Linear(stress_embed_dim, d_model)

        # 0 = input role, 1 = target role.
        self.role_embed = nn.Embedding(2, d_model)

        # Optional raga conditioning. 0 is reserved for UNK/no-raga-available.
        self.n_ragas = n_ragas
        if n_ragas > 0:
            self.raga_embed = nn.Embedding(n_ragas + 1, d_model)
        else:
            self.raga_embed = None

        self.use_input_residual = use_input_residual

        # Conditioning mode:
        #   "add"   — the original path: cond is added ONCE to the input token
        #             embeddings before the transformer.
        #   "adaln" — DiT-style per-block additive modulation. Every GPT-2
        #             block gets a zero-initialised linear projection of the
        #             conditioning vector added to its OUTPUT (post-MLP +
        #             residual). Zero-init means step-0 behaviour matches the
        #             uncondditioned pretrain warm-start exactly, then the
        #             model is free to learn per-block conditioning strength.
        #             Implementation uses forward-hooks on each GPT2Block so
        #             the rest of HF's attention machinery, KV cache, and
        #             masking are untouched.
        if conditioning not in ("add", "adaln"):
            raise ValueError(
                f"conditioning must be 'add' or 'adaln', got {conditioning!r}"
            )
        self.conditioning = conditioning
        self._adaln_current_cond: torch.Tensor | None = None  # set per fwd

        if conditioning == "adaln":
            # One projection per transformer block, cond_dim == d_model.
            self.adaln_block_projs = nn.ModuleList([
                nn.Linear(d_model, d_model) for _ in range(n_layers)
            ])
            # Zero-init ensures step-0 output == "add" mode with the SAME
            # cond contribution at the embedding level; DiT-Zero prescription.
            for proj in self.adaln_block_projs:
                nn.init.zeros_(proj.weight)
                nn.init.zeros_(proj.bias)

            # Attach forward hooks to each GPT-2 block. The hook fires on
            # block output; we slice the cond vector to the block's actual
            # output sequence length (handles KV-cache 1-token generation).
            for i, block in enumerate(self.transformer.transformer.h):
                proj = self.adaln_block_projs[i]
                block.register_forward_hook(self._make_adaln_hook(proj))

    @staticmethod
    def _make_adaln_hook(proj: nn.Linear):
        """Factory: returns a forward-hook closure that adds
        ``proj(self._adaln_current_cond[:, -T_out:, :])`` to the block's
        output hidden_states. Returns None if no cond is active (safe
        no-op when forward is called without AdaLN context)."""
        def _hook(module, inputs, output):
            # GPT2Block returns a tuple; element 0 is hidden_states.
            # Some HF versions emit just the tensor — handle both.
            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
            else:
                hidden = output
                rest = None

            # The hook needs to pull the cond tensor from the enclosing
            # CodecEditor instance. We stash a reference on the hook itself.
            cond = _hook._cond_ref  # type: ignore[attr-defined]
            if cond is None:
                return output  # no-op
            # Broadcast the cond across whatever length the block emitted
            # (may be less than full T_total during KV-cache generation).
            T_out = hidden.size(1)
            T_c = cond.size(1)
            if T_c >= T_out:
                cond_slice = cond[:, -T_out:, :]
            else:
                # Last cond frame repeated to fill the output length.
                pad = cond[:, -1:, :].expand(-1, T_out - T_c, -1)
                cond_slice = torch.cat([cond, pad], dim=1)[:, -T_out:, :]
            # Cast to the block output's dtype (AMP safety).
            add = proj(cond_slice.to(hidden.dtype))
            new_hidden = hidden + add
            if rest is not None:
                return (new_hidden,) + rest
            return new_hidden

        _hook._cond_ref = None  # type: ignore[attr-defined]
        return _hook

    def _set_adaln_cond(self, cond: torch.Tensor | None) -> None:
        """Update the cond tensor visible to every block hook. Safe to
        call at the start of every forward; pass ``None`` to disable."""
        self._adaln_current_cond = cond
        if self.conditioning != "adaln":
            return
        for block in self.transformer.transformer.h:
            for h in block._forward_hooks.values():
                h._cond_ref = cond  # type: ignore[attr-defined]

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _target_aligned_input_emb(
        self,
        input_ids: torch.Tensor,
        t_out: int,
    ) -> torch.Tensor:
        """For each target position t in [0, t_out), return
        ``wte(input[:, min(t, T_in-1)])`` so the last input token is reused
        whenever the target overshoots the input length."""
        t_in = input_ids.size(1)
        if t_in == 0:
            B = input_ids.size(0)
            D = self.transformer.transformer.wte.embedding_dim
            return torch.zeros(B, t_out, D,
                               device=input_ids.device,
                               dtype=self.transformer.transformer.wte.weight.dtype)
        idx = torch.arange(t_out, device=input_ids.device).clamp(max=t_in - 1)
        aligned_ids = input_ids[:, idx]                          # (B, t_out)
        return self.transformer.transformer.wte(aligned_ids)     # (B, t_out, D)

    def _conditioning_embedding(
        self,
        B: int,
        T_total: int,
        T_in: int,
        device: torch.device,
        stress_embeds: torch.Tensor | None,
        raga_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        """Build role + stress + raga conditioning at every position."""
        # Role.
        role_ids = torch.zeros(B, T_total, dtype=torch.long, device=device)
        role_ids[:, T_in:] = 1
        cond = self.role_embed(role_ids)

        # Stress (broadcast across all positions).
        if stress_embeds is not None:
            # Pad to full length if stress is only given over the input span.
            if stress_embeds.size(1) < T_total:
                # Repeat the last frame (u is constant per clip anyway).
                pad_n = T_total - stress_embeds.size(1)
                pad = stress_embeds[:, -1:, :].expand(B, pad_n, -1)
                stress_full = torch.cat([stress_embeds, pad], dim=1)
            else:
                stress_full = stress_embeds[:, :T_total, :]
            cond = cond + self.stress_proj(stress_full)

        # Raga (one vector per clip, broadcast).
        if raga_ids is not None and self.raga_embed is not None:
            raga_vec = self.raga_embed(raga_ids).unsqueeze(1)    # (B, 1, D)
            cond = cond + raga_vec

        return cond

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor | None = None,
        stress_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        raga_ids: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> dict:
        """Training (teacher-forced) or inference forward pass.

        Args:
            input_ids:      (B, T_in) source tokens.
            target_ids:     (B, T_out) target tokens (training only).
            stress_embeds:  (B, T_in, stress_embed_dim) or (B, T_total, ...).
            attention_mask: (B, T_total) 1=valid, 0=pad. Used by GPT-2 attention.
            raga_ids:       (B,) long tensor of raga indices (0=UNK) or None.
            target_mask:    (B, T_out) 1=real target token, 0=pad. Padded
                            positions are excluded from the LM loss via labels.
        """
        B, T_in = input_ids.shape
        device = input_ids.device
        wte = self.transformer.transformer.wte

        if target_ids is not None:
            # ── Training path ───────────────────────────────────────────────
            T_out = target_ids.size(1)
            T_total = T_in + T_out

            tok_emb = wte(torch.cat([input_ids, target_ids], dim=1))

            # Input-aligned residual on the target portion.
            if self.use_input_residual:
                aligned = self._target_aligned_input_emb(input_ids, T_out)
                tok_emb = torch.cat(
                    [tok_emb[:, :T_in, :], tok_emb[:, T_in:, :] + aligned],
                    dim=1,
                )

            cond = self._conditioning_embedding(
                B, T_total, T_in, device, stress_embeds, raga_ids
            )
            # AdaLN mode also uses the additive cond at the embedding level
            # (so a zero-init hook reproduces "add" mode exactly at step 0)
            # AND injects the per-block modulation on top.
            inputs_embeds = tok_emb + cond
            self._set_adaln_cond(cond)
            try:
                # Labels: -100 for input + padded target positions.
                labels = torch.cat(
                    [torch.full((B, T_in), -100, dtype=torch.long, device=device),
                     target_ids.clone()],
                    dim=1,
                )
                if target_mask is not None:
                    labels[:, T_in:][target_mask == 0] = -100

                outputs = self.transformer(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            finally:
                self._set_adaln_cond(None)
            return {"loss": outputs.loss, "logits": outputs.logits}

        # ── Inference path (encode-only) ────────────────────────────────────
        T_total = T_in
        tok_emb = wte(input_ids)
        cond = self._conditioning_embedding(
            B, T_total, T_in, device, stress_embeds, raga_ids
        )
        self._set_adaln_cond(cond)
        try:
            outputs = self.transformer(
                inputs_embeds=tok_emb + cond,
                attention_mask=attention_mask,
            )
        finally:
            self._set_adaln_cond(None)
        return {"logits": outputs.logits}

    # -------------------------------------------------------------------------
    # Generation
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def generate_edited(
        self,
        input_ids: torch.Tensor,
        stress_embeds: torch.Tensor,
        raga_ids: torch.Tensor | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 40,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """Autoregressively produce target tokens given the input.

        Decoding mirrors the training-time layout exactly:

          1. The input prefix is embedded with role=0 conditioning and a
             single forward pass is run; ``logits[:, -1, :]`` naturally
             predicts ``target[0]`` — same shift convention as training.
          2. For ``t >= 1`` we embed the just-sampled ``target[t-1]``,
             add the input-aligned residual ``wte(input[min(t-1, T_in-1)])``
             plus role=1 / stress / raga conditioning, and advance the
             cached KV state one step at a time. No ``prev_token = 0``
             sentinel; the model is never fed a token-ID it didn't see
             during training.

        Uses GPT-2's ``past_key_values`` so cost is O(T) total (one kv
        update per new token) instead of the O(T^2) re-concat loop.
        """
        B, T_in = input_ids.shape
        device = input_ids.device
        wte = self.transformer.transformer.wte

        # ── Helpers ─────────────────────────────────────────────────────
        def _sample(logits_step: torch.Tensor) -> torch.Tensor:
            logits_step = logits_step / max(temperature, 1e-6)
            if top_k > 0:
                topk_vals, _ = torch.topk(logits_step, top_k)
                threshold = topk_vals[:, -1].unsqueeze(-1)
                logits_step = logits_step.masked_fill(
                    logits_step < threshold, float("-inf")
                )
            probs = torch.softmax(logits_step, dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(1)

        def _raga_vec() -> torch.Tensor | None:
            if raga_ids is None or self.raga_embed is None:
                return None
            return self.raga_embed(raga_ids).unsqueeze(1)  # (B, 1, D)

        # Classifier-free-guidance support: when ``cfg_scale > 1`` we run a
        # SECOND parallel AR stream with zeroed conditioning (u=0, raga=UNK)
        # and combine logits every step as
        #   logits = logits_uncond + cfg_scale · (logits_cond - logits_uncond)
        # which is the standard CFG sampling formula. Both streams share
        # input_ids / tok_emb / input-residual (the unconditioning only
        # touches the stress + raga + AdaLN contributions).
        use_cfg = cfg_scale > 1.0 + 1e-6
        if use_cfg:
            stress_uncond = (torch.zeros_like(stress_embeds)
                             if stress_embeds is not None else None)
            raga_ids_uncond = (torch.zeros_like(raga_ids)
                               if raga_ids is not None else None)

        def _build_cond(stress, raga_vec, T):
            """Assemble role + stress + raga conditioning for a given T."""
            role = self.role_embed(
                torch.zeros(B, T, dtype=torch.long, device=device)
            )
            c = role
            if stress is not None:
                s = stress[:, :T, :] if stress.size(1) >= T else stress
                c = c + self.stress_proj(s)
            if raga_vec is not None:
                c = c + raga_vec
            return c

        def _raga_vec_from(raga_ids_arg):
            if raga_ids_arg is None or self.raga_embed is None:
                return None
            return self.raga_embed(raga_ids_arg).unsqueeze(1)

        # ── 1. Prefix forward (input tokens, role=0). ───────────────────
        input_prefix = wte(input_ids)
        raga_vec = _raga_vec_from(raga_ids)
        cond_input = _build_cond(stress_embeds, raga_vec, T_in)

        self._set_adaln_cond(cond_input)
        try:
            out_c = self.transformer(
                inputs_embeds=input_prefix + cond_input, use_cache=True,
            )
        finally:
            self._set_adaln_cond(None)
        past_kv_c = out_c.past_key_values
        first_logits_c = out_c.logits[:, -1, :]

        if use_cfg:
            raga_vec_u = _raga_vec_from(raga_ids_uncond)
            cond_input_u = _build_cond(stress_uncond, raga_vec_u, T_in)
            self._set_adaln_cond(cond_input_u)
            try:
                out_u = self.transformer(
                    inputs_embeds=input_prefix + cond_input_u, use_cache=True,
                )
            finally:
                self._set_adaln_cond(None)
            past_kv_u = out_u.past_key_values
            first_logits_u = out_u.logits[:, -1, :]
            first_logits = first_logits_u + cfg_scale * (
                first_logits_c - first_logits_u
            )
        else:
            past_kv_u = None
            first_logits = first_logits_c

        prev_token = _sample(first_logits)
        generated = [prev_token.unsqueeze(1)]

        # ── 2. Autoregressive target loop with KV-cache. ────────────────
        role_tgt = self.role_embed(
            torch.ones(B, 1, dtype=torch.long, device=device)
        )

        def _step_cond(stress, raga_vec_step, t):
            c = role_tgt
            if stress is not None:
                s_idx = min(t, stress.size(1) - 1)
                c = c + self.stress_proj(stress[:, s_idx:s_idx + 1, :])
            if raga_vec_step is not None:
                c = c + raga_vec_step
            return c

        raga_vec_u = _raga_vec_from(raga_ids_uncond) if use_cfg else None

        for t in range(1, max_new_tokens):
            src_idx = min(t - 1, T_in - 1)
            tok_vec = wte(prev_token).unsqueeze(1)
            if self.use_input_residual:
                tok_vec = tok_vec + wte(input_ids[:, src_idx:src_idx + 1])

            cond_t = _step_cond(stress_embeds, raga_vec, t)
            self._set_adaln_cond(cond_t)
            try:
                out_c = self.transformer(
                    inputs_embeds=tok_vec + cond_t,
                    past_key_values=past_kv_c,
                    use_cache=True,
                )
            finally:
                self._set_adaln_cond(None)
            past_kv_c = out_c.past_key_values
            logits_c = out_c.logits[:, -1, :]

            if use_cfg:
                cond_t_u = _step_cond(stress_uncond, raga_vec_u, t)
                self._set_adaln_cond(cond_t_u)
                try:
                    out_u = self.transformer(
                        inputs_embeds=tok_vec + cond_t_u,
                        past_key_values=past_kv_u,
                        use_cache=True,
                    )
                finally:
                    self._set_adaln_cond(None)
                past_kv_u = out_u.past_key_values
                logits_u = out_u.logits[:, -1, :]
                step_logits = logits_u + cfg_scale * (logits_c - logits_u)
            else:
                step_logits = logits_c

            prev_token = _sample(step_logits)
            generated.append(prev_token.unsqueeze(1))

        return torch.cat(generated, dim=1)
