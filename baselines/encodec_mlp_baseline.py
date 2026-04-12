"""
EnCodec + MLP baseline: a simpler codec-editing approach.

Uses EnCodec (multi-codebook, higher token rate) instead of WavTokenizer,
and a shallow MLP instead of a transformer for token editing.

This ablates:
  1. Whether WavTokenizer's extreme compression is necessary
  2. Whether the transformer editor outperforms a simple feed-forward network

Usage:
    python baselines/encodec_mlp_baseline.py \
        --input data/saraga_yaman/ \
        --output results/baseline_encodec_mlp/ \
        --u 0.6
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
from tqdm import tqdm


class MLPTokenEditor(nn.Module):
    """
    Simple MLP that maps input codec tokens → edited tokens,
    conditioned on a scalar edit intensity u.
    """

    def __init__(self, codebook_size: int = 1024, n_codebooks: int = 4,
                 hidden_dim: int = 256):
        super().__init__()
        self.codebook_size = codebook_size
        self.n_codebooks = n_codebooks

        # Embed each codebook token
        self.embed = nn.Embedding(codebook_size, hidden_dim)

        # Conditioning: scalar u → hidden
        self.cond_proj = nn.Linear(1, hidden_dim)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * n_codebooks + hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim * n_codebooks),
        )

        # Output heads: one per codebook
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, codebook_size) for _ in range(n_codebooks)
        ])

    def forward(self, tokens: torch.Tensor, u: torch.Tensor):
        """
        Args:
            tokens: (B, n_codebooks, T) input token IDs.
            u: (B, 1) edit intensity.
        Returns:
            logits: (B, n_codebooks, T, codebook_size).
        """
        B, Q, T = tokens.shape

        # Embed all codebooks
        embs = []
        for q in range(Q):
            embs.append(self.embed(tokens[:, q, :]))  # (B, T, hidden)
        emb_cat = torch.cat(embs, dim=-1)  # (B, T, hidden * Q)

        # Condition on u
        u_emb = self.cond_proj(u.unsqueeze(-1))  # (B, 1, hidden)
        u_emb = u_emb.expand(-1, T, -1)  # (B, T, hidden)

        x = torch.cat([emb_cat, u_emb], dim=-1)  # (B, T, hidden*(Q+1))
        h = self.mlp(x)  # (B, T, hidden * Q)

        # Split and project to logits
        h_split = h.chunk(Q, dim=-1)
        logits = torch.stack([
            self.heads[q](h_split[q]) for q in range(Q)
        ], dim=1)  # (B, Q, T, codebook_size)

        return logits


def run_baseline(input_dir: str, output_dir: str, u: float = 0.6,
                 sr: int = 24000, device: str = "cuda"):
    """
    Run the EnCodec + MLP baseline.

    NOTE: This requires a pretrained MLP model. For initial evaluation,
    the MLP is randomly initialized (untrained baseline) to establish
    a minimum performance floor. Training the MLP uses the same paired
    data as the proposed pipeline.
    """
    from transformers import EncodecModel, AutoProcessor

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    device = device if torch.cuda.is_available() else "cpu"

    # Load EnCodec
    encodec = EncodecModel.from_pretrained("facebook/encodec_24khz").to(device)
    processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
    encodec.eval()

    # Initialize MLP editor (random weights = untrained baseline)
    editor = MLPTokenEditor(
        codebook_size=encodec.config.codebook_size,
        n_codebooks=4,
    ).to(device)
    editor.eval()

    audio_files = sorted(Path(input_dir).glob("**/*.wav"))
    if not audio_files:
        audio_files = sorted(Path(input_dir).glob("**/*.mp3"))

    print(f"EnCodec+MLP baseline: processing {len(audio_files)} files at u={u}")

    for path in tqdm(audio_files, desc="EnCodec+MLP"):
        wav, wav_sr = torchaudio.load(str(path))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if wav_sr != sr:
            wav = torchaudio.functional.resample(wav, wav_sr, sr)

        inputs = processor(
            raw_audio=wav.squeeze().numpy(),
            sampling_rate=sr,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            encoded = encodec.encode(**inputs, bandwidth=6.0)
            codes = encoded.audio_codes[0, 0]  # (Q, T)

            # Run MLP editor
            u_tensor = torch.tensor([[u]], device=device, dtype=torch.float32)
            logits = editor(codes.unsqueeze(0), u_tensor)
            edited_codes = logits.argmax(dim=-1)  # (1, Q, T)

            # Decode
            audio_out = encodec.decode(
                edited_codes.unsqueeze(0), [None]
            ).audio_values[0, 0]

        out_path = Path(output_dir) / (path.stem + "_encodec_mlp.wav")
        sf.write(str(out_path), audio_out.cpu().numpy(), sr)

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EnCodec + MLP baseline")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--u", type=float, default=0.6)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_baseline(args.input, args.output, args.u, device=args.device)
