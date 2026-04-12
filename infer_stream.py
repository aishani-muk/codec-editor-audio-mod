"""
End-to-end streaming inference pipeline.

  WAV in → WavTokenizer encode → (BPE compress) → stress embed
         → codec-to-codec transformer edit (windowed)
         → (BPE decompress) → WavTokenizer decode → overlap-add → WAV out

This is the core runtime loop that would run in real time with a live
stress proxy.  For offline evaluation it processes a file in one pass.

Usage:
    python infer_stream.py \
        --input  data/saraga_yaman/YMN-01.wav \
        --output results/proposed_v1/YMN-01_modulated.wav \
        --checkpoint checkpoints/proposed_v1/best/ \
        --config configs/proposed.yaml \
        --u 0.6

    # With WESAD-trained stress proxy (variable u over time):
    python infer_stream.py \
        --input  data/saraga_yaman/YMN-01.wav \
        --output results/proposed_v1/YMN-01_modulated.wav \
        --checkpoint checkpoints/proposed_v1/best/ \
        --config configs/proposed.yaml \
        --stress_profile pulse --peak 0.8
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio
import soundfile as sf
import yaml

from models.codec_editor import CodecEditor
from models.stress_proxy import (
    SyntheticStressProxy,
    StressEmbedding,
    stress_to_edit_intensity,
)
from models.overlap_add import StreamingOverlapAdd, overlap_add_waveform


# ─────────────────────────────────────────────────────────────
# WavTokenizer wrapper
# ─────────────────────────────────────────────────────────────

class WavTokenizerWrapper:
    """Unified interface around WavTokenizer encode / decode."""

    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = device
        self.sr = 24000

        try:
            from wavtokenizer import WavTokenizer
            self.model = WavTokenizer.from_pretrained(model_name).to(device).eval()
            self._backend = "package"
        except ImportError:
            import sys
            sys.path.insert(0, str(Path(__file__).parent / "third_party" / "WavTokenizer"))
            from decoder.pretrained import WavTokenizer as WT
            from huggingface_hub import hf_hub_download
            cfg = hf_hub_download(repo_id=model_name, filename="config.yaml")
            ckpt = hf_hub_download(repo_id=model_name, filename="model.ckpt")
            self.model = WT.from_pretrained0802(cfg, ckpt).to(device).eval()
            self._backend = "repo"

    @torch.no_grad()
    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (1, T_samples) → codes: (1, T_tokens) int64"""
        wav = wav.to(self.device)
        bid = torch.tensor([0], device=self.device)
        _, codes = self.model.encode_infer(wav, bandwidth_id=bid)
        return codes.squeeze(0) if codes.dim() == 3 else codes  # (n_q, T)

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: (n_q, T_tokens) → wav: (1, T_samples)"""
        codes = codes.to(self.device)
        if codes.dim() == 1:
            codes = codes.unsqueeze(0)
        feats = self.model.codes_to_features(codes)
        bid = torch.tensor([0], device=self.device)
        wav = self.model.decode(feats, bandwidth_id=bid)
        return wav  # (1, T_samples)


# ─────────────────────────────────────────────────────────────
# BPE wrapper (optional)
# ─────────────────────────────────────────────────────────────

class BPEWrapper:
    """Compress / decompress codec tokens via codec-BPE."""

    def __init__(self, bpe_model_dir: str, codebook_size: int = 4096):
        from transformers import AutoTokenizer
        from codec_bpe import codes_to_chars, chars_to_codes
        self.tokenizer = AutoTokenizer.from_pretrained(bpe_model_dir)
        self.codebook_size = codebook_size
        self._to_chars = codes_to_chars
        self._to_codes = chars_to_codes

    def compress(self, codes: np.ndarray) -> np.ndarray:
        """codes: (n_q, T) int → bpe_ids: (K,) int64"""
        codes_t = torch.from_numpy(codes)
        ustr = self._to_chars(codes_t, codebook_size=self.codebook_size)
        ids = self.tokenizer.encode(ustr)
        return np.array(ids, dtype=np.int64)

    def decompress(self, bpe_ids: np.ndarray, n_codebooks: int = 1) -> np.ndarray:
        """bpe_ids: (K,) int → codes: (n_q, T) int"""
        ustr = self.tokenizer.decode(bpe_ids.tolist(), skip_special_tokens=False)
        codes = self._to_codes(
            ustr, num_codebooks=n_codebooks,
            codebook_size=self.codebook_size, return_tensors="np",
        )
        return codes


# ─────────────────────────────────────────────────────────────
# Main streaming inference
# ─────────────────────────────────────────────────────────────

def run_streaming_inference(
    input_wav: str,
    output_wav: str,
    checkpoint_dir: str,
    config_path: str,
    u_fixed: float | None = None,
    stress_profile: str = "ramp",
    peak: float = 0.6,
    device: str = "cuda",
):
    """
    Full pipeline: load audio → tokenize → window → edit → decode → save.
    """
    # ── Load config ──
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if "_base_" in cfg:
        base_path = Path(config_path).parent / cfg["_base_"]
        with open(base_path) as f:
            base_cfg = yaml.safe_load(f)
        base_cfg.update({k: v for k, v in cfg.items() if k != "_base_"})
        cfg = base_cfg

    device = device if torch.cuda.is_available() else "cpu"
    sr = cfg["sample_rate"]
    tok_rate = cfg["wavtokenizer"]["tokens_per_sec"]
    bpe_enabled = cfg["bpe"]["enabled"]
    ed_cfg = cfg["editor"]

    print(f"Device: {device}")
    print(f"BPE: {'ON' if bpe_enabled else 'OFF'}")

    # ── Load audio ──
    wav, orig_sr = torchaudio.load(input_wav)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)
    duration_sec = wav.shape[1] / sr
    print(f"Input: {input_wav}  ({duration_sec:.1f}s, {sr} Hz)")

    # ── Tokenize ──
    print("Encoding with WavTokenizer...")
    wtok = WavTokenizerWrapper(cfg["wavtokenizer"]["model_name"], device)
    codes = wtok.encode(wav)  # (n_q, T_tok)
    codes_np = codes.cpu().numpy()
    T_tok = codes_np.shape[-1]
    print(f"  {T_tok} tokens ({T_tok / duration_sec:.1f} tok/s)")

    # ── BPE compress (optional) ──
    if bpe_enabled:
        print("Compressing with codec-BPE...")
        bpe = BPEWrapper(cfg["bpe"]["model_dir"], cfg["wavtokenizer"]["codebook_size"])
        bpe_ids = bpe.compress(codes_np)
        T_bpe = len(bpe_ids)
        print(f"  {T_tok} → {T_bpe} tokens (ratio {T_bpe/T_tok:.2f})")
        edit_tokens = bpe_ids
    else:
        edit_tokens = codes_np.flatten()  # (T_tok,) for single codebook

    # ── Generate stress trajectory ──
    token_len = len(edit_tokens)
    if u_fixed is not None:
        u_arr = np.full(token_len, u_fixed, dtype=np.float32)
        print(f"Stress proxy: fixed u={u_fixed}")
    else:
        proxy = SyntheticStressProxy(token_rate=tok_rate)
        s_arr = proxy.generate(
            duration_sec, peak=peak,
            onset_sec=cfg["stress_proxy"]["onset_sec"],
            ramp_sec=cfg["stress_proxy"]["ramp_sec"],
            profile=stress_profile,
        )
        # Resample to match edit_tokens length if BPE changed it
        if len(s_arr) != token_len:
            s_arr = np.interp(
                np.linspace(0, 1, token_len),
                np.linspace(0, 1, len(s_arr)),
                s_arr,
            )
        u_arr = stress_to_edit_intensity(s_arr)
        print(f"Stress proxy: profile={stress_profile}, peak={peak}")

    # ── Load editor model ──
    print("Loading editor checkpoint...")
    ckpt_dir = Path(checkpoint_dir)
    model = CodecEditor(
        vocab_size=cfg["wavtokenizer"]["codebook_size"],
        bpe_vocab_size=cfg["bpe"]["vocab_size"] if bpe_enabled else None,
        n_layers=ed_cfg["n_layers"],
        n_heads=ed_cfg["n_heads"],
        d_model=ed_cfg["d_model"],
        d_ff=ed_cfg["d_ff"],
        max_seq_len=ed_cfg["max_seq_len"],
        dropout=0.0,
        stress_embed_dim=cfg["stress_proxy"]["embed_dim"],
    ).to(device)
    model.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location=device))
    model.eval()

    stress_embed = StressEmbedding(
        embed_dim=cfg["stress_proxy"]["embed_dim"]
    ).to(device)
    stress_embed.load_state_dict(
        torch.load(ckpt_dir / "stress_embed.pt", map_location=device)
    )
    stress_embed.eval()
    print(f"  Loaded from {ckpt_dir}")

    # ── Windowed editing ──
    win_tok = int(ed_cfg["window_sec"] * tok_rate)
    hop_tok = int(ed_cfg["hop_sec"] * tok_rate)
    if bpe_enabled:
        # BPE changes the effective token rate; scale window accordingly
        ratio = token_len / T_tok
        win_tok = max(1, int(win_tok * ratio))
        hop_tok = max(1, int(hop_tok * ratio))

    ola = StreamingOverlapAdd(
        window_tokens=win_tok,
        hop_tokens=hop_tok,
        crossfade_tokens=max(1, int(ed_cfg.get("crossfade_sec", 0.1) * tok_rate)),
    )

    windows = ola.segment(edit_tokens)
    print(f"Processing {len(windows)} windows (L={win_tok}, H={hop_tok})...")

    edited_windows = []
    for start, win_tokens in windows:
        win_u = u_arr[start : start + len(win_tokens)]

        inp = torch.from_numpy(win_tokens).unsqueeze(0).long().to(device)
        u_t = torch.from_numpy(win_u).unsqueeze(0).float().to(device)
        se = stress_embed(u_t)

        with torch.no_grad():
            out = model.generate_edited(
                inp, se, max_new_tokens=len(win_tokens),
                temperature=0.85, top_k=40,
            )
        edited_windows.append((start, out[0].cpu().numpy()))

    # ── Merge tokens ──
    merged = ola.merge(edited_windows, token_len)
    print(f"Merged {len(merged)} edited tokens")

    # ── BPE decompress ──
    if bpe_enabled:
        print("Decompressing BPE...")
        decoded_codes = bpe.decompress(
            merged, n_codebooks=cfg["wavtokenizer"]["n_codebooks"]
        )
    else:
        decoded_codes = merged.reshape(cfg["wavtokenizer"]["n_codebooks"], -1)

    # ── Decode to waveform ──
    print("Decoding with WavTokenizer...")
    codes_t = torch.from_numpy(decoded_codes).long()
    wav_out = wtok.decode(codes_t)  # (1, T_samples)
    wav_np = wav_out.squeeze().cpu().numpy()

    # ── Save ──
    Path(output_wav).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_wav, wav_np, sr)
    print(f"Saved: {output_wav}  ({len(wav_np)/sr:.1f}s)")


# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Streaming codec-to-codec inference pipeline"
    )
    parser.add_argument("--input", required=True, help="Input WAV file")
    parser.add_argument("--output", required=True, help="Output WAV file")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory")
    parser.add_argument("--config", required=True, help="Config YAML")
    parser.add_argument("--u", type=float, default=None,
                        help="Fixed edit intensity (overrides stress profile)")
    parser.add_argument("--stress_profile", default="ramp",
                        choices=["ramp", "pulse", "episodic"])
    parser.add_argument("--peak", type=float, default=0.6)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_streaming_inference(
        input_wav=args.input,
        output_wav=args.output,
        checkpoint_dir=args.checkpoint,
        config_path=args.config,
        u_fixed=args.u,
        stress_profile=args.stress_profile,
        peak=args.peak,
        device=args.device,
    )
