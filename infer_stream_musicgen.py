"""Offline inference for the MusicGen-LoRA editor (rescue-v3 Track B).

Batch-edits a directory of test WAVs at a fixed u, writing results to
an output directory. Produces the same layout (``<stem>.wav`` matching
the input's stem) that ``evaluate.py`` consumes.

Pipeline per clip:
    WAV in (24 kHz) -> resample 32 kHz -> Encodec-32k encode
                    -> prepend as decoder prefix with text prompt
                    -> greedy / top-k generate T_out frames
                    -> Encodec decode -> resample 24 kHz -> WAV out

Test-clip metadata in ``data/test_clips_v3/<stem>.json`` provides the
raga label; the ``--raga`` flag can override it. Fixed u passed via
``--u``.

Usage
-----
    python infer_stream_musicgen.py \\
        --ckpt_dir   checkpoints/editor_v3_lora \\
        --input_dir  data/test_clips_v3 \\
        --output_dir results/editor_v3_lora/u0.6 \\
        --u 0.6
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from models.musicgen_lora import (  # noqa: E402
    MusicGenEditor,
    inject_lora_into_musicgen,
    load_lora_state_dict,
    make_raga_u_prompt,
)


def _load_cfg(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "_base_" in cfg:
        base = _load_cfg(str(Path(path).parent / cfg["_base_"]))
        merged = dict(base)
        for k, v in cfg.items():
            if k == "_base_":
                continue
            if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                merged[k] = {**merged[k], **v}
            else:
                merged[k] = v
        cfg = merged
    return cfg


def _read_raga(wav_path: Path) -> str:
    """Read the raga from the sibling .json (test_clips_v3 layout)."""
    j = wav_path.with_suffix(".json")
    if j.exists():
        try:
            with open(j) as f:
                return str(json.load(f).get("raga", "UNK"))
        except Exception:
            pass
    return "UNK"


@torch.no_grad()
def _edit_one_wav(
    editor: MusicGenEditor,
    audio_encoder,
    wav_in: torch.Tensor,
    in_sr: int,
    out_sr: int,
    prompt: str,
    temperature: float,
    top_k: int,
    device: str,
) -> torch.Tensor:
    """Returns (1, T_samples) at ``out_sr`` (original rate)."""
    wav = wav_in
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    # Resample to Encodec's rate.
    enc_sr = int(audio_encoder.config.sampling_rate)
    if in_sr != enc_sr:
        wav = torchaudio.functional.resample(wav, in_sr, enc_sr)
    wav = wav.unsqueeze(0).to(device)              # (1, 1, T_samples)
    codes = audio_encoder.encode(wav).audio_codes  # (1, 1, n_q, T)
    codes = codes.squeeze(0).squeeze(0)            # (n_q, T)
    T_in = codes.shape[-1]
    out_codes = editor.edit(
        codes.to(device), prompt=prompt,
        max_new_tokens=T_in, temperature=temperature, top_k=top_k,
    )                                              # (n_q, T_new)
    # Decode output (prefix + generated) by concatenating generated
    # tokens and running audio_encoder.decode.
    full = torch.cat([codes, out_codes], dim=-1).unsqueeze(0).unsqueeze(0).long()
    wav_out = audio_encoder.decode(full, [None]).audio_values
    wav_out = wav_out.squeeze(0).squeeze(0)            # (T_samples,)
    # The editor-output portion starts at T_in * samples_per_frame.
    frames_per_sec = float(audio_encoder.config.frame_rate)
    start_samp = int(round((T_in / frames_per_sec) * enc_sr))
    wav_out = wav_out[start_samp:]
    # Resample back to evaluate.py's expected rate.
    wav_out = wav_out.unsqueeze(0)
    if enc_sr != out_sr:
        wav_out = torchaudio.functional.resample(wav_out, enc_sr, out_sr)
    return wav_out.cpu()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True,
                    help="Dir containing best/lora.pt (or a specific .pt).")
    ap.add_argument("--config", default=None,
                    help="Config used at training time. Defaults to "
                         "<ckpt_dir>/config.yaml.")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--u", type=float, default=0.6)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=250)
    ap.add_argument("--out_sr", type=int, default=24000,
                    help="Output sample rate (matches evaluate.py default).")
    ap.add_argument("--raga_override", default=None,
                    help="Force all clips to use this raga in the prompt.")
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir).resolve()
    cfg_path = args.config or str(ckpt_dir / "config.yaml")
    cfg = _load_cfg(cfg_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Reload MusicGen + inject LoRA with the same shape as training.
    from transformers import MusicgenForConditionalGeneration, AutoProcessor
    mg_name = cfg["musicgen"]["model_name"]
    print(f"[lora-mg-infer] loading {mg_name} ...")
    mg = MusicgenForConditionalGeneration.from_pretrained(mg_name).to(device)
    processor = AutoProcessor.from_pretrained(mg_name)
    lora_cfg = cfg["musicgen"].get("lora", {})
    inject_lora_into_musicgen(
        mg,
        r=int(lora_cfg.get("r", 16)),
        alpha=int(lora_cfg.get("alpha", 32)),
        dropout=float(lora_cfg.get("dropout", 0.0)),
        include_self_attn=bool(lora_cfg.get("include_self_attn", True)),
        include_cross_attn=bool(lora_cfg.get("include_cross_attn", True)),
    )
    # Load LoRA adapter weights.
    lora_ckpt = ckpt_dir / "best" / "lora.pt"
    if not lora_ckpt.exists():
        alt = ckpt_dir / "lora.pt"
        if alt.exists():
            lora_ckpt = alt
    print(f"[lora-mg-infer] loading LoRA from {lora_ckpt}")
    sd = torch.load(str(lora_ckpt), map_location=device, weights_only=False)
    sd = {k[3:] if k.startswith("mg.") else k: v for k, v in sd.items()}
    n = load_lora_state_dict(mg, sd)
    print(f"[lora-mg-infer] loaded {n} LoRA tensors.")
    if n == 0:
        raise RuntimeError(
            f"0 LoRA tensors loaded from {lora_ckpt}; "
            f"sample saved key: {next(iter(sd))!r}"
        )
    editor = MusicGenEditor(mg, processor).to(device).eval()

    in_dir = Path(args.input_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    wavs = sorted(in_dir.glob("*.wav"))
    print(f"[lora-mg-infer] editing {len(wavs)} clips at u={args.u}")
    for wav_path in wavs:
        raga = args.raga_override or _read_raga(wav_path)
        prompt = make_raga_u_prompt(raga, args.u)
        print(f"  {wav_path.name}  raga={raga!r}  prompt={prompt!r}")
        wav, in_sr = torchaudio.load(str(wav_path))
        wav_out = _edit_one_wav(
            editor, mg.audio_encoder, wav, in_sr, args.out_sr,
            prompt, args.temperature, args.top_k, device,
        )
        out_path = out_dir / wav_path.name
        sf.write(str(out_path), wav_out.squeeze(0).numpy(),
                 args.out_sr, subtype="PCM_16")
    print(f"[lora-mg-infer] done -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
