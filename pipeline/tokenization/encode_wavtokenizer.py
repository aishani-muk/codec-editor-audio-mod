"""
Encode WAV files into WavTokenizer discrete codes.

Usage:
    python tokenization/encode_wavtokenizer.py \
        --input data/saraga_kalyan_thaat/ \
        --output data/tokens/wavtok/ \
        --model novateur/WavTokenizer-large-unify-40token \
        --batch_size 4
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm import tqdm


_HF_CKPT_FILENAMES = {
    "novateur/WavTokenizer-large-unify-40token":
        "wavtokenizer_large_unify_600_24k.ckpt",
    "novateur/WavTokenizer-large-speech-75token":
        "wavtokenizer_large_speech_320_v2.ckpt",
}

_LOCAL_CONFIG_FOR_MODEL = {
    "novateur/WavTokenizer-large-unify-40token":
        "third_party/WavTokenizer/configs/"
        "wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
    "novateur/WavTokenizer-large-speech-75token":
        "third_party/WavTokenizer/configs/"
        "wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
}


def load_wavtokenizer(model_name: str, device: str = "cuda"):
    """
    Load WavTokenizer using the original repo's ``from_pretrained0802`` API.

    The architecture config (.yaml) ships with the GitHub repo under
    ``third_party/WavTokenizer/configs/`` and the .ckpt is hosted on HF.
    The HF repo does NOT include a config.yaml, so we have to use the
    cloned repo's config.
    """
    import sys
    repo_root = Path(__file__).parent.parent
    wt_dir = repo_root / "third_party" / "WavTokenizer"
    sys.path.insert(0, str(wt_dir))
    from decoder.pretrained import WavTokenizer as WT

    if model_name not in _HF_CKPT_FILENAMES:
        raise ValueError(
            f"Unknown WavTokenizer model '{model_name}'. "
            f"Known: {list(_HF_CKPT_FILENAMES)}"
        )

    from huggingface_hub import hf_hub_download
    ckpt_path = hf_hub_download(
        repo_id=model_name, filename=_HF_CKPT_FILENAMES[model_name]
    )
    config_path = repo_root / _LOCAL_CONFIG_FOR_MODEL[model_name]
    if not config_path.exists():
        raise FileNotFoundError(
            f"WavTokenizer config missing: {config_path}. "
            "Did you clone third_party/WavTokenizer?"
        )

    model = WT.from_pretrained0802(str(config_path), ckpt_path)
    model = model.to(device).eval()
    return model


def encode_directory(input_dir: str, output_dir: str, model_name: str,
                     device: str = "cuda", batch_size: int = 4,
                     target_sr: int = 24000):
    """Encode all WAV files in input_dir to .npy code arrays in output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    device = device if torch.cuda.is_available() else "cpu"
    model = load_wavtokenizer(model_name, device)

    wav_files = sorted(Path(input_dir).glob("**/*.wav"))
    if not wav_files:
        wav_files = sorted(Path(input_dir).glob("**/*.mp3"))
    print(f"Found {len(wav_files)} audio files in {input_dir}")

    for wav_path in tqdm(wav_files, desc="Encoding"):
        wav, sr = torchaudio.load(str(wav_path))
        # Convert to mono, resample to target_sr
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)

        wav = wav.to(device)
        bandwidth_id = torch.tensor([0], device=device)

        with torch.no_grad():
            _, discrete_codes = model.encode_infer(wav, bandwidth_id=bandwidth_id)

        # discrete_codes shape: (n_q, 1, T) or (n_q, T)
        codes = discrete_codes.squeeze().cpu().numpy()
        if codes.ndim == 1:
            codes = codes.reshape(1, -1)  # Ensure (n_q, T)

        out_path = Path(output_dir) / (wav_path.stem + ".npy")
        np.save(out_path, codes)

    print(f"Saved {len(wav_files)} code files to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode WAVs with WavTokenizer")
    parser.add_argument("--input", required=True, help="Input WAV directory")
    parser.add_argument("--output", required=True, help="Output codes directory")
    parser.add_argument("--model", default="novateur/WavTokenizer-large-unify-40token")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    encode_directory(args.input, args.output, args.model, args.device,
                     args.batch_size)
