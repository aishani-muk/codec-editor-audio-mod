"""
Apply a trained codec-BPE tokenizer to compress WavTokenizer code sequences.

Usage:
    python tokenize/apply_bpe.py \
        --codes_dir data/tokens/wavtok/ \
        --bpe_model data/tokens/bpe_model/ \
        --output data/tokens/bpe_encoded/
"""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from codec_bpe import codes_to_chars, chars_to_codes
from transformers import AutoTokenizer


def apply_bpe(codes_dir: str, bpe_model: str, output_dir: str,
              codebook_size: int = 4096):
    """
    Apply the trained BPE tokenizer to each .npy code file.

    Saves both:
      - .npy of BPE token IDs (compressed sequence)
      - .txt of the unicode string (for debugging)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(bpe_model)

    code_files = sorted(Path(codes_dir).glob("*.npy"))
    print(f"Applying BPE to {len(code_files)} files")

    ratios = []
    for code_path in tqdm(code_files, desc="BPE encoding"):
        codes = np.load(code_path)  # (n_q, T)
        T_orig = codes.shape[-1]

        # Convert codes to unicode string
        import torch
        codes_tensor = torch.from_numpy(codes)
        unicode_str = codes_to_chars(codes_tensor, codebook_size=codebook_size)

        # Tokenize with BPE
        bpe_tokens = tokenizer.encode(unicode_str)
        T_bpe = len(bpe_tokens)

        ratios.append(T_bpe / T_orig)

        # Save
        out_stem = code_path.stem
        np.save(Path(output_dir) / f"{out_stem}.npy",
                np.array(bpe_tokens, dtype=np.int64))

    mean_ratio = np.mean(ratios)
    print(f"Mean compression ratio K/T = {mean_ratio:.3f} "
          f"({(1-mean_ratio)*100:.1f}% reduction)")
    print(f"BPE-encoded files saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply codec-BPE to codes")
    parser.add_argument("--codes_dir", required=True)
    parser.add_argument("--bpe_model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--codebook_size", type=int, default=4096)
    args = parser.parse_args()

    apply_bpe(args.codes_dir, args.bpe_model, args.output, args.codebook_size)
