"""
Train a codec-BPE tokenizer on WavTokenizer-encoded audio codes.

Uses the codec-bpe library (https://github.com/AbrahamSanders/codec-bpe).

Usage:
    python tokenize/train_bpe.py \
        --codes_dir data/tokens/wavtok/ \
        --output data/tokens/bpe_model/ \
        --vocab_size 8192
"""

import argparse
import subprocess
import sys
from pathlib import Path


def train_bpe(codes_dir: str, output_dir: str, vocab_size: int = 8192,
              codebook_size: int = 4096, max_ngrams: int = 4):
    """
    Train codec-BPE tokenizer using the codec_bpe CLI.

    Steps:
    1. The .npy files from encode_wavtokenizer.py are already in the right format.
    2. Use codec_bpe.train to build the BPE vocabulary.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "codec_bpe.train",
        "--codes_path", codes_dir,
        "--output_path", output_dir,
        "--codebook_size", str(codebook_size),
        "--vocab_size", str(vocab_size),
        "--max_token_codebook_ngrams", str(max_ngrams),
    ]

    print(f"Training BPE tokenizer: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    print(f"BPE tokenizer saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train codec-BPE tokenizer")
    parser.add_argument("--codes_dir", required=True,
                        help="Directory of .npy code files")
    parser.add_argument("--output", required=True,
                        help="Output directory for BPE model")
    parser.add_argument("--vocab_size", type=int, default=8192)
    parser.add_argument("--codebook_size", type=int, default=4096)
    parser.add_argument("--max_ngrams", type=int, default=4)
    args = parser.parse_args()

    train_bpe(args.codes_dir, args.output, args.vocab_size,
              args.codebook_size, args.max_ngrams)
