"""
Download the DEAM (Database for Emotional Analysis of Music) dataset.

DEAM provides per-song and per-second valence/arousal annotations for ~1800 songs.
Homepage: https://cvml.unige.ch/databases/DEAM/

Usage:
    python data/download_deam.py --output data/deam/
"""

import argparse
import os
import zipfile
from pathlib import Path
import urllib.request


DEAM_URLS = {
    "annotations_static":
        "https://cvml.unige.ch/databases/DEAM/DEAM_Annotations.zip",
    "annotations_dynamic":
        "https://cvml.unige.ch/databases/DEAM/DEAM_Annotations.zip",
}

# Audio must be obtained separately (MediaEval challenge terms)
AUDIO_NOTE = """
DEAM audio files are from the MediaEval challenge and require agreement to terms.
Download options:
  1. Kaggle: https://www.kaggle.com/datasets/imsparsh/deam-mediaeval-dataset-emotional-analysis-in-music
  2. Direct: contact the DEAM maintainers at https://cvml.unige.ch/databases/DEAM/

After downloading, place audio files in: {output_dir}/audio/
Expected format: MEMD_audio_{id}.mp3
"""


def download_deam(output_dir: str):
    """Download DEAM annotations and provide audio download instructions."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Download annotations
    annot_url = DEAM_URLS["annotations_static"]
    zip_path = Path(output_dir) / "DEAM_Annotations.zip"

    if not zip_path.exists():
        print(f"Downloading DEAM annotations from {annot_url}...")
        try:
            urllib.request.urlretrieve(annot_url, str(zip_path))
            print(f"  Saved to {zip_path}")
        except Exception as e:
            print(f"  Download failed: {e}")
            print("  Please download manually from: https://cvml.unige.ch/databases/DEAM/")

    # Extract
    if zip_path.exists():
        annot_dir = Path(output_dir) / "annotations"
        annot_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(str(annot_dir))
        print(f"  Extracted annotations to {annot_dir}")

    # Audio instructions
    audio_dir = Path(output_dir) / "audio"
    if not audio_dir.exists():
        audio_dir.mkdir(exist_ok=True)
        print(AUDIO_NOTE.format(output_dir=output_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download DEAM dataset")
    parser.add_argument("--output", default="data/deam")
    args = parser.parse_args()

    download_deam(args.output)
