"""
Convert Saraga .mp3.mp3 files to canonical .wav at the training sample rate.

Why convert:
  - WAV is lossless → no MP3 decode drift between runs (reproducibility).
  - Resampling to the training SR (24 kHz) here means every downstream stage
    (prepare_pairs, WavTokenizer) sees audio at the right rate.
  - Mono-ised + float32 PCM_16 output keeps files small while staying
    compatible with librosa / soundfile / WavTokenizer.

Default: converts every `*.mp3.mp3` under --data_dir to `<stem>.wav` next to
the source (keeps the MP3 as a backup). Pass --delete_mp3 to remove MP3s
after successful conversion.

Usage:
    python data/mp3_to_wav.py --data_dir data/saraga_kalyan_thaat --sr 24000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import soundfile as sf
from tqdm import tqdm


def convert_one(mp3_path: Path, sr: int) -> Path:
    """Decode an MP3 to mono float32, resample, save as 16-bit PCM WAV."""
    # librosa.load handles .mp3, .mp3.mp3, any audio extension
    y, _ = librosa.load(str(mp3_path), sr=sr, mono=True)
    # Strip the double .mp3.mp3 suffix correctly: "Foo.mp3.mp3" -> "Foo"
    stem = mp3_path.name
    for suf in (".mp3.mp3", ".mp3"):
        if stem.lower().endswith(suf):
            stem = stem[: -len(suf)]
            break
    wav_path = mp3_path.with_name(stem + ".wav")
    sf.write(str(wav_path), y, sr, subtype="PCM_16")
    return wav_path


def run(data_dir: str, sr: int, delete_mp3: bool, force: bool):
    root = Path(data_dir)
    mp3s = sorted(root.glob("**/*.mp3.mp3"))
    if not mp3s:
        # Fallback: plain .mp3
        mp3s = sorted(root.glob("**/*.mp3"))
    print(f"Found {len(mp3s)} MP3 file(s) under {root}")

    n_converted = 0
    n_skipped = 0
    for mp3 in tqdm(mp3s, desc="mp3 -> wav"):
        # Predict the wav path and skip if it already exists (unless --force)
        stem = mp3.name
        for suf in (".mp3.mp3", ".mp3"):
            if stem.lower().endswith(suf):
                stem = stem[: -len(suf)]
                break
        wav_path = mp3.with_name(stem + ".wav")
        if wav_path.exists() and not force:
            n_skipped += 1
            continue
        try:
            convert_one(mp3, sr)
            n_converted += 1
            if delete_mp3:
                mp3.unlink()
        except Exception as e:
            print(f"  FAILED {mp3.name}: {e}")

    print(f"\nDone. Converted {n_converted}, skipped {n_skipped} (already existed)")
    if delete_mp3:
        print(f"  MP3 sources deleted.")
    else:
        print(f"  MP3 sources kept as backup. Pass --delete_mp3 to remove them.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Saraga MP3s to canonical WAV")
    parser.add_argument("--data_dir", default="data/saraga_kalyan_thaat")
    parser.add_argument("--sr", type=int, default=24000,
                        help="Target sample rate (matches WavTokenizer)")
    parser.add_argument("--delete_mp3", action="store_true",
                        help="Remove MP3s after successful conversion")
    parser.add_argument("--force", action="store_true",
                        help="Re-convert even if WAV already exists")
    args = parser.parse_args()
    run(args.data_dir, args.sr, args.delete_mp3, args.force)
