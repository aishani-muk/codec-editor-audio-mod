"""
Filter Saraga Hindustani 1.5 to the Kalyan-thaat cluster (Yaman's family).

Extracts only the audio + tonic + pitch + metadata for tracks whose raga is
in the Kalyan-thaat set. This keeps the dataset musically coherent while
providing enough material (~6.5 hr) to train the codec editor.

Kalyan-thaat ragas in this dataset (15 tracks total):
  Yaman kalyan, Kalyan, Shuddha kalyan, Bhoop, Kedar, Basanti kedar,
  Hameer, Hindol Pancham, Nat Kamod, Jait Kalyan, Marubihag, Bihag

Usage:
    python data/filter_saraga.py \
        --zip /fs/classhomes/amukherj/research/saraga1.5_hindustani.zip \
        --output data/saraga_kalyan_thaat/

The script streams from the zip without fully unpacking it, extracting only
the files belonging to selected tracks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path


KALYAN_THAAT_RAGAS = {
    "Yaman kalyan",
    "Kalyan",
    "Shuddha kalyan",
    "Bhoop",          # Bhupali
    "Kedar",
    "Basanti kedar",
    "Hameer",
    "Hindol Pancham",
    "Nat Kamod",
    "Jait Kalyan",
    "Marubihag",
    "Bihag",
}

# Per-track file suffixes we want (Saraga 1.5 naming)
TRACK_SUFFIXES = (
    ".mp3.mp3",       # audio
    ".json",          # metadata
    ".ctonic.txt",    # tonic frequency (Hz), single float
    ".pitch.txt",     # pitch track (time, Hz)
    ".pitch-pp.txt",  # post-processed pitch (if present)
    ".tempo-manual.txt",
    ".sections-manual.txt",
)


def is_yaman(raag_names: list[str]) -> bool:
    return any(r == "Yaman kalyan" for r in raag_names)


def filter_and_extract(zip_path: str, output_dir: str, dry_run: bool = False):
    output_dir = Path(output_dir)
    zp = zipfile.ZipFile(zip_path, "r")

    # Pass 1: read every .json and decide which track dirs to keep
    print(f"Scanning {zip_path} for Kalyan-thaat tracks...")
    selected_track_prefixes = []  # e.g., "saraga1.5_hindustani/<album>/<track>/"
    yaman_track_prefixes = []

    for info in zp.infolist():
        name = info.filename
        if name.startswith("__MACOSX/"):
            continue
        if not name.endswith(".json"):
            continue
        if name.count("/") != 3:  # saraga1.5_hindustani/<album>/<track>/<track>.json
            continue

        with zp.open(info) as f:
            try:
                meta = json.load(f)
            except Exception:
                continue
        raags = [r.get("common_name", "") for r in meta.get("raags", [])]
        if any(r in KALYAN_THAAT_RAGAS for r in raags):
            prefix = name.rsplit("/", 1)[0] + "/"
            selected_track_prefixes.append((prefix, raags[0]))
            if is_yaman(raags):
                yaman_track_prefixes.append(prefix)

    print(f"  Selected {len(selected_track_prefixes)} tracks "
          f"(of which {len(yaman_track_prefixes)} are Yaman)")

    if dry_run:
        for p, r in sorted(selected_track_prefixes):
            print(f"  [{r}]  {p}")
        return

    # Pass 2: extract files under those prefixes with allowed suffixes
    output_dir.mkdir(parents=True, exist_ok=True)
    n_extracted = 0
    total_bytes = 0
    for info in zp.infolist():
        name = info.filename
        if name.startswith("__MACOSX/"):
            continue
        if info.is_dir():
            continue
        if not any(name.endswith(s) for s in TRACK_SUFFIXES):
            continue
        if not any(name.startswith(p) for p, _ in selected_track_prefixes):
            continue

        # Strip "saraga1.5_hindustani/" prefix to flatten into output_dir
        rel = name[len("saraga1.5_hindustani/"):]
        dest = output_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)

        with zp.open(info) as src, open(dest, "wb") as dst:
            while True:
                chunk = src.read(1 << 20)
                if not chunk:
                    break
                dst.write(chunk)
                total_bytes += len(chunk)
        n_extracted += 1
        if n_extracted % 20 == 0:
            print(f"  ... {n_extracted} files, {total_bytes/1e6:.1f} MB")

    # Also extract the top-level file_paths.csv (useful for reference)
    try:
        zp.extract("saraga1.5_hindustani/file_paths.csv", path=str(output_dir.parent / "_saraga_staging"))
    except KeyError:
        pass

    # Write a manifest of selected tracks
    manifest = {
        "n_tracks": len(selected_track_prefixes),
        "n_yaman": len(yaman_track_prefixes),
        "kalyan_thaat_ragas": sorted(KALYAN_THAAT_RAGAS),
        "tracks": [
            {"raga": r, "path": p[len("saraga1.5_hindustani/"):].rstrip("/")}
            for p, r in sorted(selected_track_prefixes)
        ],
        "yaman_tracks": [
            p[len("saraga1.5_hindustani/"):].rstrip("/") for p in sorted(yaman_track_prefixes)
        ],
    }
    with open(output_dir / "_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone.")
    print(f"  {n_extracted} files extracted ({total_bytes/1e9:.2f} GB)")
    print(f"  Output: {output_dir}")
    print(f"  Manifest: {output_dir}/_manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter Saraga Hindustani 1.5 to Kalyan-thaat cluster"
    )
    parser.add_argument("--zip", required=True, help="Path to saraga1.5_hindustani.zip")
    parser.add_argument("--output", default="data/saraga_kalyan_thaat",
                        help="Output directory")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only list selected tracks, do not extract")
    args = parser.parse_args()

    filter_and_extract(args.zip, args.output, dry_run=args.dry_run)
