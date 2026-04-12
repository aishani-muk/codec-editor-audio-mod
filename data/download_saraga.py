"""
Download Saraga Hindustani recordings for the Yaman raga subset.

Saraga is hosted at https://mtg.github.io/saraga/ and the Hindustani
file-path mapping is at:
https://github.com/MTG/saraga/blob/master/dataset/hindustani/file_paths.csv

The recordings can be downloaded via the CompMusic Dunya API or directly
from the Saraga Zenodo archive.

Usage:
    python data/download_saraga.py --raga yaman --output data/saraga_yaman/
"""

import argparse
import os
import csv
from pathlib import Path


def download_saraga_yaman(output_dir: str, raga: str = "yaman"):
    """
    Download Saraga Hindustani recordings for the specified raga.

    NOTE: Saraga requires authentication via the CompMusic Dunya API.
    You need to:
    1. Register at https://dunya.compmusic.upf.edu/
    2. Get an API token
    3. Set the DUNYA_TOKEN environment variable

    Alternatively, download manually from the Zenodo archive:
    https://zenodo.org/record/4301737 (Hindustani subset)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    token = os.environ.get("DUNYA_TOKEN")
    if not token:
        print("=" * 60)
        print("MANUAL DOWNLOAD REQUIRED")
        print("=" * 60)
        print(f"""
Saraga Hindustani requires a Dunya API token for programmatic access.

Option A: Dunya API
  1. Register at https://dunya.compmusic.upf.edu/
  2. Get your API token from your profile page
  3. Re-run with: DUNYA_TOKEN=<your_token> python data/download_saraga.py

Option B: Manual download
  1. Go to https://zenodo.org/record/4301737
  2. Download the Hindustani subset
  3. Extract Yaman recordings to: {output_dir}
  4. Also download the tonic (.tonic) and pitch (.pitch) annotation files

Option C: Use the Saraga GitHub file_paths.csv to identify Yaman recordings:
  https://github.com/MTG/saraga/blob/master/dataset/hindustani/file_paths.csv
  Filter for rows containing "Yaman" in the raga field.

Expected directory structure after download:
  {output_dir}/
    ├── YMN-01.wav          # Audio recordings
    ├── YMN-01.pitch        # Pitch track (timestamp, Hz)
    ├── YMN-01.tonic        # Tonic frequency (Hz)
    ├── YMN-02.wav
    ├── ...
""")
        return

    # If token available, use Dunya API
    try:
        import compmusic
        compmusic.dunya.set_token(token)
        from compmusic.dunya import hindustani

        recordings = hindustani.get_recordings()
        yaman_recordings = [
            r for r in recordings
            if raga.lower() in str(r.get("raags", "")).lower()
        ]
        print(f"Found {len(yaman_recordings)} {raga} recordings via Dunya API")

        for rec in yaman_recordings:
            rec_id = rec["mbid"]
            # Download audio
            mp3_content = hindustani.get_recording(rec_id)["file"]
            out_path = Path(output_dir) / f"{rec_id}.mp3"
            with open(out_path, "wb") as f:
                f.write(mp3_content)
            print(f"  Downloaded {out_path.name}")

    except ImportError:
        print("compmusic package not installed. Install with: pip install compmusic")
        print("Falling back to manual download instructions above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Saraga Hindustani data")
    parser.add_argument("--raga", default="yaman")
    parser.add_argument("--output", default="data/saraga_yaman")
    args = parser.parse_args()

    download_saraga_yaman(args.output, args.raga)
