"""
Download and prepare the WESAD (Wearable Stress and Affect Detection) dataset
for training the stress-proxy classifier.

WESAD provides multimodal physiological data from 15 subjects with labels:
  1 = baseline (neutral)
  2 = stress (Trier Social Stress Test)
  3 = amusement
  4 = meditation (some subjects only)

We train a classifier on {baseline, stress} to produce P(stress) ∈ [0,1],
which becomes our continuous stress proxy s(t).

Dataset access:
  - UCI ML Repository: https://archive.ics.uci.edu/ml/datasets/WESAD
  - Kaggle: https://www.kaggle.com/datasets/orvile/wesad-wearable-stress-affect-detection-dataset

Usage:
    python data/download_wesad.py --output data/wesad/
"""

import argparse
from pathlib import Path


def download_wesad(output_dir: str):
    """Provide instructions for WESAD download and preparation."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("WESAD Dataset Setup")
    print("=" * 60)
    print(f"""
WESAD must be downloaded manually due to license terms.

Option 1: Kaggle (easiest)
  1. Go to: https://www.kaggle.com/datasets/orvile/wesad-wearable-stress-affect-detection-dataset
  2. Download and extract to: {output_dir}

Option 2: UCI ML Repository
  1. Go to: https://archive.ics.uci.edu/ml/datasets/WESAD
  2. Download WESAD.zip
  3. Extract to: {output_dir}

Expected structure after download:
  {output_dir}/
    ├── S2/
    │   ├── S2.pkl          # All sensor data + labels for subject 2
    │   └── S2_readme.pdf
    ├── S3/
    │   ├── S3.pkl
    │   └── ...
    ├── ...
    └── S17/

Each .pkl file contains a dict with:
  - 'signal': dict of sensor modalities
    - 'chest': dict with ACC, ECG, EDA, EMG, Resp, Temp
    - 'wrist': dict with ACC, BVP, EDA, TEMP
  - 'label': array of condition labels (1=baseline, 2=stress, 3=amusement)

After downloading, train the stress proxy model:
  python models/train_stress_classifier.py --data_dir {output_dir} --output models/stress_model/
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WESAD dataset setup")
    parser.add_argument("--output", default="data/wesad")
    args = parser.parse_args()
    download_wesad(args.output)
