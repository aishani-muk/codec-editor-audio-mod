"""Render celtic_panels.png with proper axes + CI bands."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "results"
US = [0.0, 0.3, 0.6, 0.9]
TRACKS = {
    "Track A (GPT-2 codec)": "celtic_eval_track_a",
    "Track B (LoRA-MusicGen)": "celtic_eval_track_b",
    "Track C (hybrid encoder)": "celtic_eval_track_c",
}
COLORS = {"Track A (GPT-2 codec)": "#1f77b4",
          "Track B (LoRA-MusicGen)": "#ff7f0e",
          "Track C (hybrid encoder)": "#2ca02c"}

KEYS = [
    ("ΔA (M2E arousal)", "delta_arousal", -0.40, "lower-is-better"),
    ("Pitch-class JSD", "pcd_jsd", 0.05, "upper-bound"),
    ("Rhythm JSD", "rhythm_jsd", None, None),
    ("Meter retention", "meter_match", None, None),
]


def collect():
    out = {}
    for label in TRACKS:
        out[label] = {k[1]: {"mean": [], "lo": [], "hi": []} for k in KEYS}
    for label, sub in TRACKS.items():
        for u in US:
            p = ROOT / sub / f"u{u}" / "summary.json"
            ov = (json.loads(p.read_text()).get("overall", {})
                  if p.exists() else {})
            for k in KEYS:
                m = ov.get(k[1], {})
                out[label][k[1]]["mean"].append(float(m.get("mean", np.nan))
                                                if isinstance(m, dict) else np.nan)
                out[label][k[1]]["lo"].append(float(m.get("ci_lo", np.nan))
                                              if isinstance(m, dict) else np.nan)
                out[label][k[1]]["hi"].append(float(m.get("ci_hi", np.nan))
                                              if isinstance(m, dict) else np.nan)
    return out


def main() -> int:
    data = collect()
    fig, axes = plt.subplots(1, len(KEYS), figsize=(5.5 * len(KEYS), 4.4),
                             constrained_layout=True)
    for ax, (title, key, gate, dirn) in zip(axes, KEYS):
        for label in TRACKS:
            mu = np.array(data[label][key]["mean"])
            lo = np.array(data[label][key]["lo"])
            hi = np.array(data[label][key]["hi"])
            ax.plot(US, mu, marker="o", lw=2, label=label, color=COLORS[label])
            ax.fill_between(US, lo, hi, alpha=0.15, color=COLORS[label])
        if gate is not None:
            ax.axhline(gate, color="red", linestyle="--", lw=1.2, alpha=0.6,
                       label=f"gate {dirn} {gate:g}")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("u (control intensity)", fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.set_xticks(US)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best", framealpha=0.9)
    fig.suptitle("Celtic test set (n=29) · bootstrap mean with 95 % CI bands",
                 fontsize=13, fontweight="bold")
    out_path = ROOT / "celtic_panels.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
