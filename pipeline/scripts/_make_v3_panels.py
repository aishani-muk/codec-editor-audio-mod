"""Render saraga_eval/v3_panels.png with proper axes + legend."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "results" / "saraga_eval"
US = [0.0, 0.3, 0.6, 0.9]
SYSTEMS = {
    "editor_v3 (Saraga GPT-2 codec)": "editor_v3_eval",
    "editor_v3_lora (Saraga LoRA-MusicGen)": "editor_v3_lora",
}

KEYS = [
    ("ΔA (M2E arousal)", "mean_delta_arousal", "std_delta_arousal", -0.40, "lower-is-better"),
    ("Tonic drift (cents)", "mean_drift_cents", "std_drift_cents", 100.0, "upper-bound"),
    ("Pitch-class JSD", "mean_jsd", "std_jsd", 0.05, "upper-bound"),
    ("Velocity-TV change (%)", "mean_velocity_tv_change_pct", "std_velocity_tv_change_pct", 50.0, "upper-bound"),
]

COLORS = {"editor_v3 (Saraga GPT-2 codec)": "#1f77b4",
          "editor_v3_lora (Saraga LoRA-MusicGen)": "#ff7f0e"}


def collect():
    out = {}
    for label in SYSTEMS:
        out[label] = {}
        for k in KEYS:
            out[label][k[1]] = []
            out[label][k[2]] = []
    for label, sysdir in SYSTEMS.items():
        for u in US:
            p = ROOT / sysdir / f"u{u}" / "eval_results.json"
            agg = (json.loads(p.read_text()).get("aggregate", {})
                   if p.exists() else {})
            for k in KEYS:
                v = agg.get(k[1])
                s = agg.get(k[2])
                out[label][k[1]].append(float(v) if v is not None else np.nan)
                out[label][k[2]].append(float(s) if s is not None else np.nan)
    return out


def main() -> int:
    data = collect()

    fig, axes = plt.subplots(1, len(KEYS), figsize=(5.5 * len(KEYS), 4.4),
                             constrained_layout=True)

    for ax, (title, mean_k, std_k, gate, dirn) in zip(axes, KEYS):
        for label in SYSTEMS:
            mu = np.array(data[label][mean_k])
            sd = np.array(data[label][std_k])
            n = 30
            err = sd / np.sqrt(max(n, 1))
            ax.errorbar(US, mu, yerr=err, marker="o", lw=2, capsize=4,
                        label=label, color=COLORS[label])
        ax.axhline(gate, color="red", linestyle="--", lw=1.2, alpha=0.6,
                   label=f"gate {dirn} {gate:g}")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("u (control intensity)", fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.set_xticks(US)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best", framealpha=0.9)

    fig.suptitle("Saraga test set (n=30) · bootstrap mean ± SE per u",
                 fontsize=13, fontweight="bold")

    out_path = ROOT / "v3_panels.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
