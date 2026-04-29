"""Regenerate every demo WAV in results/demo_clips/ and the A/B HTML.

Re-running this from a fresh checkout (with the venv set up) should
produce byte-identical files to those shipped in this folder.
"""
from pathlib import Path
import sys, soundfile as sf

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from demo.runner import DemoRunner
from demo.profiles import ramp_up, pulse


CLIPS = {
    "yaman": "data/demo_inputs/yaman_c00.wav",
    "todi":  "data/demo_inputs/todi_c00.wav",
    "shree": "data/demo_inputs/shree_c00.wav",
}
PROFILES = ["ramp", "pulse"]
OUT_DIR = REPO / "results" / "demo_clips"
HTML_PATH = REPO / "results" / "ab_demo.html"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runner = DemoRunner()
    for name, in_rel in CLIPS.items():
        in_wav = REPO / in_rel
        n_samples = int(sf.info(str(in_wav)).duration * 24000)
        n_frames = max(2, n_samples // 256)
        for prof in PROFILES:
            u = ramp_up(n_frames, peak=0.8) if prof == "ramp" \
                else pulse(n_frames, peak=0.9, width=0.30)
            out_wav = OUT_DIR / f"{name}_{prof}.wav"
            r = runner.run(str(in_wav), u, model="dsp",
                           output_wav=str(out_wav), compute_metrics=False)
            rtf = r.get("latency", {}).get("rtf", 0.0)
            ok = r.get("ok", True)
            print(f"  {name:6s} {prof:5s}  ok={ok}  rtf={rtf:.3f}  -> {out_wav.name}")

    rows_ramp = "\n".join(_row(n, "ramp") for n in CLIPS)
    rows_pulse = "\n".join(_row(n, "pulse") for n in CLIPS)
    HTML_PATH.write_text(_HTML.format(rows_ramp=rows_ramp, rows_pulse=rows_pulse))
    print(f"\nwrote {HTML_PATH}")
    print(f"open  file://{HTML_PATH}  in a browser")
    return 0


def _row(name: str, prof: str) -> str:
    pretty = {"yaman": "Yaman (Bahar Gaud Malhar)",
              "todi": "Todi (Kumar Gandharva)",
              "shree": "Shree (Ajoy Chakrabarty)"}[name]
    in_rel = f"../data/demo_inputs/{name}_c00.wav"
    out_rel = f"demo_clips/{name}_{prof}.wav"
    return (
        f'<div class="row">'
        f'<div class="raga">{pretty}</div>'
        f'<div><div class="lbl">in</div><audio controls preload="none" src="{in_rel}"></audio></div>'
        f'<div><div class="lbl">out</div><audio controls preload="none" src="{out_rel}"></audio></div>'
        f'</div>'
    )


_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Pipeline demo - streaming codec-token editor for raga modulation</title>
<style>
body {{ font-family: system-ui, -apple-system, sans-serif; max-width: 920px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; line-height: 1.5; }}
h1 {{ font-size: 1.5rem; margin-bottom: .2rem; }}
.sub {{ color: #666; margin-bottom: 1.5rem; }}
.row, .header {{ display: grid; grid-template-columns: 220px 1fr 1fr; gap: 1rem; align-items: center; padding: .8rem 0; border-bottom: 1px solid #eee; }}
.header {{ font-weight: 600; color: #444; border-bottom: 2px solid #444; padding-bottom: .5rem; }}
.raga {{ font-weight: 600; }}
.lbl {{ font-size: .75rem; color: #888; text-transform: uppercase; letter-spacing: .05em; margin-bottom: .2rem; }}
audio {{ width: 100%; }}
details {{ margin-top: 2rem; padding: 1rem; background: #fafafa; border-radius: 6px; }}
code {{ background: #f0f0f0; padding: .1em .3em; border-radius: 3px; font-size: .9em; }}
</style></head><body>
<h1>Streaming codec-token editor for raga modulation</h1>
<p class="sub">A/B demo: original Saraga raga clips (10 s, 24 kHz) vs DSP-edited output under a time-varying stress-proxy signal u(t). Both clips at each row are decoded to identical PCM_16 24 kHz so the only audible difference is from u-controlled editing.</p>

<h2>u(t) ramp 0 to 0.8</h2>
<div class="header"><div>raga</div><div>input</div><div>edited (ramp)</div></div>
{rows_ramp}

<h2 style="margin-top:2rem;">u(t) Gaussian pulse, peak 0.9 at t=0.5</h2>
<div class="header"><div>raga</div><div>input</div><div>edited (pulse)</div></div>
{rows_pulse}

<details><summary>What's running</summary>
<p>Pipeline: WavTokenizer encode &rarr; BPE compress &rarr; DSP back-end (parametric EQ + compressor + mild pitch shift, all conditioned on u) &rarr; cosine-crossfade overlap-add &rarr; 24 kHz WAV. Real-time factor &lt; 0.12 on these clips.</p>
<p>The trained LoRA-MusicGen editor (<code>checkpoints/editor_v3_lora/</code>) was found to be u-blind on this corpus (bit-identical output across u in [0.0, 0.3, 0.6, 0.9]). The DSP back-end is wired into this demo because it actually responds to u. See <code>RESULTS.md</code> for the gate verdicts.</p>
</details>
</body></html>
"""


if __name__ == "__main__":
    sys.exit(main())
