"""Pipeline integrity check. Run from pipeline/ as cwd."""
import sys, warnings, subprocess
warnings.filterwarnings("ignore")

print("=== package versions ===")
for m in ["torch", "librosa", "soundfile", "pedalboard", "numpy"]:
    try:
        x = __import__(m)
        print(f"  {m}: {getattr(x, '__version__', '?')}")
    except Exception as e:
        print(f"  {m}: FAIL ({type(e).__name__})")

print()
print("=== internal imports ===")
sys.path.insert(0, ".")
mods = [
    "baselines.dsp_baseline",
    "demo.runner",
    "demo.profiles",
    "models.codec_editor",
    "models.musicgen_lora",
    "models.stress_proxy",
    "tokenization.encode_wavtokenizer",
    "evaluation.emotion_regressor",
]
for m in mods:
    try:
        __import__(m)
        print(f"  OK  {m}")
    except Exception as e:
        print(f"  FAIL {m}: {type(e).__name__}: {e}")

print()
print("=== regenerate_demo.py end-to-end ===")
r = subprocess.run([sys.executable, "regenerate_demo.py"], capture_output=True, text=True, timeout=120)
print(r.stdout[-700:] if r.stdout else "(no stdout)")
if r.returncode != 0:
    print("STDERR:", r.stderr[-700:])
print(f"  exit code: {r.returncode}")
