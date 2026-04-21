"""Score Music2Emo valence/arousal/moods on every (input, target) pair in
data/paired_edits/manifest.jsonl and cache to results/va_cache.jsonl.

Designed for an unattended, resumable, multi-day CPU run:

  - Scores inputs ONCE per unique (track_stem, clip_idx). The input audio is
    byte-identical across all λ in each group (verified by
    scripts/verify_input_dedup.py), so we consult `logs/dedup_groups.json`
    to collapse 7 920 pairs → ~360 unique input scores + 7 920 target scores.
  - Writes a one-line JSON record to results/va_cache.jsonl after every
    finished scoring call, keyed by ``stem`` (for targets) or ``input_key``
    (for inputs). Ctrl-C / kill -9 safe: already-cached keys are skipped on
    restart.
  - Emits a heartbeat every scoring call to logs/m2e_scoring_progress.jsonl
    with {ts, idx, n_total, pct, elapsed_s, eta_s, kind, key}.
  - Stdout prints a terse percent-done line every 30 s via a separate
    wall-clock tick (not tied to call cadence so you see the script isn't
    frozen even while Music2Emo's first forward is warming up).

Usage (typical, from the repo root):
    nohup python -u scripts/m2e_cpu_score.py \
        > logs/m2e_scoring.out 2>&1 &
    echo $! > logs/m2e_scoring.pid
"""
from __future__ import annotations
import json, os, re, signal, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

MANIFEST = Path("data/paired_edits/manifest.jsonl")
INPUT_DIR = Path("data/paired_edits/input")
TARGET_DIR = Path("data/paired_edits/target")
DEDUP_JSON = Path("logs/dedup_groups.json")
CACHE_PATH = Path("results/va_cache.jsonl")
PROGRESS_LOG = Path("logs/m2e_scoring_progress.jsonl")

LAM_RE = re.compile(r"^(.*)_c(\d{3})_lam[0-9.]+$")


# ── Cache I/O ─────────────────────────────────────────────────────────────
def _load_cache() -> dict[str, dict]:
    """Return {key: row} from the append-only JSONL cache. Last write wins."""
    if not CACHE_PATH.exists():
        return {}
    out: dict[str, dict] = {}
    with open(CACHE_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a torn last line from prior kill
            key = row.get("key")
            if key:
                out[key] = row
    return out


_cache_fh = None
def _cache_append(row: dict) -> None:
    global _cache_fh
    if _cache_fh is None:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _cache_fh = open(CACHE_PATH, "a", buffering=1)  # line-buffered
    _cache_fh.write(json.dumps(row) + "\n")


_prog_fh = None
def _progress(row: dict) -> None:
    global _prog_fh
    if _prog_fh is None:
        PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)
        _prog_fh = open(PROGRESS_LOG, "a", buffering=1)
    _prog_fh.write(json.dumps(row) + "\n")


# ── Key derivation ────────────────────────────────────────────────────────
def _input_key(stem: str) -> str:
    """Derive the canonical-input key for a manifest stem."""
    m = LAM_RE.match(stem)
    if not m:
        return f"in:{stem}"
    return f"in:{m.group(1)}__c{int(m.group(2)):03d}"


def _target_key(stem: str) -> str:
    return f"out:{stem}"


# ── Main ──────────────────────────────────────────────────────────────────
def main() -> int:
    print(f"[m2e] cwd={os.getcwd()}", flush=True)
    print(f"[m2e] manifest={MANIFEST}", flush=True)
    print(f"[m2e] cache={CACHE_PATH}", flush=True)

    # Parse the manifest into (stem, input_wav, target_wav) rows.
    stems = []
    with open(MANIFEST) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            stems.append(entry["stem"])
    print(f"[m2e] manifest entries: {len(stems)}", flush=True)

    # Build the scoring schedule: inputs by unique key, targets by stem.
    input_groups: dict[str, Path] = {}
    target_jobs: list[tuple[str, str, Path]] = []   # (key, stem, path)

    for stem in stems:
        ikey = _input_key(stem)
        if ikey not in input_groups:
            in_path = INPUT_DIR / f"{stem}.wav"
            # Symlinks already resolved by soundfile; use directly.
            input_groups[ikey] = in_path
        tkey = _target_key(stem)
        target_jobs.append((tkey, stem, TARGET_DIR / f"{stem}.wav"))

    input_jobs = [(k, v) for k, v in sorted(input_groups.items())]
    print(f"[m2e] unique inputs: {len(input_jobs)}", flush=True)
    print(f"[m2e] target jobs:   {len(target_jobs)}", flush=True)

    cache = _load_cache()
    print(f"[m2e] resuming: {len(cache)} keys already cached", flush=True)

    # Skip already-done jobs up front so we can report an accurate ETA.
    pending_inputs = [(k, p) for k, p in input_jobs if k not in cache]
    pending_targets = [(k, s, p) for k, s, p in target_jobs if k not in cache]
    n_total = len(pending_inputs) + len(pending_targets)
    print(f"[m2e] pending: {len(pending_inputs)} inputs + "
          f"{len(pending_targets)} targets = {n_total} scoring calls",
          flush=True)
    if n_total == 0:
        print("[m2e] nothing to do; exiting.", flush=True)
        return 0

    # Lazy-load the model after we know we have work to do.
    from evaluation.emotion_regressor import Music2EmoRegressor
    reg = Music2EmoRegressor()
    if not reg.available:
        print("[m2e] ERROR: Music2Emo not available (see emotion_regressor.py "
              "for expected paths). Aborting.", flush=True)
        return 2
    print("[m2e] warming up Music2Emo model ...", flush=True)
    t_warm = time.time()
    reg._ensure_loaded()  # triggers first-time model load  # noqa: SLF001
    print(f"[m2e] model ready in {time.time()-t_warm:.1f}s", flush=True)

    t_start = time.time()
    done = 0
    errs = 0
    last_print = t_start

    def _handle_exit(signum, frame):  # graceful shutdown
        print(f"\n[m2e] caught signal {signum}; {done}/{n_total} done, "
              f"elapsed={time.time()-t_start:.1f}s. "
              f"Cache is intact at {CACHE_PATH}; rerun to resume.",
              flush=True)
        if _cache_fh is not None:
            _cache_fh.flush()
        if _prog_fh is not None:
            _prog_fh.flush()
        sys.exit(130 if signum == signal.SIGINT else 143)

    signal.signal(signal.SIGINT, _handle_exit)
    signal.signal(signal.SIGTERM, _handle_exit)

    def _score_one(key: str, kind: str, path: Path, stem: str | None = None) -> None:
        nonlocal done, errs, last_print
        t0 = time.time()
        try:
            result = reg.predict(path)
            row = {
                "key": key, "kind": kind, "stem": stem,
                "path": str(path),
                "valence": float(result["valence"]),
                "arousal": float(result["arousal"]),
                "moods": list(result.get("moods", [])),
                "t_s": round(time.time() - t0, 2),
                "ts": round(time.time(), 1),
            }
            _cache_append(row)
        except Exception as exc:
            errs += 1
            row = {
                "key": key, "kind": kind, "stem": stem,
                "path": str(path),
                "error": repr(exc),
                "t_s": round(time.time() - t0, 2),
                "ts": round(time.time(), 1),
            }
            _cache_append(row)

        done += 1
        elapsed = time.time() - t_start
        rate = done / elapsed if elapsed else 0.0
        eta = (n_total - done) / rate if rate else float("inf")
        pct = 100.0 * done / n_total
        _progress({
            "ts": round(time.time(), 1), "idx": done, "n_total": n_total,
            "pct": round(pct, 2), "elapsed_s": round(elapsed, 1),
            "eta_s": None if eta == float("inf") else round(eta, 1),
            "kind": kind, "key": key, "err": "error" in row,
        })
        # Stdout heartbeat at most once every 30 s (plus one every 50 calls).
        now = time.time()
        if (now - last_print) >= 30 or done % 50 == 0 or done == n_total:
            eta_h = eta / 3600 if eta != float("inf") else 0
            print(f"[m2e] {pct:5.1f}% ({done}/{n_total})  "
                  f"elapsed={elapsed/3600:.2f}h  "
                  f"eta={eta_h:.2f}h  "
                  f"rate={rate:.3f}/s  "
                  f"errs={errs}  "
                  f"last={kind}:{Path(key).name[:40]}", flush=True)
            last_print = now

    # Score inputs first (fewer, faster convergence to meaningful
    # dist_to_neutral_in distributions for early manifest preview).
    for key, path in pending_inputs:
        _score_one(key, "input", path)
    for key, stem, path in pending_targets:
        _score_one(key, "target", path, stem=stem)

    total_elapsed = time.time() - t_start
    print(f"\n[m2e] DONE. {done}/{n_total} scored in "
          f"{total_elapsed/3600:.2f}h. errors={errs}. "
          f"Cache: {CACHE_PATH}", flush=True)
    print(f"[m2e] Build the filtered manifest with: "
          f"python scripts/build_va_manifest.py", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
