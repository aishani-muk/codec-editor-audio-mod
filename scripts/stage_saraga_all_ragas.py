"""Stage the full, already-unzipped Saraga 1.5 Hindustani corpus into a
training-ready directory for `train_raga_classifier_pcd_v2.py`.

Input layout (unzipped on disk at --raw_dir):
    {raw_dir}/{album}/{recording_folder}/{stem}.mp3.mp3
    {raw_dir}/{album}/{recording_folder}/{stem}.ctonic.txt
    {raw_dir}/{album}/{recording_folder}/{stem}.json
    {raw_dir}/{album}/{recording_folder}/{stem}.pitch.txt        (optional)

Output layout (staged to --output):
    {output}/{album}/Raag {common_name}/{stem}.wav               (freshly decoded)
    {output}/{album}/Raag {common_name}/{stem}.ctonic.txt        (hard-linked)
    {output}/{album}/Raag {common_name}/{stem}.pitch.txt         (hard-linked if exists)
    {output}/{album}/Raag {common_name}/{stem}.json              (hard-linked)

Rules:
  * Only recordings with BOTH a ``.ctonic.txt`` AND a ``.json`` that resolves
    to a raga (raags[0].common_name) are staged; commentary / lyrics folders
    are skipped.
  * Raga label comes from ``raags[0].common_name`` in the .json file — NOT
    from the folder name. This ensures consistent labels even when the
    folder is e.g. "Bhairavi Thumri" (form, not raga).
  * Metadata files are hard-linked (same-inode, O(0) bytes, same filesystem
    guaranteed because both paths are under /fs/classhomes/amukherj/).
  * MP3s are decoded to 24 kHz mono PCM_16 WAV (matches WavTokenizer SR).
  * Decoding is parallel across workers; every conversion reports its own
    percent progress.
  * Skips a recording whose target .wav already exists (so reruns are cheap).

Heartbeat: writes a JSONL row per recording to `logs/saraga_stage_progress.jsonl`
so the live dashboard can track progress.

Usage:
    python scripts/stage_saraga_all_ragas.py \
        --raw_dir /fs/classhomes/amukherj/research/saraga1.5_hindustani \
        --output  data/saraga_all_ragas \
        --workers 4
"""
from __future__ import annotations
import argparse, json, os, shutil, signal, sys, time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)

PROGRESS_LOG = Path("logs/saraga_stage_progress.jsonl")
SUMMARY = Path("logs/saraga_stage_summary.json")


def discover_recordings(raw_dir: Path) -> list[dict]:
    """Yield one dict per recording folder that has everything we need."""
    out = []
    for album_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        if album_dir.name.startswith("."):
            continue
        for rec_dir in sorted(p for p in album_dir.iterdir() if p.is_dir()):
            if rec_dir.name.startswith("."):
                continue
            jsons = list(rec_dir.glob("*.json"))
            mp3s = list(rec_dir.glob("*.mp3.mp3"))
            ctonics = list(rec_dir.glob("*.ctonic.txt"))
            if not (jsons and mp3s and ctonics):
                continue
            try:
                with open(jsons[0]) as f:
                    meta = json.load(f)
                common_name = meta["raags"][0]["common_name"].strip()
            except Exception:
                continue
            if not common_name:
                continue
            stem = mp3s[0].name
            for suf in (".mp3.mp3", ".mp3"):
                if stem.lower().endswith(suf):
                    stem = stem[: -len(suf)]
                    break
            pitch = rec_dir / f"{stem}.pitch.txt"
            out.append({
                "album": album_dir.name,
                "recording_folder": rec_dir.name,
                "stem": stem,
                "mp3": str(mp3s[0]),
                "ctonic": str(ctonics[0]),
                "pitch": str(pitch) if pitch.exists() else None,
                "json": str(jsons[0]),
                "raga": common_name,
            })
    return out


def _stage_one(task: dict) -> dict:
    """Worker: convert mp3 -> wav (24 kHz mono) and hard-link metadata."""
    import librosa
    import soundfile as sf

    out_dir = Path(task["output"]) / task["album"] / f"Raag {task['raga']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / f"{task['stem']}.wav"
    errors = []

    t0 = time.time()
    if not wav_path.exists():
        try:
            y, _ = librosa.load(task["mp3"], sr=task["sr"], mono=True)
            sf.write(str(wav_path), y, task["sr"], subtype="PCM_16")
        except Exception as exc:
            errors.append(("wav", repr(exc)))

    # Hard-link metadata (fall back to copy if cross-device).
    for kind, src_key, suffix in [
        ("ctonic", "ctonic", ".ctonic.txt"),
        ("pitch", "pitch", ".pitch.txt"),
        ("json", "json", ".json"),
    ]:
        src = task.get(src_key)
        if not src:
            continue
        dst = out_dir / f"{task['stem']}{suffix}"
        if dst.exists():
            continue
        try:
            os.link(src, dst)
        except OSError:
            try:
                shutil.copy2(src, dst)
            except Exception as exc:
                errors.append((kind, repr(exc)))

    return {
        "album": task["album"], "stem": task["stem"], "raga": task["raga"],
        "wav": str(wav_path),
        "t_s": round(time.time() - t0, 2),
        "errors": errors,
        "ts": round(time.time(), 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="/fs/classhomes/amukherj/research/saraga1.5_hindustani")
    ap.add_argument("--output",  default="data/saraga_all_ragas")
    ap.add_argument("--sr",      type=int, default=24000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    if not raw_dir.is_dir():
        print(f"[stage] raw_dir not found: {raw_dir}", file=sys.stderr)
        return 2

    print(f"[stage] scanning {raw_dir} ...", flush=True)
    records = discover_recordings(raw_dir)
    print(f"[stage] found {len(records)} eligible recordings "
          f"(have .mp3.mp3 + .ctonic.txt + .json with raag)", flush=True)

    by_raga = Counter(r["raga"] for r in records)
    print(f"[stage] distinct ragas: {len(by_raga)}", flush=True)
    for raga, n in by_raga.most_common(15):
        print(f"          {n:3d}  {raga}", flush=True)
    if len(by_raga) > 15:
        print(f"          ... and {len(by_raga)-15} more", flush=True)

    # Drop classes with < 2 recordings (can't be split into train+val).
    singletons = {r for r, n in by_raga.items() if n < 2}
    if singletons:
        print(f"[stage] dropping {len(singletons)} singleton ragas "
              f"(< 2 recordings each)", flush=True)
        records = [r for r in records if r["raga"] not in singletons]

    print(f"[stage] staging to:   {args.output}", flush=True)
    print(f"[stage] workers:      {args.workers}", flush=True)
    if args.dry_run:
        print("[stage] --dry_run: exiting before conversion.", flush=True)
        return 0

    Path(args.output).mkdir(parents=True, exist_ok=True)
    PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    prog = open(PROGRESS_LOG, "a", buffering=1)

    tasks = [
        {**rec, "output": args.output, "sr": args.sr}
        for rec in records
    ]
    t_start = time.time()
    done = n_err = 0
    n_total = len(tasks)

    def _handle_exit(signum, frame):
        print(f"\n[stage] signal {signum}; {done}/{n_total} done, "
              f"{n_err} errors. Rerun to resume.", flush=True)
        prog.flush()
        sys.exit(130 if signum == signal.SIGINT else 143)
    signal.signal(signal.SIGINT, _handle_exit)
    signal.signal(signal.SIGTERM, _handle_exit)

    with Pool(args.workers) as pool:
        for result in pool.imap_unordered(_stage_one, tasks):
            done += 1
            if result["errors"]:
                n_err += 1
            elapsed = time.time() - t_start
            rate = done / elapsed if elapsed else 0
            eta = (n_total - done) / rate if rate else float("inf")
            pct = 100.0 * done / n_total
            prog.write(json.dumps({
                "ts": round(time.time(), 1), "idx": done,
                "n_total": n_total, "pct": round(pct, 2),
                "elapsed_s": round(elapsed, 1),
                "eta_s": None if eta == float("inf") else round(eta, 1),
                **result,
            }) + "\n")
            print(f"[stage] {pct:5.1f}% ({done}/{n_total})  "
                  f"elapsed={elapsed:.0f}s  "
                  f"eta={eta:.0f}s  "
                  f"errs={n_err}  "
                  f"last={result['album']}/{result['raga']}/{result['stem'][:30]}",
                  flush=True)

    # Final summary + per-raga counts on the staged dir.
    staged = []
    out_dir = Path(args.output)
    for wav in out_dir.rglob("*.wav"):
        raga = wav.parent.name.replace("Raag ", "")
        staged.append(raga)
    counts = Counter(staged)
    summary = {
        "raw_dir": str(raw_dir),
        "output": args.output,
        "n_total_candidates": len(records),
        "n_staged": len(staged),
        "n_errors": n_err,
        "n_ragas": len(counts),
        "per_raga": dict(counts.most_common()),
        "elapsed_s": round(time.time() - t_start, 1),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2))
    print(f"\n[stage] DONE in {summary['elapsed_s']:.0f}s. "
          f"{summary['n_staged']} recordings across "
          f"{summary['n_ragas']} ragas.  Summary: {SUMMARY}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
