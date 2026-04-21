"""
Generate synthetic paired edits for training the codec-to-codec editor.

For each audio clip, apply parameterized DSP transformations at varying
intensity levels λ ∈ [0,1] to produce (input, edited) pairs that simulate
"lower-arousal" modulation. The editor is then trained to map
input_tokens → edited_tokens conditioned on u = λ.

Transformations applied:
  - Slight pitch lowering (up to -0.5 semitones)
  - Gentle low-pass filtering (reduce high-frequency energy)
  - Dynamic range compression (reduce peaks → calmer dynamics)
  - Tempo micro-adjustment (up to -3% slower)

Window selection modes (``--select_mode``):
  - ``uniform``     : non-overlapping clip_sec chunks from every recording
                     (legacy default, lots of silent / low-info segments)
  - ``informative`` : per recording pick the top ``clips_per_track``
                     non-overlapping clip_sec windows ranked by
                     RMS × (1 − spectral-flatness) (voiced melodic energy).
                     Deterministic. Writes the selection score into meta.

Usage:
    python data/prepare_pairs.py --config configs/proposed.yaml
"""

import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm


def apply_calming_transform(audio: np.ndarray, sr: int,
                            lam: float) -> np.ndarray:
    """
    Apply a parameterized "calming" transformation at intensity λ ∈ [0,1].

    λ=0 → identity (no change)
    λ=1 → maximum calming effect
    """
    y = audio.copy()

    # 1. Pitch shift: up to -0.5 semitones
    pitch_shift = -0.5 * lam
    if abs(pitch_shift) > 0.01:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift)

    # 2. Low-pass filter: reduce high-frequency energy
    # Cutoff from sr/2 (no filter) down to 6000 Hz at max λ
    cutoff = sr / 2 - (sr / 2 - 6000) * lam
    if cutoff < sr / 2 - 100:
        from scipy.signal import butter, sosfilt
        sos = butter(4, cutoff, btype='low', fs=sr, output='sos')
        y = sosfilt(sos, y).astype(np.float32)

    # 3. Dynamic range compression (simple soft-knee)
    threshold_db = -20 + 10 * (1 - lam)  # lower threshold at higher λ
    threshold = 10 ** (threshold_db / 20)
    ratio = 1 + 2 * lam  # 1:1 at λ=0, 3:1 at λ=1
    mask = np.abs(y) > threshold
    if mask.any():
        excess = np.abs(y[mask]) - threshold
        compressed = threshold + excess / ratio
        y[mask] = np.sign(y[mask]) * compressed

    # 4. Tempo: up to -3% slower
    rate = 1.0 - 0.03 * lam
    if rate < 0.99:
        y = librosa.effects.time_stretch(y, rate=rate)

    # Match original length
    if len(y) > len(audio):
        y = y[:len(audio)]
    elif len(y) < len(audio):
        y = np.pad(y, (0, len(audio) - len(y)))

    return y


def pick_informative_windows(audio_path: str, clip_sec: float,
                             clips_per_track: int,
                             analysis_sr: int = 16000,
                             hop_ms: float = 20.0,
                             smooth_sec: float = 0.5):
    """Pick the top-K non-overlapping ``clip_sec`` windows in a recording,
    ranked by a *voiced melodic energy* proxy.

    Score = smoothed RMS energy × (1 − spectral flatness). This prefers
    segments that are both loud AND harmonic (singing / sustained notes)
    and suppresses silence, noise, applause, and percussive-only sections.
    Selection is fully deterministic (argmax with fixed-precision ties).

    Returns
    -------
    list[tuple[int, float]]
        ``[(start_sample_native, score), ...]`` where ``start_sample_native``
        is in the file's native sample rate so it plugs straight into
        :func:`sf.read(start=..., frames=...)`.
    """
    info = sf.info(audio_path)
    native_sr = info.samplerate

    # Read at a modest analysis SR just for scoring (faster, enough detail).
    y, file_sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if file_sr != analysis_sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=analysis_sr)

    hop = max(1, int(analysis_sr * hop_ms / 1000.0))
    win = hop * 2
    rms = librosa.feature.rms(y=y, frame_length=win, hop_length=hop,
                              center=True)[0]
    flat = librosa.feature.spectral_flatness(y=y, n_fft=win,
                                             hop_length=hop,
                                             center=True)[0]
    n = min(len(rms), len(flat))
    rms, flat = rms[:n], flat[:n]

    voiced = rms * (1.0 - np.clip(flat, 0.0, 1.0))

    # Smooth by ``smooth_sec`` moving average so brief spikes don't win.
    smooth_frames = max(1, int(round(smooth_sec / (hop_ms / 1000.0))))
    kernel = np.ones(smooth_frames, dtype=np.float32) / smooth_frames
    voiced = np.convolve(voiced, kernel, mode="same")

    clip_frames = max(1, int(round(clip_sec * analysis_sr / hop)))
    if n < clip_frames:
        return []  # recording too short

    # Cumulative sum trick: O(n) mean-over-window scoring.
    csum = np.concatenate([[0.0], np.cumsum(voiced, dtype=np.float64)])
    window_score = (csum[clip_frames:] - csum[:-clip_frames]) / clip_frames
    window_score = np.round(window_score, 8)  # deterministic tie-breaking

    picks: list[tuple[int, float]] = []
    mask = np.ones_like(window_score, dtype=bool)
    native_clip = int(round(clip_sec * native_sr))
    sec_per_frame = hop / analysis_sr

    for _ in range(clips_per_track):
        if not mask.any():
            break
        candidate = window_score.copy()
        candidate[~mask] = -np.inf
        i = int(np.argmax(candidate))
        if not np.isfinite(candidate[i]):
            break
        start_sec = i * sec_per_frame
        start_native = int(round(start_sec * native_sr))
        start_native = max(0, min(start_native, info.frames - native_clip))
        picks.append((start_native, float(window_score[i])))

        # Invalidate all windows that overlap with this pick.
        lo = max(0, i - clip_frames + 1)
        hi = min(len(window_score), i + clip_frames)
        mask[lo:hi] = False

    return picks


def _process_clip(args_tuple):
    """Worker function for multiprocessing. Loads its own slice on demand
    (streaming) so the parent process doesn't hold all audio in RAM.

    Each (clip, λ) triplet (input.wav, target.wav, meta.npz) is written
    atomically: either all three land on disk or none do.
    """
    (audio_path, start_sample, clip_samples, sr, lambdas, track_stem,
     clip_idx, source_file, input_dir, target_dir, meta_dir,
     window_score) = args_tuple

    try:
        y, file_sr = sf.read(audio_path, start=start_sample,
                             frames=clip_samples, dtype="float32",
                             always_2d=False)
    except Exception as exc:
        return 0, [(track_stem, clip_idx, "load", repr(exc))]
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    if y.ndim > 1:
        y = y.mean(axis=1)

    n_written = 0
    errors = []
    for lam in lambdas:
        pair_name = f"{track_stem}_c{clip_idx:03d}_lam{lam:.3f}"
        in_path = Path(input_dir) / f"{pair_name}.wav"
        tg_path = Path(target_dir) / f"{pair_name}.wav"
        mt_path = Path(meta_dir) / f"{pair_name}.npz"
        try:
            y_edited = apply_calming_transform(y, sr, lam)
            sf.write(str(in_path), y, sr)
            sf.write(str(tg_path), y_edited, sr)
            np.savez(str(mt_path),
                     lambda_val=lam,
                     track_stem=track_stem,
                     clip_idx=clip_idx,
                     source_file=source_file,
                     start_sec=start_sample / sr,
                     duration_sec=len(y) / sr,
                     window_score=float(window_score))
            n_written += 1
        except Exception as exc:
            for p in (in_path, tg_path, mt_path):
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass
            errors.append((track_stem, clip_idx, f"lam{lam}", repr(exc)))
    return n_written, errors


def prepare_pairs(audio_dir: str, output_dir: str, sr: int = 24000,
                  lambdas: list[float] | None = None,
                  clip_sec: float = 10.0,
                  hop_sec: float | None = None,
                  num_workers: int = 1,
                  select_mode: str = "informative",
                  clips_per_track: int = 8):
    """
    Create paired (original, edited) WAV files for training.

    ``select_mode='uniform'``: segment every source into non-overlapping
    ``clip_sec`` chunks (hopping every ``hop_sec``).

    ``select_mode='informative'``: per source recording, pick the top
    ``clips_per_track`` non-overlapping ``clip_sec`` windows ranked by a
    voiced-melodic-energy score. Deterministic, silence-averse, ideal
    for Hindustani vocal where a recording is often dominated by long
    alap/aalap sections and a tabla-heavy tail.

    Regardless of mode, for each selected clip × each λ ∈ lambdas a
    (input, target, meta) triple is written. The track stem is embedded
    in each filename so downstream code can do **source-level splits**.
    """
    if lambdas is None:
        lambdas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    if hop_sec is None:
        hop_sec = clip_sec  # non-overlapping by default

    input_dir = Path(output_dir) / "input"
    target_dir = Path(output_dir) / "target"
    meta_dir = Path(output_dir) / "meta"
    for d in [input_dir, target_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Prefer WAV (canonical; produced by data/mp3_to_wav.py). Fall back to MP3.
    audio_files = sorted(Path(audio_dir).glob("**/*.wav"))
    if not audio_files:
        audio_files = sorted(Path(audio_dir).glob("**/*.mp3.mp3"))
    if not audio_files:
        audio_files = sorted(Path(audio_dir).glob("**/*.mp3"))
    print(f"Found {len(audio_files)} audio files "
          f"(select_mode={select_mode}, clip_sec={clip_sec}, "
          f"clips_per_track={clips_per_track})")

    # Build the full job list: only file paths + offsets (no audio buffers).
    jobs = []
    print("Indexing source recordings...")
    for audio_path in tqdm(audio_files, desc="Indexing"):
        info = sf.info(str(audio_path))
        native_sr = info.samplerate
        native_clip = int(clip_sec * native_sr)
        native_hop = int(hop_sec * native_sr)
        native_total = info.frames

        album = audio_path.parent.parent.name
        track_stem = f"{album}__{audio_path.parent.name}__{audio_path.stem}"
        for ch in (" ", ",", "/", "&", "(", ")", ":"):
            track_stem = track_stem.replace(ch, "_")

        if select_mode == "informative":
            picks = pick_informative_windows(
                str(audio_path), clip_sec, clips_per_track
            )
            for clip_idx, (start, score) in enumerate(picks):
                jobs.append((str(audio_path), start, native_clip, sr,
                             lambdas, track_stem, clip_idx,
                             str(audio_path.name), str(input_dir),
                             str(target_dir), str(meta_dir), score))
        elif select_mode == "uniform":
            clip_idx = 0
            for start in range(0, native_total - native_clip + 1,
                               native_hop):
                jobs.append((str(audio_path), start, native_clip, sr,
                             lambdas, track_stem, clip_idx,
                             str(audio_path.name), str(input_dir),
                             str(target_dir), str(meta_dir), 0.0))
                clip_idx += 1
        else:
            raise ValueError(f"Unknown select_mode={select_mode!r}")

    print(f"Generating pairs for {len(jobs)} clips x {len(lambdas)} lambdas "
          f"= {len(jobs) * len(lambdas)} total pairs "
          f"(workers={num_workers})")

    pair_count = 0
    all_errors = []
    if num_workers <= 1:
        for job in tqdm(jobs, desc="Generating pairs"):
            n, errs = _process_clip(job)
            pair_count += n
            all_errors.extend(errs)
    else:
        with Pool(num_workers) as pool:
            for n, errs in tqdm(pool.imap_unordered(_process_clip, jobs),
                                total=len(jobs), desc="Generating pairs"):
                pair_count += n
                all_errors.extend(errs)

    print(f"Generated {pair_count} training pairs in {output_dir}")
    if all_errors:
        err_path = Path(output_dir) / "errors.log"
        with open(err_path, "w") as f:
            for e in all_errors:
                f.write("\t".join(str(x) for x in e) + "\n")
        print(f"  {len(all_errors)} pairs failed; see {err_path}")

    _reap_orphans_and_write_manifest(output_dir)


def _reap_orphans_and_write_manifest(output_dir: str) -> None:
    """Ensure input/target/meta directories stay in sync and emit a
    ``manifest.jsonl`` listing only triples with all three files present.
    Any unmatched file is deleted.
    """
    import json
    out = Path(output_dir)
    in_dir, tg_dir, mt_dir = out / "input", out / "target", out / "meta"

    in_stems = {p.stem for p in in_dir.glob("*.wav")}
    tg_stems = {p.stem for p in tg_dir.glob("*.wav")}
    mt_stems = {p.stem for p in mt_dir.glob("*.npz")}
    valid = sorted(in_stems & tg_stems & mt_stems)

    orphans_removed = 0
    for s in in_stems - set(valid):
        (in_dir / f"{s}.wav").unlink(missing_ok=True)
        orphans_removed += 1
    for s in tg_stems - set(valid):
        (tg_dir / f"{s}.wav").unlink(missing_ok=True)
        orphans_removed += 1
    for s in mt_stems - set(valid):
        (mt_dir / f"{s}.npz").unlink(missing_ok=True)
        orphans_removed += 1

    manifest_path = out / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for stem in valid:
            meta = np.load(mt_dir / f"{stem}.npz")
            entry = {
                "stem": stem,
                "track_stem": str(meta["track_stem"]),
                "clip_idx": int(meta["clip_idx"]),
                "lambda_val": float(meta["lambda_val"]),
                "source_file": str(meta["source_file"]),
                "start_sec": float(meta["start_sec"]),
                "duration_sec": float(meta["duration_sec"]),
            }
            if "window_score" in meta.files:
                entry["window_score"] = float(meta["window_score"])
            f.write(json.dumps(entry) + "\n")
    print(f"Manifest: {len(valid)} valid pairs → {manifest_path}"
          f"  (reaped {orphans_removed} orphaned files)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic paired edits")
    parser.add_argument("--audio_dir", default="data/saraga_kalyan_thaat",
                        help="Source audio directory")
    parser.add_argument("--output", default="data/paired_edits",
                        help="Output paired data directory")
    parser.add_argument("--sr", type=int, default=24000)
    parser.add_argument("--clip_sec", type=float, default=10.0,
                        help="Length of each segmented clip (seconds)")
    parser.add_argument("--hop_sec", type=float, default=None,
                        help="Stride between segmented clips "
                             "(uniform mode; default = clip_sec)")
    parser.add_argument("--select_mode",
                        choices=["uniform", "informative"],
                        default="informative",
                        help="How to choose windows within each source")
    parser.add_argument("--clips_per_track", type=int, default=8,
                        help="Top-K windows per recording in "
                             "'informative' mode")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Parallel workers for the DSP transforms")
    parser.add_argument("--reap_only", action="store_true",
                        help="Skip DSP; just reap orphans and rebuild manifest.jsonl")
    parser.add_argument("--lambda_mode",
                        choices=["discrete", "uniform"], default="discrete",
                        help="discrete: fixed 0.0,0.2,...,1.0 grid; "
                             "uniform: n_lambdas points sampled from U(0,1)")
    parser.add_argument("--n_lambdas", type=int, default=6,
                        help="Number of lambda points per clip "
                             "(used with --lambda_mode=uniform)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for uniform lambda sampling")
    args = parser.parse_args()

    if args.lambda_mode == "uniform":
        _rng = np.random.default_rng(args.seed)
        _lam = sorted(_rng.uniform(0.0, 1.0, size=args.n_lambdas).tolist())
        # Always pin the extremes so the model sees identity (0) and
        # full-effect (1) regardless of n.
        _lam = sorted(set([0.0, 1.0]) | set(round(x, 4) for x in _lam))
        print(f"Uniform-λ schedule ({len(_lam)} points): {_lam}")
        _lambdas = _lam
    else:
        _lambdas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    if args.reap_only:
        _reap_orphans_and_write_manifest(args.output)
    else:
        prepare_pairs(args.audio_dir, args.output, args.sr,
                      lambdas=_lambdas,
                      clip_sec=args.clip_sec, hop_sec=args.hop_sec,
                      num_workers=args.num_workers,
                      select_mode=args.select_mode,
                      clips_per_track=args.clips_per_track)
