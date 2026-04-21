"""Replace duplicated input WAVs in data/paired_edits/input/ with symlinks to
a single canonical file per (track_stem, clip_idx) group.

Prereq: `scripts/verify_input_dedup.py` must have exited 0 (every group is
internally byte-identical).

Action per group:
  1. Move the first (lexicographically sorted) file into
       data/paired_edits/input/_canonical/<key>.wav
  2. Replace every other file in the group — and the just-moved file's
     original path — with a relative symlink to the canonical file.

Downstream code that reads ``input/*_lam*.wav`` works unchanged because
WAV-reading libraries follow symlinks transparently.

Safety:
  - Uses os.rename then os.symlink. If anything fails mid-group, the
    partially-symlinked state is recoverable by re-running with --rollback
    (which re-copies the canonical back over every symlink).
  - Writes logs/dedup_apply.log with every action.
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
from pathlib import Path

INPUT_DIR = Path("data/paired_edits/input")
CANON_DIR = INPUT_DIR / "_canonical"
GROUPS_JSON = Path("logs/dedup_groups.json")
LOG = Path("logs/dedup_apply.log")

LAM_RE = re.compile(r"^(.*)_c(\d{3})_lam[0-9.]+$")


def group_key(stem: str) -> tuple[str, int] | None:
    m = LAM_RE.match(stem)
    if not m:
        return None
    return m.group(1), int(m.group(2))


def _rel_symlink_target(link: Path, target: Path) -> str:
    """Relative path from link's parent to target — keeps moves portable."""
    return os.path.relpath(target, start=link.parent)


def apply_dedup(dry_run: bool = False) -> int:
    if not GROUPS_JSON.exists():
        print(f"Missing {GROUPS_JSON}; run verify_input_dedup.py first.",
              file=sys.stderr)
        return 2

    groups = json.loads(GROUPS_JSON.read_text())
    CANON_DIR.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)

    before_bytes = sum(p.stat().st_size for p in INPUT_DIR.glob("*.wav")
                       if not p.is_symlink())
    t0 = time.time()
    n_groups = len(groups)
    n_linked = 0
    n_canon = 0
    n_errors = 0

    with open(LOG, "w") as flog:
        flog.write(f"# dedup apply @ {time.ctime()}  dry_run={dry_run}\n")
        flog.write(f"# input_dir={INPUT_DIR}  canon_dir={CANON_DIR}\n\n")

        for i, (key, names) in enumerate(sorted(groups.items())):
            if not names:
                continue
            canon_name = f"{key}.wav"
            canon_path = CANON_DIR / canon_name
            files = [INPUT_DIR / n for n in sorted(names)]

            # Skip groups already deduped (every file already a symlink).
            if all(p.is_symlink() for p in files) and canon_path.exists():
                continue

            # Move first real file to canonical.
            first = next((p for p in files if not p.is_symlink()), None)
            if first is None:
                flog.write(f"[WARN] {key}: no non-symlink file found; skipping\n")
                continue

            try:
                if not dry_run:
                    if canon_path.exists():
                        # Canonical already exists (e.g. a resumed run); drop
                        # the redundant file.
                        first.unlink()
                    else:
                        os.rename(first, canon_path)
                        n_canon += 1
                else:
                    flog.write(f"[DRY ] mv {first.name} -> _canonical/{canon_name}\n")
                    n_canon += 1
            except Exception as exc:
                flog.write(f"[ERR ] move {first} -> {canon_path}: {exc!r}\n")
                n_errors += 1
                continue

            # Every other file (and the vacated first path) becomes a symlink.
            for p in files:
                if p.exists() and p.is_symlink():
                    continue  # already linked
                try:
                    if p.exists() and not p.is_symlink():
                        if not dry_run:
                            p.unlink()
                        else:
                            flog.write(f"[DRY ] rm {p.name}\n")
                    rel = _rel_symlink_target(p, canon_path)
                    if not dry_run:
                        os.symlink(rel, p)
                    else:
                        flog.write(f"[DRY ] ln -s {rel} {p.name}\n")
                    n_linked += 1
                except Exception as exc:
                    flog.write(f"[ERR ] symlink {p} -> {canon_path}: {exc!r}\n")
                    n_errors += 1

            if (i + 1) % 20 == 0 or i == n_groups - 1:
                dt = time.time() - t0
                rate = (i + 1) / dt if dt else 0
                eta = (n_groups - i - 1) / rate if rate else 0
                print(f"  [{100*(i+1)/n_groups:5.1f}%] ({i+1}/{n_groups}) "
                      f"canon={n_canon} linked={n_linked} errs={n_errors} "
                      f"elapsed={dt:.0f}s eta={eta:.0f}s", flush=True)

    after_bytes = sum(p.stat().st_size for p in INPUT_DIR.glob("*.wav")
                      if not p.is_symlink())
    canon_bytes = sum(p.stat().st_size for p in CANON_DIR.glob("*.wav"))

    freed = before_bytes - (after_bytes + canon_bytes)
    print(f"\nDONE in {time.time()-t0:.1f}s")
    print(f"  groups:          {n_groups}")
    print(f"  canonical moved: {n_canon}")
    print(f"  symlinks made:   {n_linked}")
    print(f"  errors:          {n_errors}")
    print(f"  before:          {before_bytes/1e9:.2f} GB")
    print(f"  after non-link:  {after_bytes/1e9:.2f} GB")
    print(f"  canonical size:  {canon_bytes/1e9:.2f} GB")
    print(f"  space freed:     {freed/1e9:.2f} GB")
    print(f"  log:             {LOG}")
    return 1 if n_errors else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    sys.exit(apply_dedup(dry_run=args.dry_run))
