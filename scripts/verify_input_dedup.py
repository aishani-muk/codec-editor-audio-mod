"""Verify that every input WAV is byte-identical across all λ for the same
(track_stem, clip_idx) pair.

Writes:
  logs/dedup_verify.log     -- full per-group report (only violations detailed)
  logs/dedup_groups.json    -- {"(track_stem, clip_idx)": [stems ...]} groups
  stdout                     -- summary

Exit code: 0 if safe to dedup (all groups internally identical), 1 otherwise.
"""
from __future__ import annotations
import hashlib, json, re, sys, time
from collections import defaultdict
from pathlib import Path

INPUT_DIR = Path("data/paired_edits/input")
LOG = Path("logs/dedup_verify.log")
GROUPS_JSON = Path("logs/dedup_groups.json")

LAM_RE = re.compile(r"^(.*)_c(\d{3})_lam[0-9.]+$")

def group_key(stem: str) -> tuple[str, int] | None:
    m = LAM_RE.match(stem)
    if not m:
        return None
    return m.group(1), int(m.group(2))

def md5(path: Path, buf: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(buf)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def main() -> int:
    t0 = time.time()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    wavs = sorted(INPUT_DIR.glob("*.wav"))
    print(f"Scanning {len(wavs)} input WAVs in {INPUT_DIR} ...", flush=True)

    groups: dict[tuple[str, int], list[Path]] = defaultdict(list)
    orphans: list[Path] = []
    for p in wavs:
        k = group_key(p.stem)
        if k is None:
            orphans.append(p)
        else:
            groups[k].append(p)

    print(f"  {len(groups)} unique (track, clip_idx) groups, "
          f"{sum(len(v) for v in groups.values())} files, "
          f"{len(orphans)} orphans (no lam suffix).", flush=True)

    with open(LOG, "w") as flog:
        flog.write(f"# dedup verify @ {time.ctime()}\n")
        flog.write(f"# input_dir: {INPUT_DIR}  n_files: {len(wavs)}  "
                   f"n_groups: {len(groups)}\n\n")

        violations: list[tuple[str, int, list[str]]] = []
        total = len(groups)
        for i, (key, paths) in enumerate(sorted(groups.items())):
            if len(paths) == 1:
                continue
            hashes = {md5(p) for p in paths}
            if len(hashes) != 1:
                violations.append((key[0], key[1], [str(p.name) for p in paths]))
                flog.write(f"VIOLATION: {key} has {len(hashes)} distinct md5s "
                           f"across {len(paths)} files\n")
                for p in paths:
                    flog.write(f"    {md5(p)}  {p.name}\n")
            if (i + 1) % 20 == 0 or i == total - 1:
                dt = time.time() - t0
                rate = (i + 1) / dt
                eta = (total - i - 1) / rate if rate > 0 else 0
                print(f"  [{100*(i+1)/total:5.1f}%] ({i+1}/{total}) "
                      f"elapsed={dt:.0f}s eta={eta:.0f}s  "
                      f"violations={len(violations)}", flush=True)

        flog.write(f"\n# summary: {len(violations)} violations out of "
                   f"{len(groups)} groups\n")

    GROUPS_JSON.write_text(json.dumps(
        {f"{k[0]}__c{k[1]:03d}": sorted(p.name for p in v)
         for k, v in groups.items()},
        indent=2))

    print(f"\nDONE in {time.time()-t0:.1f}s")
    print(f"  groups:    {len(groups)}")
    print(f"  total:     {sum(len(v) for v in groups.values())}")
    print(f"  orphans:   {len(orphans)}")
    print(f"  violations:{len(violations)}")
    print(f"  log:       {LOG}")
    print(f"  groups:    {GROUPS_JSON}")
    if violations:
        print("\n  [!] VIOLATIONS FOUND — do NOT dedup without manual review.",
              flush=True)
        return 1
    print("\n  [OK] safe to dedup: every group is internally byte-identical.",
          flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
