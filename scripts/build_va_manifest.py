"""Build a V/A-filtered manifest from results/va_cache.jsonl + the original
paired_edits manifest.

A pair (input, target) is kept iff
    dist_to_neutral_out < dist_to_neutral_in
where ``neutral = (valence=5, arousal=3)`` on the DEAM 1-9 scale.

Outputs:
    data/paired_edits/manifest_va_filtered.jsonl   (superset of original fields
                                                    + va_in / va_out / d_in /
                                                    d_out / moved_toward_neutral)
    logs/va_manifest_summary.json                  (retention stats, retention
                                                    by λ-bucket)

If the cache is only partially populated (scoring still running), we emit the
filtered manifest for the covered fraction anyway and note how many rows were
skipped in the summary.
"""
from __future__ import annotations
import json, math, re
from pathlib import Path
from collections import defaultdict, Counter

NEUTRAL_V = 5.0
NEUTRAL_A = 3.0

MANIFEST_IN = Path("data/paired_edits/manifest.jsonl")
CACHE = Path("results/va_cache.jsonl")
MANIFEST_OUT = Path("data/paired_edits/manifest_va_filtered.jsonl")
SUMMARY = Path("logs/va_manifest_summary.json")

LAM_RE = re.compile(r"^(.*)_c(\d{3})_lam[0-9.]+$")


def _input_key(stem: str) -> str:
    m = LAM_RE.match(stem)
    if not m:
        return f"in:{stem}"
    return f"in:{m.group(1)}__c{int(m.group(2)):03d}"


def _dist_to_neutral(v: float, a: float) -> float:
    return math.sqrt((v - NEUTRAL_V) ** 2 + (a - NEUTRAL_A) ** 2)


def _load_cache() -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(CACHE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = row.get("key")
            if key:
                out[key] = row
    return out


def main() -> int:
    if not CACHE.exists():
        print(f"ERROR: no cache at {CACHE}; run scripts/m2e_cpu_score.py first.")
        return 2

    cache = _load_cache()
    print(f"[va-manifest] loaded {len(cache)} cached scores from {CACHE}")

    kept = 0
    n_total = 0
    n_missing_in = 0
    n_missing_out = 0
    n_errored = 0
    per_lambda: dict[float, Counter] = defaultdict(Counter)

    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_IN) as fin, open(MANIFEST_OUT, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            n_total += 1
            stem = entry["stem"]
            lam = round(float(entry["lambda_val"]), 3)
            per_lambda[lam]["total"] += 1

            ikey = _input_key(stem)
            tkey = f"out:{stem}"
            row_in = cache.get(ikey)
            row_out = cache.get(tkey)
            if row_in is None or "error" in row_in:
                n_missing_in += 1
                per_lambda[lam]["missing_in"] += 1
                continue
            if row_out is None or "error" in row_out:
                n_missing_out += 1
                per_lambda[lam]["missing_out"] += 1
                continue
            if "error" in row_in or "error" in row_out:
                n_errored += 1
                per_lambda[lam]["errored"] += 1
                continue

            v_in, a_in = row_in["valence"], row_in["arousal"]
            v_out, a_out = row_out["valence"], row_out["arousal"]
            d_in = _dist_to_neutral(v_in, a_in)
            d_out = _dist_to_neutral(v_out, a_out)
            moved = d_out < d_in

            enriched = dict(entry)
            enriched.update({
                "valence_in": v_in, "arousal_in": a_in,
                "valence_out": v_out, "arousal_out": a_out,
                "dist_to_neutral_in": round(d_in, 4),
                "dist_to_neutral_out": round(d_out, 4),
                "moved_toward_neutral": bool(moved),
                "moods_in": row_in.get("moods", []),
                "moods_out": row_out.get("moods", []),
            })
            per_lambda[lam]["scored"] += 1
            if moved:
                fout.write(json.dumps(enriched) + "\n")
                kept += 1
                per_lambda[lam]["kept"] += 1

    summary = {
        "neutral": {"valence": NEUTRAL_V, "arousal": NEUTRAL_A},
        "input_manifest": str(MANIFEST_IN),
        "output_manifest": str(MANIFEST_OUT),
        "cache": str(CACHE),
        "n_total": n_total,
        "n_kept": kept,
        "retention_pct": round(100 * kept / n_total, 2) if n_total else 0,
        "n_missing_in": n_missing_in,
        "n_missing_out": n_missing_out,
        "n_errored": n_errored,
        "per_lambda": {
            str(k): dict(v) for k, v in sorted(per_lambda.items())
        },
    }
    SUMMARY.write_text(json.dumps(summary, indent=2))

    print(f"[va-manifest] total:         {n_total}")
    print(f"[va-manifest] scored:        "
          f"{n_total - n_missing_in - n_missing_out - n_errored}")
    print(f"[va-manifest] kept:          {kept}  "
          f"({summary['retention_pct']}% retention)")
    print(f"[va-manifest] missing in/out/err: "
          f"{n_missing_in}/{n_missing_out}/{n_errored}")
    print(f"[va-manifest] written:       {MANIFEST_OUT}")
    print(f"[va-manifest] summary:       {SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
