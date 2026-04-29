# Rescue v3 — Bootstrap-CI Aggregated Results

Each (run, u) cell shows **mean [95% CI lo, hi]** over per-recording metrics, with 1000-sample percentile bootstrap resampled at the clip level. `status=COLLAPSED` means ≥50% of clips had `|velocity_tv_change_pct| ≥ 99.5` (silent/constant output); treat those rows as failed runs.

| run | u | n | status | drift_c | jsd | vel_tv% | ΔA | raga-pres % |
|---|---|---|---|---|---|---|---|---|
| proposed_v2 | 0.0 | 2 | OK | 333.2 [66.4, 600.0] | 0.314 [0.150, 0.479] | 30.9 [-69.2, 130.9] | - | - |
| proposed_v2 | 0.3 | 2 | OK | 51.7 [3.3, 100.0] | 0.182 [0.123, 0.241] | -100.0 [-100.0, -100.0] | - | - |
| proposed_v2 | 0.6 | 2 | OK | 51.7 [3.3, 100.0] | 0.182 [0.123, 0.241] | -100.0 [-100.0, -100.0] | - | - |
| proposed_v2 | 0.9 | 2 | OK | 51.7 [3.3, 100.0] | 0.182 [0.123, 0.241] | -100.0 [-100.0, -100.0] | - | - |
| baseline_dsp | - | 2 | OK | 50.0 [33.3, 66.7] | 0.042 [0.032, 0.052] | 38.5 [4.1, 73.0] | - | - |
