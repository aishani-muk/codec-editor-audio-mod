# Rescue v3 — Bootstrap-CI Aggregated Results

Each (run, u) cell shows **mean [95% CI lo, hi]** over per-recording metrics, with 1000-sample percentile bootstrap resampled at the clip level. `status=COLLAPSED` means ≥50% of clips had `|velocity_tv_change_pct| ≥ 99.5` (silent/constant output); treat those rows as failed runs.

| run | u | n | status | drift_c | jsd | vel_tv% | ΔA | raga-pres % |
|---|---|---|---|---|---|---|---|---|
| editor_v3_lora | 0.0 | 30 | OK | 1.7 [0.0, 5.1] | 0.000 [0.000, 0.000] | -0.0 [-0.0, 0.0] | -0.05 [-0.06, -0.04] | - |
| editor_v3_lora | 0.3 | 30 | OK | 1.7 [0.0, 5.1] | 0.000 [0.000, 0.000] | -0.0 [-0.0, 0.0] | -0.05 [-0.06, -0.04] | - |
| editor_v3_lora | 0.6 | 30 | OK | 1.7 [0.0, 5.1] | 0.000 [0.000, 0.000] | -0.0 [-0.0, 0.0] | -0.05 [-0.06, -0.04] | - |
| editor_v3_lora | 0.9 | 30 | OK | 1.7 [0.0, 5.1] | 0.000 [0.000, 0.000] | -0.0 [-0.0, 0.0] | -0.05 [-0.06, -0.04] | - |
| baseline_dsp_v3 | - | 30 | OK | 67.5 [23.0, 151.9] | 0.027 [0.022, 0.032] | 10.0 [2.7, 17.7] | 0.32 [0.21, 0.43] | - |
