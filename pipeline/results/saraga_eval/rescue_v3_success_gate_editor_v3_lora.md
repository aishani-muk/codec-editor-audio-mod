# Rescue v3 — Success gate @ u=0.6

Source: `results/rescue_v3_table_editor_v3_lora.csv`

Verdicts: **PASS** = mean + adverse-side 95% CI clears the threshold. **MARGN** = mean clears but CI doesn't. **FAIL** = mean fails.

## editor_v3_lora  (u=0.6, status=OK, n=30)

| criterion | mean | CI bound | verdict |
|---|---|---|---|
| ΔA ≤ -0.40 |   -0.05 |   -0.04 | FAIL |
| drift ≤ 100 c |   +1.69 |   +5.08 | PASS |
| raga-pres ≥ 70 % |     —    |     —    | — |
| |ΔV| ≤ 0.50 |   +0.03 |   +0.04 | PASS |
| |vel_tv%| ≤ 50 |   +0.01 |   +0.02 | PASS |

**Overall: FAIL**

