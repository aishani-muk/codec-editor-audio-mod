# Rescue v3 — Success gate @ u=0.6

Source: `results/v3_table.csv`

Verdicts: **PASS** = mean + adverse-side 95% CI clears the threshold. **MARGN** = mean clears but CI doesn't. **FAIL** = mean fails.

## proposed_v2  (u=0.6, status=OK, n=2)

| criterion | mean | CI bound | verdict |
|---|---|---|---|
| ΔA ≤ -0.40 |     —    |     —    | — |
| drift ≤ 100 c |  +51.66 |  +99.99 | PASS |
| raga-pres ≥ 70 % |     —    |     —    | — |
| |ΔV| ≤ 0.50 |     —    |     —    | — |
| |vel_tv%| ≤ 50 | +100.00 | +100.00 | FAIL |

