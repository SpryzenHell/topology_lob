# Experimental validation record

All results below are deterministic synthetic-data engineering validation. They are not market-data claims and do not establish the historical OOS Information Coefficient target.

## Main run

Events: **6,000**; model rows: **5,735**; train/test: **3,994/1,721**; purge: **20**; selected d: **0.1**; TDA backend: **exact-vr-gf2-fallback**.

| Metric | Focal | Log-loss control |
|---|---:|---:|
| Pearson IC | 0.120444 | 0.116689 |
| Rank IC | 0.099255 | 0.104940 |
| ROC-AUC | 0.579135 | 0.579952 |
| PR-AUC | 0.263428 | 0.254891 |
| Log loss | 0.501762 | 0.562889 |

## Ablation

| Variant | Pearson IC | Rank IC | ROC-AUC | PR-AUC | Log loss |
|---|---:|---:|---:|---:|---:|
| micro_only | 0.179593 | 0.157019 | 0.574446 | 0.289559 | 0.659387 |
| micro_plus_fd | 0.091214 | 0.053805 | 0.534220 | 0.240239 | 0.510816 |
| micro_plus_topology | 0.142977 | 0.137714 | 0.615018 | 0.331311 | 0.635875 |
| full_focal | 0.120444 | 0.099255 | 0.579135 | 0.263428 | 0.501762 |
| full_logloss | 0.116689 | 0.104940 | 0.579952 | 0.254891 | 0.562889 |

## Seed sensitivity

- 2025: Pearson IC +0.120444, Rank IC +0.099255, d=0.1
- 2026: Pearson IC +0.086524, Rank IC +0.054994, d=0.4
- 2027: Pearson IC +0.081958, Rank IC +0.046601, d=0.5
- 2028: Pearson IC +0.047487, Rank IC -0.003027, d=0.1
- 2029: Pearson IC +0.283301, Rank IC +0.240703, d=0.5
- 2030: Pearson IC +0.013827, Rank IC -0.012470, d=0.3

## Walk-forward

- Fold 1: d=0.0, ADF p=0.04778, Focal Pearson=-0.121312, Log-loss Pearson=-0.120106
- Fold 2: d=0.1, ADF p=0.002296, Focal Pearson=+0.122285, Log-loss Pearson=+0.119397
- Fold 3: d=0.2, ADF p=0.001492, Focal Pearson=+0.227274, Log-loss Pearson=+0.216254
- Fold 4: d=0.2, ADF p=0.02628, Focal Pearson=+0.246110, Log-loss Pearson=+0.229193
- Fold 5: d=0.3, ADF p=0.002521, Focal Pearson=+0.201433, Log-loss Pearson=+0.193034

## Placebo and correctness

- Label-permutation placebo Pearson IC: **-0.009044**
- Gradient max error: **7.409e-12**
- Hessian max error: **1.059e-11**
- Malformed L2 rejection: **duplicate timestamps, crossed/locked book, negative size all rejected**
- Common-price translation error in the point-cloud representation: **0**
- Causal FFD prefix error after changing only future observations: **0**

The six-seed and walk-forward analyses are reported specifically to avoid using one synthetic seed as evidence of a stable market relationship. The repository preserves the historical 0.064 resume IC as a reproduction target only.
