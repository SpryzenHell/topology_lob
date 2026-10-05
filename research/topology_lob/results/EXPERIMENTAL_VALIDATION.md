# Extended experimental validation

All results below are deterministic synthetic-data engineering validation. They are not market-data claims and do not establish the historical OOS Information Coefficient target.

## Supported CI environment

Python 3.11.16; NumPy 1.26.4; pandas 2.2.2; SciPy 1.11.4; scikit-learn 1.3.2; statsmodels 0.14.4; XGBoost 2.1.4; imbalanced-learn 0.12.4; Numba 0.59.1; Matplotlib 3.8.4; Giotto-TDA 0.6.2.

The core test suite passed **11 tests in 2.51 seconds**. The synthetic demo exercised the real Giotto-TDA path, and the extended evidence validation, C++ build, C++ benchmark, and raster asset validation all passed.

## Main run

Events: **6,000**; model rows: **5,735**; train/test: **3,994/1,721**; purge: **20**; selected d: **0.1**; TDA backend: **giotto-tda**.

| Metric | Focal | Log-loss control |
|---|---:|---:|
| Pearson IC | 0.073292 | 0.095368 |
| Rank IC | 0.039856 | 0.036762 |
| ROC-AUC | 0.531075 | 0.528676 |
| PR-AUC | 0.244568 | 0.242809 |
| Log loss | 0.516779 | 0.561761 |

## Five-way ablation

| Variant | Pearson IC | Rank IC | ROC-AUC | PR-AUC | Log loss |
|---|---:|---:|---:|---:|---:|
| micro_only | 0.179593 | 0.157019 | 0.574446 | 0.289559 | 0.659387 |
| micro_plus_fd | 0.091214 | 0.053805 | 0.534220 | 0.240239 | 0.510816 |
| micro_plus_topology | 0.110662 | 0.087276 | 0.571300 | 0.283838 | 0.651177 |
| full_focal | 0.073292 | 0.039856 | 0.531075 | 0.244568 | 0.516779 |
| full_logloss | 0.095368 | 0.036762 | 0.528676 | 0.242809 | 0.561761 |

## Six-seed sensitivity

| Seed | Selected d | Pearson IC | Rank IC |
|---:|---:|---:|---:|
| 2025 | 0.1 | 0.073292 | 0.039856 |
| 2026 | 0.4 | 0.018178 | 0.003513 |
| 2027 | 0.5 | 0.109719 | 0.047696 |
| 2028 | 0.1 | 0.055979 | 0.004799 |
| 2029 | 0.5 | 0.307903 | 0.252003 |
| 2030 | 0.3 | -0.015051 | -0.047488 |

Across the six seeds, Pearson IC has mean **0.09167**, standard deviation **0.11444**, minimum **-0.01505**, and maximum **0.30790**.

## Five-fold walk-forward

| Fold | d | ADF p | Focal Pearson | Log-loss Pearson |
|---:|---:|---:|---:|---:|
| 1 | 0.0 | 0.04778 | -0.054441 | -0.059235 |
| 2 | 0.1 | 0.002296 | 0.126380 | 0.186337 |
| 3 | 0.2 | 0.001492 | 0.133533 | 0.143274 |
| 4 | 0.2 | 0.02628 | 0.202818 | 0.145676 |
| 5 | 0.3 | 0.002521 | 0.181259 | 0.205566 |

Focal Pearson IC is positive in **4 of 5** chronological folds. The mean is **0.11791** with standard deviation **0.10154**.

## Label-permutation placebo

The real full-stack Pearson IC is **0.073292**. After permuting the training labels, the placebo Pearson IC is **-0.028421** and placebo ROC-AUC is **0.472307**.

This is a sanity control against obtaining a similar signal from arbitrary labels under the same training pipeline.

## Numerical and structural correctness

- Focal gradient maximum finite-difference error: **7.409e-12**
- Focal Hessian maximum finite-difference error: **1.059e-11**
- Duplicate timestamp rejection: **PASS**
- Crossed/locked book rejection: **PASS**
- Negative-size rejection: **PASS**
- Common-price translation invariance error: **0**
- Causal FFD prefix invariance error: **0**

## Topology diagnostic

For the deterministic synthetic cloud set, H1 at the configured 0.55 radius has mean **0.002694**, median **0**, maximum **1**, and Pearson correlation **-0.103443** with the 10-event forward log return. The large zero mass is a useful diagnostic: the fallback/production topology features do not imply that every L2 state contains a persistent loop.

## Performance

The supported CI CPU runner measured the standalone C++ FFD implementation at **5,247,526 events/second** for 200,000 events, width 256 and d=0.45. This is a micro-benchmark, not a trading latency claim.

The committed dashboards are generated from this supported-environment run. The historical 0.064 resume IC remains an external reproduction target.
