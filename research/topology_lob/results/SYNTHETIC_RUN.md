# Synthetic validation record

This record is an engineering validation run from the deterministic synthetic L2 generator. It is not historical market-data evidence and is not used to substantiate the resume target of OOS Information Coefficient 0.064.

## Main integration run

| Item | Value |
|---|---:|
| Synthetic events | 6,000 |
| Model rows | 5,735 |
| Training rows | 3,994 |
| Test rows | 1,721 |
| Purge gap | 20 |
| Topology clouds | 743 |
| Selected d | 0.1 |
| TDA backend | exact-vr-gf2-fallback |
| FFD backend | CPU fallback (no CUDA) |

| Metric | Focal Loss | Log-loss control |
|---|---:|---:|
| Pearson IC | 0.120444 | 0.116689 |
| Rank IC | 0.099255 | 0.104940 |
| ROC-AUC | 0.579135 | 0.579952 |
| PR-AUC | 0.263428 | 0.254891 |
| Log loss | 0.501762 | 0.562889 |

The local run used Python 3.13, NumPy 2.3.5, pandas 2.2.3, XGBoost 3.1.3 and Matplotlib 3.10.8. Python 3.13 is outside the project's declared Python 3.10-3.12 support range, so this is a local fallback-engineering snapshot rather than the supported-environment release result.

## Extended experiments

The evidence bundle evaluates five feature/objective variants, six synthetic generator seeds, five chronological walk-forward folds, a label-permutation placebo, Focal Loss gradient/Hessian finite-difference checks, malformed L2 rejection, point-cloud translation invariance, causal FFD prefix invariance, and descriptive L2/topology diagnostics.

The corresponding raster evidence is kept under `../assets/figures/`, with readable terminal and report captures under `../assets/screenshots/`.

## Interpretation

The six-seed results are intentionally variable and the walk-forward sequence includes both negative and positive folds. Topology improves some synthetic classification/IC measures in ablations but the full stack does not dominate every metric. These synthetic results are therefore not used to claim a stable trading relationship. The historical 0.064 resume IC remains a reproduction target requiring the intended real L2 dataset and supported TDA/accelerator environment.
