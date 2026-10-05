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
| TDA backend | giotto-tda |
| FFD backend | CPU fallback (no CUDA) |

| Metric | Focal Loss | Log-loss control |
|---|---:|---:|
| Pearson IC | 0.073292 | 0.095368 |
| Rank IC | 0.039856 | 0.036762 |
| ROC-AUC | 0.531075 | 0.528676 |
| PR-AUC | 0.244568 | 0.242809 |
| Log loss | 0.516779 | 0.561761 |

The supported-environment GitHub Actions run used Python 3.11.16 with the declared pinned stack and Giotto-TDA 0.6.2. The complete machine-generated evidence was uploaded as a CI artifact and then used to refresh the committed raster assets.

## Extended experiments

The evidence bundle evaluates five feature/objective variants, six synthetic generator seeds, five chronological walk-forward folds, a label-permutation placebo, Focal Loss gradient/Hessian finite-difference checks, malformed L2 rejection, point-cloud translation invariance, causal FFD prefix invariance, and descriptive L2/topology diagnostics.

The corresponding raster evidence is kept under `../assets/figures/`, with readable terminal and report captures under `../assets/screenshots/`.

## Interpretation

The six-seed results are intentionally variable and the walk-forward sequence includes both negative and positive folds. In the supported-environment run, Focal Loss does not dominate the log-loss control on every metric; this is preserved rather than hidden. These synthetic results are therefore not used to claim a stable trading relationship. The historical 0.064 resume IC remains a reproduction target requiring the intended real L2 dataset and supported TDA/accelerator environment.

The supported-environment validation record is in `CI_VALIDATION.md`, while the extended experiment details are in `EXPERIMENTAL_VALIDATION.md`.
