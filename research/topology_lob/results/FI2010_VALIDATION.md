# FI-2010 benchmark validation

This record is based on a successful GitHub Actions execution of the repository's manual FI-2010 workflow. It uses the public FI-2010 Decimal-Precision benchmark mirror documented by the project. FI-2010 is used here as a benchmark input; this result is not presented as the project's original exchange-feed experiment.

## Execution record

| Item | Value |
|---|---:|
| Workflow | topology-lob-fi2010 |
| Run number | 116 |
| Run ID | 37296970151 |
| Train events | 254,750 |
| Held-out segments | 55,478 / 52,172 / 31,937 |
| Combined events | 394,337 |
| Topology clouds | 395 |
| Selected d | 0.3 |
| TDA backend | giotto-tda 0.6.2 |
| FFD backend | CPU fallback (no CUDA) |
| Horizon | 10 events |
| Purge | 50 events |
| Label threshold | 0.0005 |

## Daily full-stack metrics

| Segment | Pearson IC | Rank IC | ROC-AUC | PR-AUC | Log loss |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.007156 | 0.173629 | 0.723816 | 0.261587 | 0.636420 |
| 9 | -0.002349 | 0.144605 | 0.691614 | 0.224680 | 0.645734 |
| 10 | 0.000786 | 0.161694 | 0.709854 | 0.253142 | 0.641049 |

The focal model's mean daily Pearson IC is **0.001864** with standard deviation **0.004843**. Mean daily Rank IC is **0.159976** with standard deviation **0.014588**. The pooled Pearson IC is **0.002023** and pooled Rank IC is **0.159088**.

The difference between Pearson and Rank IC is retained deliberately. It shows that a rank-order classification signal can coexist with a much weaker linear relationship to the continuous forward return target.

## Feature ablation

| Variant | Pooled Pearson IC | Pooled Rank IC | Pooled ROC-AUC |
|---|---:|---:|---:|
| Microstructure only | 0.003538 | 0.170967 | 0.702435 |
| Microstructure + fractional differentiation | 0.002793 | 0.158005 | 0.705547 |
| Microstructure + topology | 0.002955 | 0.174857 | 0.703799 |
| Full Focal Loss | 0.001826 | 0.160033 | 0.709805 |
| Full log-loss | 0.001184 | 0.162941 | 0.707911 |

The full feature stack obtains the highest pooled ROC-AUC among these variants, but not the highest Pearson or Rank IC. The repository keeps these distinctions visible rather than selecting only the most favorable metric.

## Stationarity selection

The ADF sweep selects **d = 0.3** from the training stream. At d = 0.2 the ADF p-value is **0.087446**; at d = 0.3 it falls to **0.001702**. The correlation between the transformed series and price remains **0.997563** at the selected order.

## Target and claim boundary

The repository stores **0.064** as the historical resume reproduction target only. The FI-2010 benchmark produces a pooled continuous-return Pearson IC of **0.002023** for the focal model in this verified run, so this benchmark does not substantiate the 0.064 claim.

The original L2 claim remains dependent on the intended raw L2 dataset and its corresponding hardware/data environment.
