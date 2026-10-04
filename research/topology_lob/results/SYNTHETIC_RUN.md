# Synthetic integration record

This record documents a deterministic local execution of the application using the repository's synthetic L2 generator.

It is an engineering validation result. It is not market data and it is not used to support the resume Information Coefficient claim.

## Run

```bash
cd research/topology_lob
python run.py demo   --config configs/demo.json   --out results/demo
```

The validation environment did not have a CUDA device and did not have Giotto-TDA installed, so the run intentionally exercised the explicit CPU path and exact small-cloud VR fallback.

## Measured execution

| Item | Value |
|---|---:|
| Synthetic events | 6,000 |
| Model rows | 5,735 |
| Training rows | 3,964 |
| Test rows | 1,711 |
| Purge gap | 20 |
| Topology clouds | 594 |
| Selected fractional-differencing order | 0.4 |
| TDA backend | exact-vr-gf2-fallback |
| FFD backend | cpu-fallback:no-cuda |
| GPU used | false |

## Holdout metrics

| Metric | Focal Loss | Log-loss control |
|---|---:|---:|
| Pearson IC | 0.009943 | 0.039237 |
| Rank IC | 0.018416 | 0.045511 |
| ROC-AUC | 0.511312 | 0.542148 |
| PR-AUC | 0.061286 | 0.064195 |
| Log loss | 0.742249 | 0.218352 |

The scores come from the same chronological split. They are included so another developer can distinguish a functioning code path from an unverified performance claim.

## Standalone C++ benchmark

A separate local run of the included C++ CPU implementation used:

```text
events = 200,000
width  = 256
d      = 0.45
```

Measured throughput was approximately **7.26 million events/second** on the validation machine.

This is a micro-benchmark of the standalone implementation. It is not a trading-performance result.

## Missing real-data verification

A real FI-2010 or project-specific L2 run still requires the corresponding dataset files and, for the production topology path, an environment with Giotto-TDA installed. The repository contains the loaders and benchmark scripts; no real-data result is inserted here without an executed run.
