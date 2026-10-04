# Synthetic integration record

This record documents a deterministic local execution of the current application using the repository's synthetic L2 generator.

It is an engineering validation result. It is not market data and it is not used to support the resume Information Coefficient claim.

## Run

```bash
cd research/topology_lob
python run.py demo --config configs/demo.json --out results/demo
```

The validation environment did not have a CUDA device and did not have Giotto-TDA installed, so this run exercised the explicit CPU path and the exact small-cloud Vietoris-Rips fallback.

## Measured execution

| Item | Value |
|---|---:|
| Synthetic events | 6,000 |
| Model rows | 5,735 |
| Training rows | 3,994 |
| Test rows | 1,721 |
| Purge gap | 20 |
| Topology clouds | 743 |
| Selected fractional-differencing order | 0.1 |
| TDA backend | exact-vr-gf2-fallback |
| FFD backend | cpu-fallback:no-cuda |
| GPU used | false |

## Holdout metrics

| Metric | Focal Loss | Log-loss control |
|---|---:|---:|
| Pearson IC | 0.131209 | 0.116689 |
| Rank IC | 0.101913 | 0.104940 |
| ROC-AUC | 0.581376 | 0.579952 |
| PR-AUC | 0.263163 | 0.254891 |
| Log loss | 0.501007 | 0.562889 |

The scores come from the same chronological split. They document a functioning implementation and are not presented as a market-data result.

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
