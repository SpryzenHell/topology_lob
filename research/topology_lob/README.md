# Topology LOB — L2 Liquidity Topology Research Engine

Topology LOB is an end-to-end quantitative research pipeline for the project specification: Level-2 liquidity geometry → Vietoris–Rips persistence → Betti features → fractional differentiation → custom XGBoost Focal Loss → purged OOS / walk-forward evaluation.

## What is implemented

- **TDA:** L2 snapshots are converted into liquidity-support point clouds using price distance and log depth. Exact mode uses `giotto-tda` `VietorisRipsPersistence` in homology dimensions 0 and 1, producing Betti-0 / Betti-1 curves plus H1 persistence summaries.
- **Stationarity:** log-mid prices are fractionally differentiated with a fixed-width causal filter. The smallest candidate order passing ADF on the training prefix is selected; Numba CUDA is used when available, otherwise the backend is explicitly reported as CPU fallback.
- **Rare-event ML:** a custom binary Focal Loss gradient/Hessian is supplied to XGBoost, with `RandomOverSampler` applied only to the training slice. The control model uses the same features/split with standard log loss.
- **Evaluation:** chronological holdout with a purge gap, plus expanding walk-forward folds that re-select fractional-differencing order from each training window.

## Run

```bash
cd research/topology_lob
python -m pip install -e .[tda,dev]
python run.py demo --config configs/demo.json --out results/demo
python run.py ablation --config configs/demo.json --out results/ablation
python run.py walk-forward --config configs/demo.json --out results/walk_forward.json
python run.py ffd-benchmark --events 200000 --width 256 --d 0.45
```

The default demo is synthetic. It is an integration test, not market data.

## Real L2 data

Expected columns:

```text
timestamp,
bid_price_1,bid_size_1,ask_price_1,ask_size_1,
...,
bid_price_N,bid_size_N,ask_price_N,ask_size_N
```

Run exact persistent homology with:

```bash
python run.py demo --config configs/demo.json --data /absolute/path/to/l2.csv --require-gtda --out results/real_l2
```

For the GPU FFD requirement, add `--require-cuda`; the command exits unless a CUDA backend actually runs.

## Experimental hygiene

The requested resume metric `0.064` is stored only as a target/reference. The application never substitutes it for a measured result. Synthetic results and CPU fallbacks are labelled as such. A resume-grade number requires the intended real L2 data, locked evaluation protocol, and the exact backend/hardware required by the bullet.

The research application deliberately lives outside the mechanically renamed upstream library sources retained in the fork. See `THIRD_PARTY.md` for provenance/licensing.