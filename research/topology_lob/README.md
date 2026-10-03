# Topology LOB — L2 Liquidity Topology Research Engine

This application implements the three technical claims in the project brief as one reproducible research pipeline.

1. Topological Data Analysis
   L2 snapshots are converted into a two-dimensional liquidity-support point cloud using price distance from mid and normalized log depth. With Giotto-TDA installed, VietorisRipsPersistence is run in homology dimensions 0 and 1. Betti curves, H1 maximum persistence, and persistence entropy are extracted. A small threshold-graph fallback exists only for local smoke tests and is explicitly labelled as non-persistent topology.

2. Fractional differentiation
   Candidate fractional-differencing orders are evaluated using the Augmented Dickey-Fuller test on the training prefix only. The smallest candidate passing the configured ADF threshold is selected. The resulting fixed-width causal filter can run with Numba CUDA, a NumPy CPU path, or the included C++ CPU / optional C++ CUDA benchmark.

3. Rare-event prediction
   XGBoost is trained with a hand-written binary Focal Loss gradient and Hessian. RandomOverSampler is applied only to the training slice after the chronological boundary is fixed. A standard binary-logloss XGBoost model is trained as the control.

4. Evaluation
   The default evaluation uses a chronological holdout with a purge gap. The walk-forward runner uses expanding windows and re-selects the fractional-differencing order inside each training window.

## Install

    cd research/topology_lob
    python -m pip install -e ".[tda,dev]"

For the exact TDA path, install the tda extra and run demo with --require-gtda.

## Synthetic smoke run

    python run.py demo --config configs/demo.json --out results/demo
    python run.py ablation --config configs/demo.json --out results/ablation
    python run.py walk-forward --config configs/demo.json --out results/walk_forward.json
    python run.py ffd-benchmark --events 200000 --width 256 --d 0.45

The included synthetic generator creates L2 geometry in which a rare liquidity void changes deeper book support and influences future returns. It is deterministic and is not market data.

## Real L2 data

Required columns:

    timestamp
    bid_price_1,bid_size_1,ask_price_1,ask_size_1
    ...
    bid_price_N,bid_size_N,ask_price_N,ask_size_N

Example:

    python run.py demo --data /absolute/path/to/l2.csv --config configs/demo.json --require-gtda --require-cuda --out results/real_l2

The command fails instead of silently substituting a topology or accelerator fallback.

## Artifacts

demo produces metrics.json, test_scores.csv, betti_curves.png, fractional_stationarity.png, and index.html.

## Resume metric

The project brief says OOS IC = 0.064. This repository records 0.064 only as target_resume_ic. Any final resume claim must be recomputed on the intended real L2 dataset with exact Giotto-TDA execution and the required CUDA environment.