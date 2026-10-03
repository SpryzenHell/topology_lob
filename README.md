# Topology LOB

Quantitative research application for Level-2 limit-order-book topology, fractional differentiation, and rare-event prediction.

The root repository retains the three upstream codebases named in the supplied project specification. The maintainable application layer lives under research/topology_lob/.

## Pipeline

L2 snapshots -> schema validation -> microstructure features -> liquidity-support point clouds -> Vietoris-Rips persistence -> Betti-0 / Betti-1 and H1 features -> train-only ADF selection of fractional-differencing order -> Numba CUDA or explicit CPU fallback -> train-only class rebalancing -> XGBoost custom Focal Loss -> chronological purged holdout -> expanding walk-forward -> IC / rank IC / ROC-AUC / PR-AUC.

## Quick start

    cd research/topology_lob
    python -m pip install -e ".[tda,dev]"
    python run.py demo --config configs/demo.json --out results/demo
    python run.py ablation --config configs/demo.json --out results/ablation
    python run.py walk-forward --config configs/demo.json --out results/walk_forward.json
    python run.py ffd-benchmark --events 200000 --width 256 --d 0.45

For real L2 data, use --require-gtda. For the GPU FFD claim, also use --require-cuda. Both options are fail-closed.

## Provenance

The supplied configuration combines giotto-ai/giotto-tda, artemmavrin/focal-loss, and scikit-learn-contrib/imbalanced-learn. The application uses documented public APIs rather than the mechanically prefixed vendored modules.

## Results policy

The resume target of 0.064 Information Coefficient is kept as a target/reference only. The code never hard-codes it as an achieved result. Synthetic results are labelled as synthetic. New application commits use the actual author/date rather than rewriting history.