# Topology LOB experiment protocol

This document is the protocol to use before recording any market-data result on the resume.

## Feature construction

1. Start from one chronological L2 stream.
2. Never shuffle event order.
3. Build microstructure features from the state at time t only.
4. Build the TDA point cloud from the same state at t. Do not use future book states to construct the point cloud.
5. Apply fractional differencing causally. Select d on the training prefix only.
6. Freeze d before touching the test interval.
7. Fit resampling only on the training data.
8. Fit the XGBoost model only on the training data.

## Split

Use a chronological test block with a purge gap at least as large as the forecast horizon. For FI-2010 benchmarking, also report the anchored forward folds exposed by the dataset.

## Primary signal metric

The project's headline metric is the Pearson Information Coefficient between the model score and the realized forward return on the untouched test interval:

    IC = corr(score_t, r_{t,t+h})

Report the mean and standard deviation across walk-forward folds. A single fold should not be called a robust OOS result.

Also report Rank IC, ROC-AUC, PR-AUC, positive rate, and the matched log-loss control.

## Focal Loss ablation

Run the same feature set and split with:
- custom Focal Loss;
- standard binary log-loss.

Do not change the feature set between these two rows.

## Topology ablation

Run:
- microstructure only;
- microstructure + fractional differentiation;
- microstructure + topology;
- full.

The goal is to identify whether the Betti/persistence block adds predictive information beyond ordinary LOB variables.

## Resume decision

The historical resume target is 0.064 OOS IC. Treat it as a reference to reproduce, not a desired output to optimize toward. If the measured result differs, keep the measured result and update the resume accordingly.