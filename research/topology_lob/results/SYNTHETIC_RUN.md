# Synthetic run record

These are measured local CPU results from the deterministic synthetic integration test. They are not market-data evidence.

- events: 1200
- model samples: 935
- train samples: 634
- test samples: 281
- purge gap: 20
- selected d: 0.3
- TDA backend: threshold-graph-fallback because Giotto-TDA was not installed locally
- FFD backend: cpu-fallback:no-cuda
- Focal Pearson IC: 0.2046176813
- Focal rank IC: 0.2046577032
- Focal ROC-AUC: 0.5293026133
- Log-loss baseline Pearson IC: 0.2375834644

## Ablation Pearson IC

microstructure only: -0.1006036138
microstructure + fractional differentiation: 0.1024243121
microstructure + topology: 0.2881402954
full feature set: 0.2046176813

## Walk-forward

Five synthetic folds were executed. Selected d values were 0.3, 0.3, 0.3, 0.2, 0.3. Focal Pearson ICs were approximately 0.0458, 0.0809, 0.3610, 0.2874, and 0.4071.

## C++ benchmark

The 200,000-event, width-256 C++ CPU benchmark completed at about 6.17M events/s in the final local run.

The standalone CUDA benchmark is included but was not executed in this CPU-only container.

## Resume target

The resume target is 0.064 OOS IC. This target is never substituted with a synthetic result.