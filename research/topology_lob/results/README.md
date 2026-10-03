# Results policy

Generated experiment output belongs here locally and should not be committed unless it is small, reproducible, and accompanied by the exact config and data provenance.

The resume target `0.064` is not a result file.

When a real-data experiment is completed, record:

- data source and exact source file names;
- normalization variant;
- row/event counts;
- exact train/test or anchored-fold boundaries;
- selected d per training fold;
- TDA backend and filtration parameters;
- CUDA availability and device;
- focal vs log-loss metrics;
- Pearson IC and Rank IC per fold;
- mean/std across folds;
- checksum/version of any prepared data file.

Never commit proprietary or licensed raw market data.