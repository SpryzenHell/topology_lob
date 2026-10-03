# Real-data FI-2010 run

Official research-data record:
https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649

A commonly used mirror is the data/data.zip archive from the DeepLOB example repository. FI-2010 contains ten LOB levels for five Nasdaq Nordic stocks over ten trading days. The published benchmark is preprocessed/normalized; Decimal-Precision FI-2010 is not the original exchange feed.

## Download

    cd research/topology_lob
    python scripts/fetch_fi2010.py --out data/external/fi2010.zip --extract

The command prints a SHA-256 checksum.

## Convert

    python scripts/convert_fi2010.py --input data/external/fi2010/fi2010/Train_Dst_NoAuction_DecPre_CF_7.txt --output data/FI2010_DecPre_train.csv

The converter maps 40 raw LOB variables (ask price, ask size, bid price, bid size for ten levels) into the application schema. FI-2010 provides event order rather than exchange wall-clock timestamps, so the converter creates an explicit event-index timestamp.

## Benchmark

    python scripts/run_fi2010_benchmark.py --train data/external/fi2010/fi2010/Train_Dst_NoAuction_DecPre_CF_7.txt --test data/external/fi2010/fi2010/Test_Dst_NoAuction_DecPre_CF_7.txt data/external/fi2010/fi2010/Test_Dst_NoAuction_DecPre_CF_8.txt data/external/fi2010/fi2010/Test_Dst_NoAuction_DecPre_CF_9.txt --tda-stride 1000 --require-gtda --out results/fi2010_benchmark.json

Add --require-cuda to require the GPU fractional-difference implementation.

## Leakage controls

Forward-return labels are constructed separately within train and each test split, so test observations at the end of one held-out day never receive targets from the next day. The FFD transform is causal, d is selected from training only, and resampling/model fitting are training-only.

## Resume target

The historical target is IC=0.064. Record measured per-day and aggregate Pearson IC, Rank IC, ROC-AUC, PR-AUC, and the matched log-loss control. Never replace measured output with the target.