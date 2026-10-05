# Real-data run guide

## FI-2010

FI-2010 is a standard high-frequency LOB benchmark with ten levels on the bid and ask sides and five Finnish equities over ten trading days. The raw representation has 40 LOB variables per event. The benchmark is widely used for short-horizon mid-price prediction research. See the paper and current benchmark documentation for exact dataset provenance.

The exact public data record is:
https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649

The public benchmark contains three normalization families. This project uses the Decimal-Precision representation for the TDA point-cloud stage because the topology should operate on economically meaningful price and depth values. Z-score data is useful for classification baselines but should not be treated as raw price/size geometry.

### Convert

After downloading the benchmark text files, run:

    python research/topology_lob/scripts/convert_fi2010.py \
      --input /path/to/Train_Dst_NoAuction_DecPre_CF_7.txt \
      --output research/topology_lob/data/FI2010_DecPre_train.csv

Then run:

    cd research/topology_lob
    python run.py demo --config configs/fi2010.json \
      --out results/fi2010 --require-gtda

For a CUDA FFD run, add:

    --require-cuda

The application will refuse to report the GPU path unless CUDA is actually exercised.

## Standard split

The FI-2010 file family also exposes anchored cross-validation folds. A current independent loader describes CF_7 as cumulative days 1–7 for training and Test_CF_7, Test_CF_8, and Test_CF_9 as the three single-day held-out portions corresponding to days 8–10. The project should use that convention when comparing against published FI-2010 baselines rather than taking only the single CF_7 test file.

Source:
https://github.com/karthik-1604/lob-midprice-forecasting/blob/master/data/loader.py

## Scope limitation

FI-2010 is a benchmark, not a modern production feed. Literature notes that its ten-day scope is insufficient by itself to establish months-long robustness. Treat the resulting score as a benchmark result, not a universal market alpha claim.