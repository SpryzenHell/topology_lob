from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fi2010 import load_raw, raw40_to_canonical
from topology_lob import (
    fractional_diff_gpu,
    fit_focal_xgb,
    fit_logloss_xgb,
    make_point_clouds,
    microstructure_features,
    persistent_features,
    select_stationary_d,
)


def corr(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def evaluate_model(model, X, y, future):
    dm = xgb.DMatrix(X)
    prob = model.predict(dm)
    margin = model.predict(dm, output_margin=True)
    roc = None
    if np.unique(y).size == 2:
        from sklearn.metrics import roc_auc_score
        roc = float(roc_auc_score(y, prob))
    return {
        "probability_pearson_ic": corr(prob, future),
        "margin_pearson_ic": corr(margin, future),
        "rank_ic": float(spearmanr(margin, future).statistic),
        "roc_auc": roc,
    }


def prepare(args):
    train = raw40_to_canonical(load_raw(args.train).features)
    tests = [raw40_to_canonical(load_raw(path).features) for path in args.test]
    combined = pd.concat([train, *tests], ignore_index=True)
    train_events = len(train)

    micro = microstructure_features(combined, depth_levels=5)
    d, stationarity = select_stationary_d(
        micro["log_mid"].to_numpy()[:train_events],
        np.linspace(0.0, 1.0, 11),
        0.05,
    )
    fd, gpu_used, fd_backend = fractional_diff_gpu(
        micro["log_mid"].to_numpy(), d, args.ffd_width
    )
    if args.require_cuda and not gpu_used:
        raise SystemExit("CUDA required but not exercised")

    micro["fracdiff"] = fd
    base = [
        c for c in micro.columns
        if c not in {"mid", "log_mid", "interarrival_us"}
    ]

    topo_index = np.arange(50, len(combined) - args.horizon, args.tda_stride)
    radii = np.asarray([0.5, 1.0, 1.5, 2.0, 2.5])
    topology, topo_meta = persistent_features(
        make_point_clouds(combined, topo_index, levels=10),
        radii,
        require_gtda=args.require_gtda,
        n_jobs=-1,
    )
    topo_df = pd.DataFrame(
        topology, index=topo_index, columns=topo_meta["feature_names"]
    ).reindex(range(len(combined))).ffill()

    future = np.full(len(combined), np.nan)
    logs = micro["log_mid"].to_numpy()
    future[:-args.horizon] = logs[args.horizon:] - logs[:-args.horizon]
    y = np.full(len(combined), -1)
    finite = np.isfinite(future)
    y[finite] = (future[finite] > args.label_threshold).astype(int)

    arrays = {}
    micro_cols = [c for c in base if c != "fracdiff"]
    arrays["micro_only"] = micro[micro_cols].to_numpy(float)
    arrays["micro_plus_fd"] = micro[base].to_numpy(float)
    arrays["micro_plus_topology"] = np.hstack([
        micro[micro_cols].to_numpy(float),
        topo_df.to_numpy(float),
    ])
    arrays["full"] = np.hstack([
        micro[base].to_numpy(float),
        topo_df.to_numpy(float),
    ])

    results = {
        "data_source": "FI-2010 Decimal-Precision",
        "train_events": train_events,
        "test_events": [len(x) for x in tests],
        "selected_d": float(d),
        "stationarity": stationarity,
        "tda": topo_meta,
        "fractional_diff_backend": fd_backend,
        "gpu_used": bool(gpu_used),
        "experiments": {},
    }

    valid = None
    for mat in arrays.values():
        current = np.isfinite(mat).all(axis=1) & np.isfinite(future) & (y >= 0)
        valid = current if valid is None else valid & current

    event_indices = np.flatnonzero(valid)
    train_mask = event_indices < train_events
    train_pos = np.flatnonzero(train_mask)
    train_pos = train_pos[:max(0, len(train_pos) - args.purge)]

    cursor = train_events
    test_positions_by_day = []
    for test_frame in tests:
        end = cursor + len(test_frame)
        test_positions_by_day.append(np.flatnonzero(
            (event_indices >= cursor) & (event_indices < end)
        ))
        cursor = end

    pooled_test = np.concatenate(test_positions_by_day)

    for name, mat in arrays.items():
        X = mat[valid]
        yv = y[valid]
        fv = future[valid]
        focal = fit_focal_xgb(X[train_pos], yv[train_pos], seed=2010)
        baseline = fit_logloss_xgb(X[train_pos], yv[train_pos], seed=2010)

        days = []
        for day, pos in enumerate(test_positions_by_day, start=8):
            days.append({
                "day": day,
                "focal": evaluate_model(focal, X[pos], yv[pos], fv[pos]),
                "logloss": evaluate_model(baseline, X[pos], yv[pos], fv[pos]),
            })

        results["experiments"][name] = {
            "days": days,
            "pooled_focal": evaluate_model(
                focal, X[pooled_test], yv[pooled_test], fv[pooled_test]
            ),
            "pooled_logloss": evaluate_model(
                baseline, X[pooled_test], yv[pooled_test], fv[pooled_test]
            ),
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", nargs=3, required=True)
    ap.add_argument("--out", default="results/fi2010_ablation.json")
    ap.add_argument("--tda-stride", type=int, default=1000)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--label-threshold", type=float, default=0.0005)
    ap.add_argument("--purge", type=int, default=50)
    ap.add_argument("--ffd-width", type=int, default=256)
    ap.add_argument("--require-gtda", action="store_true")
    ap.add_argument("--require-cuda", action="store_true")
    args = ap.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
