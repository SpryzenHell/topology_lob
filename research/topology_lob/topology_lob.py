from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


@dataclass(frozen=True)
class LOBSchema:
    timestamp: str = "timestamp"
    bid_price_prefix: str = "bid_price_"
    bid_size_prefix: str = "bid_size_"
    ask_price_prefix: str = "ask_price_"
    ask_size_prefix: str = "ask_size_"


class LOBValidationError(ValueError):
    pass


def level_columns(df: pd.DataFrame, schema: LOBSchema = LOBSchema()) -> int:
    levels = []
    for column in df.columns:
        if column.startswith(schema.bid_price_prefix):
            try:
                levels.append(int(column[len(schema.bid_price_prefix):]))
            except ValueError:
                pass
    if not levels:
        raise LOBValidationError("No bid_price_N columns found")
    n = max(levels)
    expected = []
    for i in range(1, n + 1):
        expected += [f"bid_price_{i}", f"bid_size_{i}", f"ask_price_{i}", f"ask_size_{i}"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise LOBValidationError(f"Missing L2 columns: {missing[:8]}")
    return n


def validate_lob(df: pd.DataFrame, schema: LOBSchema = LOBSchema()) -> pd.DataFrame:
    n = level_columns(df, schema)
    out = df.copy()
    out[schema.timestamp] = pd.to_datetime(out[schema.timestamp], utc=True, errors="coerce")
    if out[schema.timestamp].isna().any():
        raise LOBValidationError("Invalid timestamp")
    out = out.sort_values(schema.timestamp, kind="stable").reset_index(drop=True)
    if out[schema.timestamp].duplicated().any():
        raise LOBValidationError("Timestamps must be unique")
    numeric = []
    for i in range(1, n + 1):
        numeric += [f"bid_price_{i}", f"bid_size_{i}", f"ask_price_{i}", f"ask_size_{i}"]
    for column in numeric:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if out[numeric].isna().any().any():
        raise LOBValidationError("LOB numeric columns contain NaN/non-numeric values")
    if (out["bid_price_1"] >= out["ask_price_1"]).any():
        raise LOBValidationError("Crossed or locked top-of-book")
    bid_sizes = out[[f"bid_size_{i}" for i in range(1, n + 1)]].to_numpy()
    ask_sizes = out[[f"ask_size_{i}" for i in range(1, n + 1)]].to_numpy()
    if (bid_sizes < 0).any() or (ask_sizes < 0).any():
        raise LOBValidationError("Negative book size")
    return out


def load_lob_csv(path: str) -> pd.DataFrame:
    return validate_lob(pd.read_csv(path))


def make_synthetic_lob(n_events: int = 1200, levels: int = 10, seed: int = 2025):
    rng = np.random.default_rng(seed)
    imbalance = 0.20 * np.tanh(rng.normal(size=n_events))
    imbalance += rng.normal(0, 0.05, n_events)
    void = rng.random(n_events) < (0.045 + 0.045 * np.abs(imbalance))

    mid = np.empty(n_events)
    mid[0] = 100.0
    noise = rng.normal(0, 0.00003, n_events)
    for t in range(1, n_events):
        signal = 0.00020 * imbalance[t - 1] + 0.00065 * void[t - 1] * np.sign(imbalance[t - 1])
        mid[t] = mid[t - 1] * np.exp(noise[t] + signal)

    rows = []
    for t in range(n_events):
        row = {
            "timestamp": pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(microseconds=t)
        }
        tick = max(0.01, mid[t] * 0.00002)
        spread = tick * (2 + 2 * rng.random())
        sizes = 30 * np.exp(-0.11 * np.arange(1, levels + 1))

        if void[t] and levels >= 10:
            removed = sizes[3:6].sum()
            sizes[3:6] *= 0.03
            sizes[1] += removed * 0.33
            sizes[8] += removed * 0.33
            sizes[9] += removed * 0.34

        imb = float(np.clip(imbalance[t], -0.8, 0.8))
        for level in range(1, levels + 1):
            distance = (level - 0.5) * tick
            base = sizes[level - 1]
            row[f"bid_price_{level}"] = mid[t] - spread / 2 - distance
            row[f"ask_price_{level}"] = mid[t] + spread / 2 + distance
            row[f"bid_size_{level}"] = max(1e-4, base * (1 + imb) * rng.lognormal(0, 0.15))
            row[f"ask_size_{level}"] = max(1e-4, base * (1 - imb) * rng.lognormal(0, 0.15))
        rows.append(row)
    return validate_lob(pd.DataFrame(rows))


def microstructure_features(df: pd.DataFrame, depth_levels: int = 3):
    n = min(depth_levels, level_columns(df))
    bp = df["bid_price_1"].to_numpy(float)
    ap = df["ask_price_1"].to_numpy(float)
    bs = np.column_stack([df[f"bid_size_{i}"].to_numpy(float) for i in range(1, n + 1)])
    ass = np.column_stack([df[f"ask_size_{i}"].to_numpy(float) for i in range(1, n + 1)])

    mid = (bp + ap) / 2
    spread = ap - bp
    bid_depth = bs.sum(1)
    ask_depth = ass.sum(1)
    total = np.maximum(bid_depth + ask_depth, 1e-12)
    imbalance = (bid_depth - ask_depth) / total
    microprice = (ap * bs[:, 0] + bp * ass[:, 0]) / np.maximum(bs[:, 0] + ass[:, 0], 1e-12)

    ofi = np.zeros(len(df))
    if len(df) > 1:
        pb = np.r_[bp[0], bp[:-1]]
        pa = np.r_[ap[0], ap[:-1]]
        pbs = np.r_[bs[0, 0], bs[:-1, 0]]
        pas = np.r_[ass[0, 0], ass[:-1, 0]]
        ofi = (
            np.where(bp > pb, bs[:, 0], np.where(bp < pb, -pbs, bs[:, 0] - pbs))
            - np.where(ap < pa, ass[:, 0], np.where(ap > pa, -pas, ass[:, 0] - pas))
        )

    log_mid = np.log(mid)
    ret = np.r_[0.0, np.diff(log_mid)]
    rolling_vol = pd.Series(ret).rolling(100, min_periods=20).std().to_numpy()
    rolling_vol = np.nan_to_num(
        rolling_vol,
        nan=float(np.nanmedian(rolling_vol)) if np.isfinite(rolling_vol).any() else 0.0,
    )
    dt_us = np.r_[np.nan, np.diff(df["timestamp"].astype("int64").to_numpy()) / 1000.0]
    median_dt = float(np.nanmedian(dt_us)) if np.isfinite(dt_us).any() else 0.0

    return pd.DataFrame({
        "mid": mid,
        "spread_bps": spread / np.maximum(mid, 1e-12) * 1e4,
        "imbalance": imbalance,
        "microprice_edge_bps": (microprice - mid) / np.maximum(mid, 1e-12) * 1e4,
        "ofi": ofi,
        "log_mid": log_mid,
        "ret_1": ret,
        "rolling_vol": rolling_vol,
        "bid_depth": bid_depth,
        "ask_depth": ask_depth,
        "interarrival_us": np.nan_to_num(dt_us, nan=median_dt),
    })


def make_point_clouds(df: pd.DataFrame, indices: np.ndarray, levels: int = 10):
    n = min(levels, level_columns(df))
    clouds = []
    for idx in indices:
        row = df.iloc[int(idx)]
        mid = (float(row.bid_price_1) + float(row.ask_price_1)) / 2
        spread = max(float(row.ask_price_1 - row.bid_price_1), 1e-9)
        points = []
        for level in range(1, n + 1):
            points.append(((float(row[f"bid_price_{level}"]) - mid) / spread, float(row[f"bid_size_{level}"])))
        for level in range(1, n + 1):
            points.append(((float(row[f"ask_price_{level}"]) - mid) / spread, float(row[f"ask_size_{level}"])))

        sizes = np.asarray([size for _, size in points])
        threshold = max(float(np.quantile(sizes, 0.15)), 1e-4)
        keep = sizes >= threshold
        minimum_points = max(8, len(points) // 3)
        if keep.sum() < minimum_points:
            keep = np.zeros(len(points), dtype=bool)
            keep[np.argsort(sizes)[-minimum_points:]] = True

        cloud = np.asarray([[x, np.log1p(size)] for (x, size), ok in zip(points, keep) if ok])
        center = np.median(cloud[:, 1])
        scale = np.median(np.abs(cloud[:, 1] - center)) + 1e-6
        cloud[:, 1] = (cloud[:, 1] - center) / scale
        clouds.append(cloud)
    return clouds


def _betti(diagram, dimension, radii):
    dgm = np.asarray(diagram)
    dgm = dgm[(dgm[:, 2] == dimension) & (dgm[:, 1] > dgm[:, 0])]
    return np.asarray([
        np.sum((dgm[:, 0] <= radius) & (dgm[:, 1] > radius))
        for radius in radii
    ], dtype=int)


def _entropy(values):
    p = np.asarray(values, float)
    p = p[np.isfinite(p) & (p > 0)]
    if not len(p):
        return 0.0
    q = p / p.sum()
    return float(-(q * np.log(q)).sum())


def _fallback_topology(cloud, radii):
    n = len(cloud)
    dist = np.sqrt(((cloud[:, None, :] - cloud[None, :, :]) ** 2).sum(-1))
    betti0, betti1 = [], []

    for radius in radii:
        parent = list(range(n))
        components = n
        edges = 0

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(n):
            for j in range(i + 1, n):
                if dist[i, j] <= radius:
                    edges += 1
                    a, b = find(i), find(j)
                    if a != b:
                        parent[a] = b
                        components -= 1

        betti0.append(components)
        betti1.append(max(0, edges - n + components))

    nearest = np.partition(dist + np.eye(n) * 1e9, 1, axis=1)[:, 1]
    return np.asarray(betti0), np.asarray(betti1), float(np.quantile(nearest, 0.9))


def persistent_features(clouds, radii, require_gtda=False, n_jobs=-1):
    try:
        from gtda.homology import VietorisRipsPersistence

        vr = VietorisRipsPersistence(
            metric="euclidean",
            homology_dimensions=(0, 1),
            max_edge_length=float(np.max(radii)),
            collapse_edges=True,
            n_jobs=n_jobs,
        )
        diagrams = vr.fit_transform(clouds)
        b0 = np.stack([_betti(d, 0, radii) for d in diagrams])
        b1 = np.stack([_betti(d, 1, radii) for d in diagrams])

        h1_max, h1_entropy = [], []
        for diagram in diagrams:
            d1 = diagram[diagram[:, 2] == 1]
            persistence = d1[:, 1] - d1[:, 0] if len(d1) else np.empty(0)
            h1_max.append(float(np.max(persistence)) if len(persistence) else 0.0)
            h1_entropy.append(_entropy(persistence))

        names = (
            [f"betti0_r{i}" for i in range(len(radii))]
            + [f"betti1_r{i}" for i in range(len(radii))]
            + ["h1_max_persistence", "h1_persistence_entropy"]
        )
        return np.column_stack([b0, b1, h1_max, h1_entropy]), {
            "backend": "giotto-tda",
            "n_clouds": len(clouds),
            "radii": list(map(float, radii)),
            "feature_names": names,
            "homology_dimensions": [0, 1],
        }
    except ImportError:
        if require_gtda:
            raise RuntimeError("giotto-tda is required; install the tda extra")
    except Exception:
        if require_gtda:
            raise

    rows = []
    for cloud in clouds:
        b0, b1, void_proxy = _fallback_topology(cloud, radii)
        rows.append(np.r_[b0, b1, void_proxy, 0.0])

    names = (
        [f"betti0_r{i}" for i in range(len(radii))]
        + [f"betti1_r{i}" for i in range(len(radii))]
        + ["void_gap_proxy", "h1_persistence_entropy"]
    )
    return np.asarray(rows), {
        "backend": "threshold-graph-fallback",
        "n_clouds": len(clouds),
        "radii": list(map(float, radii)),
        "feature_names": names,
        "warning": "Not persistent homology; install giotto-tda for exact VR persistence.",
    }


def fracdiff_weights(d, width):
    if d < 0 or width < 1:
        raise ValueError("invalid d or width")
    weights = [1.0]
    for k in range(1, width):
        weights.append(-weights[-1] * (d - k + 1) / k)
    return np.asarray(weights[::-1])


def fractional_diff_cpu(x, d, width=256):
    arr = np.asarray(x, float)
    weights = fracdiff_weights(d, width)
    out = np.full(len(arr), np.nan)
    m = len(weights)
    for i in range(m - 1, len(arr)):
        out[i] = np.dot(weights, arr[i - m + 1:i + 1])
    return out


try:
    from numba import cuda

    @cuda.jit
    def _fd_kernel(x, weights, out):
        i = cuda.grid(1)
        if i >= x.size or i < weights.size - 1:
            return
        acc = 0.0
        for k in range(weights.size):
            acc += weights[k] * x[i - weights.size + 1 + k]
        out[i] = acc
except Exception:
    cuda = None
    _fd_kernel = None


def fractional_diff_gpu(x, d, width=256):
    try:
        if cuda is None or not cuda.is_available():
            return fractional_diff_cpu(x, d, width), False, "cpu-fallback:no-cuda"
        arr = np.asarray(x, float)
        weights = fracdiff_weights(d, width)
        out = np.full(len(arr), np.nan)
        dx = cuda.to_device(arr)
        dw = cuda.to_device(weights)
        dout = cuda.to_device(out)
        threads = 128
        blocks = (len(arr) + threads - 1) // threads
        _fd_kernel[blocks, threads](dx, dw, dout)
        cuda.synchronize()
        return dout.copy_to_host(), True, "numba-cuda"
    except Exception as exc:
        return fractional_diff_cpu(x, d, width), False, f"cpu-fallback:{type(exc).__name__}"


def select_stationary_d(x, candidates, adf_p=0.05, width=256):
    from statsmodels.tsa.stattools import adfuller

    rows = []
    chosen = None
    x = np.asarray(x, float)
    for d in candidates:
        y = fractional_diff_cpu(x, float(d), width)
        y = y[np.isfinite(y)]
        if len(y) < 100:
            continue
        p_value = float(adfuller(y, maxlag=min(20, len(y) // 10), autolag="AIC")[1])
        corr = float(np.corrcoef(x[-len(y):], y)[0, 1]) if np.std(y) else 0.0
        rows.append({"d": float(d), "adf_pvalue": p_value, "corr_with_price": corr})
        if p_value < adf_p and chosen is None:
            chosen = float(d)

    if chosen is None:
        chosen = float(candidates[-1])
    return chosen, {"candidates": rows, "selected_d": chosen, "adf_threshold": adf_p}


def focal_grad_hess(predt, dtrain, gamma=2.0, alpha=0.75):
    y = dtrain.get_label().astype(float)
    p = 1 / (1 + np.exp(-np.clip(predt, -40, 40)))
    q = 1 - p
    positive = y > 0.5
    a = np.where(positive, alpha, 1 - alpha)
    logp = np.log(np.clip(p, 1e-12, 1))
    logq = np.log(np.clip(q, 1e-12, 1))

    grad_pos = a * q**gamma * (gamma * p * logp - q)
    grad_neg = a * p**gamma * (-gamma * q * logq + p)
    grad = np.where(positive, grad_pos, grad_neg)

    h_pos = -a * p * q**gamma * (
        gamma**2 * p * logp + gamma * p * logp + 2 * gamma * p
        - gamma * logp - 2 * gamma + p - 1
    )
    h_neg = a * p**gamma * q * (
        gamma**2 * p * logq - gamma**2 * logq
        + gamma * p * logq + 2 * gamma * p + p
    )
    hess = np.where(positive, h_pos, h_neg)
    return grad, np.maximum(np.abs(hess), 1e-6)


def focal_loss_value(predt, y, gamma=2.0, alpha=0.75):
    p = 1 / (1 + np.exp(-np.clip(predt, -40, 40)))
    pt = np.where(y > 0.5, p, 1 - p)
    at = np.where(y > 0.5, alpha, 1 - alpha)
    return float(np.mean(
        -at * (1 - pt)**gamma * np.log(np.clip(pt, 1e-12, 1))
    ))


def _resample(X, y, seed, ratio=0.35):
    counts = np.bincount(y.astype(int), minlength=2)
    if min(counts) == 0 or min(counts) / max(counts) >= ratio:
        return X, y
    return RandomOverSampler(
        sampling_strategy=ratio,
        random_state=seed,
    ).fit_resample(X, y)


def fit_focal_xgb(X, y, seed=2025, alpha=0.75, gamma=2.0, ratio=0.35):
    X_resampled, y_resampled = _resample(X, y, seed, ratio)
    params = {
        "max_depth": 3,
        "eta": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.9,
        "min_child_weight": 8,
        "lambda": 2.0,
        "tree_method": "hist",
        "seed": seed,
        "disable_default_eval_metric": 1,
    }
    return xgb.train(
        params,
        xgb.DMatrix(X_resampled, label=y_resampled),
        num_boost_round=220,
        obj=lambda pred, dmatrix: focal_grad_hess(pred, dmatrix, gamma, alpha),
        verbose_eval=False,
    )


def fit_logloss_xgb(X, y, seed=2025, ratio=0.35):
    X_resampled, y_resampled = _resample(X, y, seed, ratio)
    params = {
        "max_depth": 3,
        "eta": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.9,
        "min_child_weight": 8,
        "lambda": 2.0,
        "tree_method": "hist",
        "seed": seed,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
    }
    return xgb.train(
        params,
        xgb.DMatrix(X_resampled, label=y_resampled),
        num_boost_round=220,
        verbose_eval=False,
    )


def evaluate(model, X, y, future):
    if len(y) == 0:
        raise ValueError("empty evaluation set")
    probability = model.predict(xgb.DMatrix(X))
    roc = float(roc_auc_score(y, probability)) if np.unique(y).size == 2 else None
    pearson_ic = (
        float(np.corrcoef(probability, future)[0, 1])
        if np.std(probability) and np.std(future) else 0.0
    )
    rank_ic = (
        float(np.corrcoef(
            np.argsort(np.argsort(probability)),
            np.argsort(np.argsort(future)),
        )[0, 1])
        if np.std(probability) and np.std(future) else 0.0
    )
    return {
        "roc_auc": roc,
        "pr_auc": float(average_precision_score(y, probability)),
        "logloss": float(log_loss(y, np.clip(probability, 1e-8, 1 - 1e-8))),
        "pearson_ic": pearson_ic,
        "rank_ic": rank_ic,
        "positive_rate": float(y.mean()),
    }, probability


def _build_matrix(cfg):
    seed = int(cfg.get("seed", 2025))
    raw = load_lob_csv(cfg["data_path"]) if cfg.get("data_path") else make_synthetic_lob(
        int(cfg.get("synthetic_events", 1200)),
        int(cfg.get("levels", 10)),
        seed,
    )
    micro = microstructure_features(raw, int(cfg.get("feature_depth", 3)))
    horizon = int(cfg.get("horizon", 10))
    ffd_width = int(cfg.get("ffd_width", 256))
    test_fraction = float(cfg.get("test_fraction", 0.30))
    purge = int(cfg.get("purge", horizon))

    raw_train_end = int(len(raw) * (1 - test_fraction)) - purge
    d, stationarity = select_stationary_d(
        micro.log_mid.to_numpy(),
        np.asarray(cfg.get("candidate_d", np.linspace(0, 1, 11))),
        float(cfg.get("adf_p", 0.05)),
        ffd_width,
    )

    fd, gpu_used, fd_backend = fractional_diff_gpu(
        micro.log_mid.to_numpy(), d, ffd_width
    )
    micro["fracdiff"] = fd

    valid_end = len(raw) - horizon
    topo_indices = np.arange(50, valid_end, int(cfg.get("tda_stride", 8)))
    radii = np.asarray(cfg.get("radii", [0.35, 0.55, 0.8, 1.1, 1.5]), float)
    topology, topology_meta = persistent_features(
        make_point_clouds(raw, topo_indices, int(cfg.get("levels", 10))),
        radii,
        bool(cfg.get("require_gtda", False)),
        int(cfg.get("tda_jobs", -1)),
    )
    topology_df = pd.DataFrame(
        topology,
        index=topo_indices,
        columns=topology_meta["feature_names"],
    ).reindex(range(len(raw))).ffill()

    base_columns = [c for c in micro.columns if c not in {"mid", "log_mid"}]
    if not bool(cfg.get("include_fracdiff", True)):
        base_columns.remove("fracdiff")

    arrays = [micro[base_columns].to_numpy(float)]
    names = base_columns[:]
    if bool(cfg.get("include_topology", True)):
        arrays.append(topology_df.to_numpy(float))
        names.extend(topology_df.columns)

    X_all = np.hstack(arrays)
    future = np.full(len(raw), np.nan)
    future[:-horizon] = micro.log_mid.to_numpy()[horizon:] - micro.log_mid.to_numpy()[:-horizon]

    y_all = np.full(len(raw), -1)
    finite_future = np.isfinite(future)
    y_all[finite_future] = (
        future[finite_future] > float(cfg.get("label_threshold", 0.0005))
    ).astype(int)

    valid = np.isfinite(X_all).all(1) & np.isfinite(future) & (y_all >= 0)
    X, y, fwd = X_all[valid], y_all[valid], future[valid]
    event_index = np.flatnonzero(valid)

    test_start = int(len(X) * (1 - test_fraction))
    train_end = max(0, test_start - purge)
    return {
        "raw": raw,
        "micro": micro,
        "X": X,
        "y": y,
        "future": fwd,
        "event_index": event_index,
        "test_start": test_start,
        "train_end": train_end,
        "d": d,
        "stationarity": stationarity,
        "topology": topology_df,
        "topology_meta": topology_meta,
        "radii": radii,
        "names": names,
        "gpu_used": gpu_used,
        "fd_backend": fd_backend,
        "raw_train_end": raw_train_end,
    }


def run_experiment(cfg):
    seed = int(cfg.get("seed", 2025))
    bundle = _build_matrix(cfg)
    X, y, future = bundle["X"], bundle["y"], bundle["future"]
    test_start, train_end = bundle["test_start"], bundle["train_end"]

    X_train, X_test = X[:train_end], X[test_start:]
    y_train, y_test = y[:train_end], y[test_start:]
    future_test = future[test_start:]

    focal = fit_focal_xgb(
        X_train,
        y_train,
        seed,
        float(cfg.get("focal_alpha", 0.75)),
        float(cfg.get("focal_gamma", 2.0)),
        float(cfg.get("oversample_target_ratio", 0.35)),
    )
    baseline = fit_logloss_xgb(
        X_train,
        y_train,
        seed,
        float(cfg.get("oversample_target_ratio", 0.35)),
    )

    focal_metrics, probability = evaluate(focal, X_test, y_test, future_test)
    baseline_metrics, _ = evaluate(baseline, X_test, y_test, future_test)

    feature_gain = {}
    for key, value in focal.get_score(importance_type="gain").items():
        try:
            feature_gain[bundle["names"][int(key[1:])]] = float(value)
        except Exception:
            feature_gain[key] = float(value)

    interarrival = bundle["micro"].interarrival_us
    median_interarrival_us = float(interarrival.iloc[1:].median()) if len(interarrival) > 1 else 0.0

    metrics = {
        "data_source": f"csv:{cfg['data_path']}" if cfg.get("data_path") else "synthetic",
        "events": len(bundle["raw"]),
        "model_samples": len(X),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "purge_gap_samples": test_start - train_end,
        "selected_d": bundle["d"],
        "stationarity": bundle["stationarity"],
        "tda": bundle["topology_meta"],
        "fractional_diff_backend": bundle["fd_backend"],
        "gpu_used": bundle["gpu_used"],
        "feature_flags": {
            "topology": bool(cfg.get("include_topology", True)),
            "fractional_diff": bool(cfg.get("include_fracdiff", True)),
        },
        "feature_names": bundle["names"],
        "focal_loss": focal_metrics,
        "logloss_baseline": baseline_metrics,
        "target_resume_ic": 0.064,
        "class_balance": {
            "train_positive_rate_before_resample": float(y_train.mean()),
            "test_positive_rate": float(y_test.mean()),
        },
        "median_interarrival_us": median_interarrival_us,
        "event_rate_hz_approx": float(1e6 / median_interarrival_us) if median_interarrival_us else None,
        "focal_feature_gain": feature_gain,
    }
    return {
        "metrics": metrics,
        "topology": bundle["topology"],
        "radii": bundle["radii"],
        "test_probability": probability,
        "test_event_index": bundle["event_index"][test_start:],
    }


def run_walk_forward(cfg):
    seed = int(cfg.get("seed", 2025))
    raw = load_lob_csv(cfg["data_path"]) if cfg.get("data_path") else make_synthetic_lob(
        int(cfg.get("synthetic_events", 1200)),
        int(cfg.get("levels", 10)),
        seed,
    )
    micro = microstructure_features(raw, int(cfg.get("feature_depth", 3)))
    horizon = int(cfg.get("horizon", 10))
    test_fraction = float(cfg.get("walk_test_fraction", 0.10))
    purge = int(cfg.get("purge", horizon))
    ffd_width = int(cfg.get("ffd_width", 256))
    eligible = np.arange(ffd_width, len(raw) - horizon)
    n = len(eligible)
    initial = int(n * float(cfg.get("walk_initial_fraction", 0.45)))
    size = int(n * test_fraction)
    radii = np.asarray(cfg.get("radii", [0.35, 0.55, 0.8, 1.1, 1.5]), float)

    topo_indices = np.arange(50, len(raw) - horizon, int(cfg.get("tda_stride", 8)))
    topology, topology_meta = persistent_features(
        make_point_clouds(raw, topo_indices, int(cfg.get("levels", 10))),
        radii,
        bool(cfg.get("require_gtda", False)),
        int(cfg.get("tda_jobs", -1)),
    )
    topology_df = pd.DataFrame(
        topology,
        index=topo_indices,
        columns=topology_meta["feature_names"],
    ).reindex(range(len(raw))).ffill()

    folds = []
    train_end = initial
    fold = 1

    while train_end + purge + size <= n:
        raw_train_end = int(eligible[train_end - 1]) + 1
        d, stationarity = select_stationary_d(
            micro.log_mid.to_numpy()[:raw_train_end],
            np.asarray(cfg.get("candidate_d", np.linspace(0, 1, 11))),
            float(cfg.get("adf_p", 0.05)),
            ffd_width,
        )
        fd, _, _ = fractional_diff_gpu(micro.log_mid.to_numpy(), d, ffd_width)
        frame = micro.copy()
        frame["fracdiff"] = fd

        base_columns = [c for c in frame.columns if c not in {"mid", "log_mid"}]
        if not bool(cfg.get("include_fracdiff", True)):
            base_columns.remove("fracdiff")

        arrays = [frame[base_columns].to_numpy(float)]
        if bool(cfg.get("include_topology", True)):
            arrays.append(topology_df.to_numpy(float))

        X_all = np.hstack(arrays)
        future = np.full(len(raw), np.nan)
        future[:-horizon] = frame.log_mid.to_numpy()[horizon:] - frame.log_mid.to_numpy()[:-horizon]
        y_all = np.full(len(raw), -1)
        finite_future = np.isfinite(future)
        y_all[finite_future] = (
            future[finite_future] > float(cfg.get("label_threshold", 0.0005))
        ).astype(int)

        valid = np.isfinite(X_all).all(1) & np.isfinite(future) & (y_all >= 0)
        X, y, fwd = X_all[valid], y_all[valid], future[valid]

        train_slice = slice(0, train_end)
        test_slice = slice(train_end + purge, train_end + purge + size)
        X_train, X_test = X[train_slice], X[test_slice]
        y_train, y_test = y[train_slice], y[test_slice]
        fwd_test = fwd[test_slice]

        focal_metrics, _ = evaluate(
            fit_focal_xgb(X_train, y_train, 2025 + fold),
            X_test, y_test, fwd_test
        )
        baseline_metrics, _ = evaluate(
            fit_logloss_xgb(X_train, y_train, 2025 + fold),
            X_test, y_test, fwd_test
        )

        selected_p = next(
            (row["adf_pvalue"] for row in stationarity["candidates"] if row["d"] == d),
            None,
        )
        folds.append({
            "fold": fold,
            "selected_d": d,
            "adf_selected_pvalue": selected_p,
            "focal": focal_metrics,
            "logloss": baseline_metrics,
        })
        fold += 1
        train_end += size

    if not folds:
        raise ValueError("Not enough observations for walk-forward")
    return {
        "data_source": "csv" if cfg.get("data_path") else "synthetic",
        "fold_count": len(folds),
        "folds": folds,
        "tda": topology_meta,
    }


def benchmark_fd(events=200000, width=256, d=0.45):
    rng = np.random.default_rng(7)
    x = np.log(100 + np.cumsum(rng.normal(0, 0.01, events)))
    t0 = time.perf_counter()
    cpu = fractional_diff_cpu(x, d, width)
    cpu_seconds = time.perf_counter() - t0
    backend, used, backend_name = fractional_diff_gpu(x, d, width)
    valid = np.isfinite(cpu) & np.isfinite(backend)
    max_error = float(np.max(np.abs(cpu[valid] - backend[valid]))) if valid.any() else None
    return {
        "events": events,
        "width": width,
        "d": d,
        "cpu_elapsed_s": cpu_seconds,
        "backend_elapsed_s": None if used else cpu_seconds,
        "backend": backend_name,
        "gpu_used": used,
        "events_per_s_cpu": (events - width + 1) / cpu_seconds,
        "checksum": float(cpu[-1]),
        "backend_max_abs_error": max_error,
    }