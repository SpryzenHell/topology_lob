"""FI-2010 raw-file adapter.

The standard FI-2010 text files are feature-major: the first axis contains
40 raw LOB variables, 104 derived variables, and 5 future labels. The first
40 variables represent ten ask/bid levels with price and volume.

This module converts the raw 40-dimensional representation into the canonical
Topology-LOB CSV schema:

    timestamp,bid_price_1,bid_size_1,ask_price_1,ask_size_1,...

FI-2010 has event order rather than exchange timestamps in the benchmark dump.
The generated timestamp column is therefore an explicit event index on a
synthetic nanosecond clock and must not be interpreted as real wall-clock time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

N_RAW_FEATURES = 40
N_TOTAL_ROWS = 149
N_LABELS = 5
HORIZONS = (10, 20, 30, 50, 100)


@dataclass(frozen=True)
class FI2010Raw:
    features: np.ndarray
    labels: np.ndarray | None
    source: str
    normalization: str


def read_feature_major(path: str | Path) -> np.ndarray:
    path = Path(path)
    matrix = np.loadtxt(path)
    if matrix.ndim != 2:
        raise ValueError(f"{path}: expected 2-D FI-2010 text matrix")
    rows, cols = matrix.shape
    if rows == N_TOTAL_ROWS or 44 <= rows <= N_TOTAL_ROWS:
        return matrix
    if cols == N_TOTAL_ROWS or 44 <= cols <= N_TOTAL_ROWS:
        return matrix.T
    raise ValueError(
        f"{path}: expected one axis near {N_TOTAL_ROWS}; got {matrix.shape}. "
        "Check that this is a raw FI-2010 benchmark text file."
    )


def load_raw(path: str | Path) -> FI2010Raw:
    path = Path(path)
    matrix = read_feature_major(path)
    n_rows = matrix.shape[0]
    features = matrix[:N_RAW_FEATURES].T.astype(np.float64, copy=False)

    labels = None
    if n_rows >= N_RAW_FEATURES + N_LABELS:
        labels = matrix[-N_LABELS:].T.astype(np.int8, copy=False)
        observed = set(np.unique(labels).tolist())
        if not observed.issubset({1, 2, 3}):
            raise ValueError(
                f"{path}: unexpected FI-2010 label values {sorted(observed)}"
            )

    name = path.name.lower()
    if "decpre" in name:
        normalization = "decimal-precision"
    elif "minmax" in name:
        normalization = "minmax"
    elif "zscore" in name or "z_score" in name:
        normalization = "zscore"
    else:
        normalization = "unknown"

    return FI2010Raw(
        features=features,
        labels=labels,
        source=str(path),
        normalization=normalization,
    )


def raw40_to_canonical(raw40: np.ndarray) -> pd.DataFrame:
    """Map FI-2010 x_t=[ask_px, ask_size, bid_px, bid_size] per level."""
    x = np.asarray(raw40, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 40:
        raise ValueError(f"expected (n_events,40), got {x.shape}")

    values: dict[str, np.ndarray] = {}
    for level in range(1, 11):
        offset = (level - 1) * 4
        ask_px = x[:, offset]
        ask_sz = x[:, offset + 1]
        bid_px = x[:, offset + 2]
        bid_sz = x[:, offset + 3]
        values[f"bid_price_{level}"] = bid_px
        values[f"bid_size_{level}"] = bid_sz
        values[f"ask_price_{level}"] = ask_px
        values[f"ask_size_{level}"] = ask_sz

    # FI-2010 contains event order, not a usable timestamp. Use a strictly
    # increasing synthetic clock only to satisfy the generic L2 schema.
    values["timestamp"] = pd.Timestamp("2000-01-01", tz="UTC") + pd.to_timedelta(
        np.arange(x.shape[0]), unit="ns"
    )
    return pd.DataFrame(values)[
        ["timestamp"]
        + sum(
            (
                [
                    f"bid_price_{i}",
                    f"bid_size_{i}",
                    f"ask_price_{i}",
                    f"ask_size_{i}",
                ]
                for i in range(1, 11)
            ),
            [],
        )
    ]


def convert_file(
    input_path: str | Path,
    output_csv: str | Path,
    *,
    require_decpre: bool = True,
) -> dict:
    raw = load_raw(input_path)
    if require_decpre and raw.normalization != "decimal-precision":
        raise ValueError(
            "Topology-LOB point clouds require FI-2010 Decimal-Precision input "
            "so price/size values remain economically interpretable. "
            "Z-score/MinMax files are acceptable for benchmark classification "
            "but are not used here for LOB price-volume geometry."
        )

    df = raw40_to_canonical(raw.features)

    # Sanity checks specific to the economic (DecPre) representation.
    if not np.all(df[[c for c in df.columns if "size" in c]].to_numpy() >= 0):
        raise ValueError("FI-2010 DecPre contains negative size values after mapping")
    if not np.all(df["bid_price_1"].to_numpy() < df["ask_price_1"].to_numpy()):
        raise ValueError("FI-2010 data contains an invalid top-of-book row")

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    return {
        "input": str(input_path),
        "output": str(output_csv),
        "events": int(len(df)),
        "levels": 10,
        "normalization": raw.normalization,
        "has_labels": raw.labels is not None,
        "timestamp_note": "synthetic monotonic event index; FI-2010 dump has event order rather than exchange timestamps",
    }