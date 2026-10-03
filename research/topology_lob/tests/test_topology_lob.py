import numpy as np
import xgboost as xgb

from topology_lob import (
    LOBValidationError,
    fracdiff_weights,
    fractional_diff_cpu,
    focal_grad_hess,
    focal_loss_value,
    make_synthetic_lob,
    persistent_features,
    validate_lob,
)


def test_schema_and_timestamps():
    df = make_synthetic_lob(200, 10, 1)
    assert df["timestamp"].is_monotonic_increasing
    assert "bid_price_10" in df


def test_crossed_book_rejected():
    df = make_synthetic_lob(20, 4, 2)
    df.loc[0, "ask_price_1"] = df.loc[0, "bid_price_1"]
    try:
        validate_lob(df)
    except LOBValidationError:
        return
    assert False, "crossed book must be rejected"


def test_ffd_d0_identity():
    x = np.arange(100, dtype=float)
    np.testing.assert_allclose(fractional_diff_cpu(x, 0, 32)[31:], x[31:])


def test_ffd_weights():
    weights = fracdiff_weights(0.45, 64)
    assert len(weights) == 64
    assert np.isfinite(weights).all()


def test_topology_detects_gap():
    left = np.column_stack([np.linspace(-2, -0.5, 8), np.zeros(8)])
    right = np.column_stack([np.linspace(0.5, 2, 8), np.zeros(8)])
    features, meta = persistent_features(
        [np.vstack([left, right])],
        np.array([0.1, 0.5, 1.0]),
        require_gtda=False,
    )
    assert meta["backend"] in {"threshold-graph-fallback", "giotto-tda"}
    assert features[0, 0] >= 2


def test_focal_gradient_against_finite_difference():
    pred = np.array([-1.1, 0.2, 1.7])
    y = np.array([0.0, 1.0, 1.0])
    dtrain = xgb.DMatrix(np.zeros((3, 1)), label=y)
    grad, _ = focal_grad_hess(pred.copy(), dtrain)

    eps = 1e-5
    numeric = []
    for i in range(3):
        plus = pred.copy()
        minus = pred.copy()
        plus[i] += eps
        minus[i] -= eps
        numeric.append(
            (focal_loss_value(plus, y) - focal_loss_value(minus, y))
            / (2 * eps) * 3
        )
    np.testing.assert_allclose(grad, numeric, rtol=2e-4, atol=2e-5)
