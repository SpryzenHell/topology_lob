import numpy as np

from fi2010 import raw40_to_canonical


def test_fi2010_mapping_order():
    raw = np.zeros((2, 40), dtype=float)
    raw[0, :4] = [101.0, 12.0, 100.5, 8.0]
    raw[1, :4] = [101.1, 13.0, 100.6, 9.0]
    frame = raw40_to_canonical(raw)

    assert frame.loc[0, "ask_price_1"] == 101.0
    assert frame.loc[0, "ask_size_1"] == 12.0
    assert frame.loc[0, "bid_price_1"] == 100.5
    assert frame.loc[0, "bid_size_1"] == 8.0
    assert frame["timestamp"].is_monotonic_increasing


def test_fi2010_has_exactly_10_levels():
    raw = np.tile(np.arange(40, dtype=float), (3, 1))
    frame = raw40_to_canonical(raw)
    price_cols = [c for c in frame.columns if "price" in c]
    size_cols = [c for c in frame.columns if "size" in c]
    assert len(price_cols) == 20
    assert len(size_cols) == 20