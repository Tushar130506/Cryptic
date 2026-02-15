import numpy as np
import pandas as pd
import config
from features_ic import add_ic_features, get_ic_feature_names


def _mock_df(n=200):
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    price = pd.Series(np.linspace(100, 150, n), index=idx)
    df = pd.DataFrame({
        "price": price,
        "Open": price * 0.995,
        "High": price * 1.01,
        "Low": price * 0.99,
        "Volume": np.full(n, 1000.0),
    })
    return df


def test_features_no_nan():
    df = _mock_df(300)
    df = add_ic_features(df, config.HORIZON)
    feature_cols = get_ic_feature_names()
    assert set(feature_cols).issubset(df.columns)
    assert df[feature_cols].isna().sum().sum() == 0


def test_target_has_expected_nans():
    df = _mock_df(300)
    df = add_ic_features(df, config.HORIZON)
    assert df["target"].isna().sum() <= config.HORIZON
