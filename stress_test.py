"""
Parameter stress test for Quant Block.
Runs a small suite of configurations and prints summary metrics.
"""

import os
import types
import numpy as np
import pandas as pd

import config as base_config
from data import load_data
from features_ic import add_ic_features
from backtest import run_backtest_ic_test


OUTPUT_DIR = "results"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "stress_test_results.csv")


def build_config(overrides):
    cfg = types.SimpleNamespace()
    for key in dir(base_config):
        if key.isupper():
            setattr(cfg, key, getattr(base_config, key))
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def run_case(case_name, overrides):
    cfg = build_config(overrides)
    df = load_data(cfg)
    df = add_ic_features(df, cfg.HORIZON)
    results = run_backtest_ic_test(df, cfg)
    return {
        "case": case_name,
        "ticker": getattr(cfg, "TICKER", base_config.TICKER),
        "total_return": results["total_return"],
        "sharpe": results["sharpe"],
        "max_dd": results["max_dd"],
        "ic": results["ic"],
        "ic_pvalue": results["ic_pvalue"],
        "win_rate": results["win_rate"],
        "profit_factor": results["profit_factor"],
    }


def main():
    base_cases = [
        ("baseline", {}),
        ("lookback_300", {"LOOKBACK": 300}),
        ("lookback_700", {"LOOKBACK": 700}),
        ("horizon_10", {"HORIZON": 10}),
        ("horizon_40", {"HORIZON": 40}),
        ("retrain_10", {"RETRAIN_FREQ": 10}),
        ("retrain_40", {"RETRAIN_FREQ": 40}),
        ("target_vol_0.10", {"TARGET_VOL": 0.10}),
        ("target_vol_0.25", {"TARGET_VOL": 0.25}),
        ("max_lev_1.0", {"MAX_LEVERAGE": 1.0}),
        ("stop_loss_0.15", {"STOP_LOSS": -0.15}),
        ("tx_cost_0.002", {"TRANSACTION_COST": 0.002}),
    ]

    tickers = [
        ("BTC-USD", "crypto"),
        ("ETH-USD", "crypto"),
        ("SPY", "equity"),
        ("QQQ", "equity"),
        ("GLD", "commodity"),
    ]

    cases = []
    for case_name, overrides in base_cases:
        for ticker, _asset in tickers:
            overrides_with_ticker = dict(overrides)
            overrides_with_ticker["TICKER"] = ticker
            cases.append((f"{case_name}_{ticker}", overrides_with_ticker))

    rows = []
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        for name, overrides in cases:
            try:
                rows.append(run_case(name, overrides))
            except Exception as exc:
                rows.append({"case": name, "error": str(exc)})

            pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)
    except KeyboardInterrupt:
        print("\nInterrupted. Saving partial results.")

    df = pd.DataFrame(rows)
    print("\nSTRESS TEST RESULTS")
    print(df.to_string(index=False))
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
