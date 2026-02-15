"""
Streamlit dashboard for tuning and testing Quant Block parameters.
"""

import os
import json
import types
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config as base_config
from data import load_data
from features_ic import add_ic_features
from backtest import run_backtest_ic_test


st.set_page_config(page_title="Quant Block Dashboard", layout="wide")

st.title("Quant Block - Tuning Dashboard")
st.caption("Tune parameters and run walk-forward backtests with IC validation.")

PRESET_FILE = os.path.join("ui", "presets", "presets.json")


def load_presets():
    """Load saved presets from disk."""
    if not os.path.exists(PRESET_FILE):
        return {}
    try:
        with open(PRESET_FILE, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_presets(presets):
    """Persist presets to disk."""
    os.makedirs(os.path.dirname(PRESET_FILE), exist_ok=True)
    with open(PRESET_FILE, "w", encoding="utf-8") as handle:
        json.dump(presets, handle, indent=2)


def apply_preset_to_state(preset):
    """Apply preset values to session state."""
    for key, value in preset.items():
        st.session_state[key] = value


def build_config(overrides):
    """Create a config-like object with overrides for experimentation."""
    cfg = types.SimpleNamespace()
    for key in dir(base_config):
        if key.isupper():
            setattr(cfg, key, getattr(base_config, key))
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def compute_equity_curve(returns_pct):
    """Compute equity curve from percent returns."""
    returns = np.array(returns_pct, dtype=float) / 100.0
    equity = np.cumprod(1.0 + returns)
    return equity


@st.cache_data(show_spinner=False)
def cached_load_data(ticker, start_date, min_points):
    cfg = types.SimpleNamespace(TICKER=ticker, START_DATE=start_date, MIN_DATA_POINTS=min_points)
    return load_data(cfg)


def run_backtest_with_overrides(overrides):
    cfg = build_config(overrides)
    df = cached_load_data(cfg.TICKER, cfg.START_DATE, cfg.MIN_DATA_POINTS).copy()
    df = add_ic_features(df, cfg.HORIZON)
    results = run_backtest_ic_test(df, cfg)
    return results


with st.sidebar:
    st.header("Configuration")

    defaults = {
        "ticker": base_config.TICKER,
        "start_date": base_config.START_DATE,
        "lookback": base_config.LOOKBACK,
        "horizon": base_config.HORIZON,
        "retrain": base_config.RETRAIN_FREQ,
        "target_vol": base_config.TARGET_VOL,
        "max_leverage": base_config.MAX_LEVERAGE,
        "stop_loss": base_config.STOP_LOSS,
        "transaction_cost": base_config.TRANSACTION_COST,
        "slippage": base_config.SLIPPAGE,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    st.subheader("Presets")
    presets = load_presets()
    preset_names = ["(none)"] + sorted(presets.keys())
    selected_preset = st.selectbox("Preset", preset_names, key="preset_select")
    preset_name = st.text_input("Preset Name", key="preset_name")

    col_save, col_load = st.columns(2)
    with col_save:
        if st.button("Save Preset"):
            if not preset_name.strip():
                st.warning("Enter a preset name before saving.")
            else:
                presets[preset_name.strip()] = {
                    "ticker": st.session_state["ticker"],
                    "start_date": st.session_state["start_date"],
                    "lookback": st.session_state["lookback"],
                    "horizon": st.session_state["horizon"],
                    "retrain": st.session_state["retrain"],
                    "target_vol": st.session_state["target_vol"],
                    "max_leverage": st.session_state["max_leverage"],
                    "stop_loss": st.session_state["stop_loss"],
                    "transaction_cost": st.session_state["transaction_cost"],
                    "slippage": st.session_state["slippage"],
                }
                save_presets(presets)
                st.success("Preset saved.")
    with col_load:
        if st.button("Load Preset"):
            if selected_preset == "(none)":
                st.warning("Select a preset to load.")
            else:
                apply_preset_to_state(presets[selected_preset])
                st.experimental_rerun()

    ticker = st.text_input("Ticker", value=base_config.TICKER, key="ticker")
    start_date = st.text_input("Start Date (YYYY-MM-DD)", value=base_config.START_DATE, key="start_date")

    lookback = st.slider("Lookback (days)", min_value=200, max_value=1000, value=base_config.LOOKBACK, step=25, key="lookback")
    horizon = st.slider("Horizon (days)", min_value=5, max_value=60, value=base_config.HORIZON, step=1, key="horizon")
    retrain = st.slider("Retrain Frequency (days)", min_value=5, max_value=60, value=base_config.RETRAIN_FREQ, step=1, key="retrain")

    target_vol = st.slider("Target Vol", min_value=0.05, max_value=0.50, value=base_config.TARGET_VOL, step=0.01, key="target_vol")
    max_leverage = st.slider("Max Leverage", min_value=1.0, max_value=5.0, value=base_config.MAX_LEVERAGE, step=0.1, key="max_leverage")
    stop_loss = st.slider("Stop Loss (drawdown)", min_value=-0.50, max_value=-0.05, value=base_config.STOP_LOSS, step=0.01, key="stop_loss")

    transaction_cost = st.slider("Transaction Cost (round-trip)", min_value=0.0, max_value=0.01, value=base_config.TRANSACTION_COST, step=0.0005, key="transaction_cost")
    slippage = st.slider("Slippage", min_value=0.0, max_value=0.01, value=base_config.SLIPPAGE, step=0.0005, key="slippage")

    run_button = st.button("Run Backtest", type="primary")


if run_button:
    overrides = {
        "TICKER": ticker,
        "START_DATE": start_date,
        "LOOKBACK": lookback,
        "HORIZON": horizon,
        "RETRAIN_FREQ": retrain,
        "TARGET_VOL": target_vol,
        "MAX_LEVERAGE": max_leverage,
        "STOP_LOSS": stop_loss,
        "TRANSACTION_COST": transaction_cost,
        "SLIPPAGE": slippage,
    }

    with st.spinner("Running backtest..."):
        try:
            results = run_backtest_with_overrides(overrides)
        except Exception as exc:
            st.error(f"Backtest failed: {exc}")
            st.stop()

    st.success("Backtest complete")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Return", f"{results['total_return']:+.2%}")
    col1.metric("Sharpe", f"{results['sharpe']:+.2f}")
    col2.metric("Max Drawdown", f"{results['max_dd']:+.2%}")
    col2.metric("Win Rate", f"{results['win_rate']:.1%}")
    col3.metric("IC", f"{results['ic']:+.4f}")
    col3.metric("IC p-value", f"{results['ic_pvalue']:.4f}")

    if os.path.exists("results/backtest_results.csv"):
        results_df = pd.read_csv("results/backtest_results.csv")
        if "return" in results_df.columns:
            equity = compute_equity_curve(results_df["return"].values)
            equity_df = pd.DataFrame({"equity": equity})
            st.subheader("Equity Curve")
            st.line_chart(equity_df)

            peak = np.maximum.accumulate(equity)
            drawdown = (equity - peak) / peak
            dd_df = pd.DataFrame({"drawdown": drawdown})
            st.subheader("Drawdown")
            st.line_chart(dd_df)

        st.subheader("Latest Results")
        st.dataframe(results_df.tail(20), use_container_width=True)
    else:
        st.warning("Results CSV not found. Run the backtest first.")

else:
    st.info("Adjust parameters in the sidebar and click Run Backtest.")

st.divider()
st.header("Parameter Sweep")

sweep_options = {
    "LOOKBACK": (200, 1000, 100),
    "HORIZON": (5, 60, 5),
    "RETRAIN_FREQ": (5, 60, 5),
    "TARGET_VOL": (0.05, 0.50, 0.05),
    "MAX_LEVERAGE": (1.0, 5.0, 0.5),
    "STOP_LOSS": (-0.50, -0.05, 0.05),
    "TRANSACTION_COST": (0.0, 0.01, 0.001),
}

param_name = st.selectbox("Sweep Parameter", list(sweep_options.keys()), index=0)
min_val, max_val, step_val = sweep_options[param_name]

col_a, col_b, col_c = st.columns(3)
with col_a:
    sweep_start = st.number_input("Start", value=min_val)
with col_b:
    sweep_end = st.number_input("End", value=max_val)
with col_c:
    sweep_step = st.number_input("Step", value=step_val)

run_sweep = st.button("Run Sweep")

if run_sweep:
    if sweep_step == 0:
        st.error("Step must be non-zero.")
        st.stop()

    values = list(np.arange(sweep_start, sweep_end + sweep_step, sweep_step))
    if len(values) > 25:
        st.warning("Large sweep detected. Consider narrowing the range.")

    base_overrides = {
        "TICKER": ticker,
        "START_DATE": start_date,
        "LOOKBACK": lookback,
        "HORIZON": horizon,
        "RETRAIN_FREQ": retrain,
        "TARGET_VOL": target_vol,
        "MAX_LEVERAGE": max_leverage,
        "STOP_LOSS": stop_loss,
        "TRANSACTION_COST": transaction_cost,
        "SLIPPAGE": slippage,
    }

    results_rows = []
    progress = st.progress(0)

    for idx, value in enumerate(values, start=1):
        overrides = dict(base_overrides)
        overrides[param_name] = value

        try:
            results = run_backtest_with_overrides(overrides)
            results_rows.append({
                param_name: value,
                "total_return": results["total_return"],
                "sharpe": results["sharpe"],
                "max_dd": results["max_dd"],
                "ic": results["ic"],
                "ic_pvalue": results["ic_pvalue"],
                "win_rate": results["win_rate"],
                "profit_factor": results["profit_factor"],
            })
        except Exception as exc:
            results_rows.append({
                param_name: value,
                "error": str(exc),
            })

        progress.progress(min(idx / len(values), 1.0))

    sweep_df = pd.DataFrame(results_rows)
    st.session_state["sweep_df"] = sweep_df
    st.subheader("Sweep Results")
    st.dataframe(sweep_df, use_container_width=True)

    if "sharpe" in sweep_df.columns:
        best = sweep_df.dropna(subset=["sharpe"]).sort_values("sharpe", ascending=False).head(1)
        if not best.empty:
            st.subheader("Best by Sharpe")
            st.dataframe(best, use_container_width=True)

    if not sweep_df.empty:
        csv_data = sweep_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Sweep CSV",
            data=csv_data,
            file_name="sweep_results.csv",
            mime="text/csv",
        )

st.divider()
st.header("2D Parameter Grid")

grid_params = list(sweep_options.keys())
grid_x = st.selectbox("X Parameter", grid_params, index=0, key="grid_x")
grid_y = st.selectbox("Y Parameter", grid_params, index=1, key="grid_y")

col_g1, col_g2, col_g3 = st.columns(3)
with col_g1:
    grid_x_start = st.number_input("X Start", value=sweep_options[grid_x][0], key="grid_x_start")
with col_g2:
    grid_x_end = st.number_input("X End", value=sweep_options[grid_x][1], key="grid_x_end")
with col_g3:
    grid_x_step = st.number_input("X Step", value=sweep_options[grid_x][2], key="grid_x_step")

col_g4, col_g5, col_g6 = st.columns(3)
with col_g4:
    grid_y_start = st.number_input("Y Start", value=sweep_options[grid_y][0], key="grid_y_start")
with col_g5:
    grid_y_end = st.number_input("Y End", value=sweep_options[grid_y][1], key="grid_y_end")
with col_g6:
    grid_y_step = st.number_input("Y Step", value=sweep_options[grid_y][2], key="grid_y_step")

run_grid = st.button("Run 2D Grid")

if run_grid:
    if grid_x_step == 0 or grid_y_step == 0:
        st.error("Step must be non-zero.")
        st.stop()

    x_values = list(np.arange(grid_x_start, grid_x_end + grid_x_step, grid_x_step))
    y_values = list(np.arange(grid_y_start, grid_y_end + grid_y_step, grid_y_step))
    total_runs = len(x_values) * len(y_values)

    if total_runs > 100:
        st.warning("Large grid detected. Consider narrowing the range.")

    base_overrides = {
        "TICKER": ticker,
        "START_DATE": start_date,
        "LOOKBACK": lookback,
        "HORIZON": horizon,
        "RETRAIN_FREQ": retrain,
        "TARGET_VOL": target_vol,
        "MAX_LEVERAGE": max_leverage,
        "STOP_LOSS": stop_loss,
        "TRANSACTION_COST": transaction_cost,
        "SLIPPAGE": slippage,
    }

    grid_rows = []
    progress = st.progress(0)
    run_count = 0

    for x_val in x_values:
        for y_val in y_values:
            run_count += 1
            overrides = dict(base_overrides)
            overrides[grid_x] = x_val
            overrides[grid_y] = y_val
            try:
                results = run_backtest_with_overrides(overrides)
                grid_rows.append({
                    grid_x: x_val,
                    grid_y: y_val,
                    "sharpe": results["sharpe"],
                    "ic": results["ic"],
                    "total_return": results["total_return"],
                })
            except Exception as exc:
                grid_rows.append({
                    grid_x: x_val,
                    grid_y: y_val,
                    "error": str(exc),
                })
            progress.progress(min(run_count / total_runs, 1.0))

    grid_df = pd.DataFrame(grid_rows)
    st.session_state["grid_df"] = grid_df
    st.subheader("2D Grid Results")
    st.dataframe(grid_df, use_container_width=True)

    if "sharpe" in grid_df.columns:
        heatmap = grid_df.pivot(index=grid_y, columns=grid_x, values="sharpe")
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(heatmap.values, aspect="auto", origin="lower")
        ax.set_title("Sharpe Heatmap")
        ax.set_xlabel(grid_x)
        ax.set_ylabel(grid_y)
        ax.set_xticks(range(len(heatmap.columns)))
        ax.set_yticks(range(len(heatmap.index)))
        ax.set_xticklabels([f"{v:.3g}" for v in heatmap.columns])
        ax.set_yticklabels([f"{v:.3g}" for v in heatmap.index])
        fig.colorbar(im, ax=ax, shrink=0.8)
        st.pyplot(fig)

    if not grid_df.empty:
        csv_data = grid_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Grid CSV",
            data=csv_data,
            file_name="grid_results.csv",
            mime="text/csv",
        )

st.divider()
st.header("Compare Presets")

presets_for_compare = load_presets()
preset_choices = sorted(presets_for_compare.keys())
selected_presets = st.multiselect("Select Presets", preset_choices)
run_compare = st.button("Run Preset Comparison")

if run_compare:
    if not selected_presets:
        st.warning("Select at least one preset.")
        st.stop()

    compare_rows = []
    progress = st.progress(0)
    for idx, name in enumerate(selected_presets, start=1):
        preset = presets_for_compare.get(name, {})
        overrides = {
            "TICKER": preset.get("ticker", ticker),
            "START_DATE": preset.get("start_date", start_date),
            "LOOKBACK": preset.get("lookback", lookback),
            "HORIZON": preset.get("horizon", horizon),
            "RETRAIN_FREQ": preset.get("retrain", retrain),
            "TARGET_VOL": preset.get("target_vol", target_vol),
            "MAX_LEVERAGE": preset.get("max_leverage", max_leverage),
            "STOP_LOSS": preset.get("stop_loss", stop_loss),
            "TRANSACTION_COST": preset.get("transaction_cost", transaction_cost),
            "SLIPPAGE": preset.get("slippage", slippage),
        }
        try:
            results = run_backtest_with_overrides(overrides)
            compare_rows.append({
                "preset": name,
                "total_return": results["total_return"],
                "sharpe": results["sharpe"],
                "max_dd": results["max_dd"],
                "ic": results["ic"],
                "ic_pvalue": results["ic_pvalue"],
            })
        except Exception as exc:
            compare_rows.append({
                "preset": name,
                "error": str(exc),
            })
        progress.progress(min(idx / len(selected_presets), 1.0))

    compare_df = pd.DataFrame(compare_rows)
    st.subheader("Preset Comparison")
    st.dataframe(compare_df, use_container_width=True)

    if not compare_df.empty:
        csv_data = compare_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Comparison CSV",
            data=csv_data,
            file_name="preset_comparison.csv",
            mime="text/csv",
        )
