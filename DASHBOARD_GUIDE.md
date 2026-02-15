# Cryptic Dashboard Guide

This guide explains every section in the Streamlit dashboard and how to use it.

## Launch

```bash
streamlit run ui/streamlit_app.py
```

## Sidebar Controls (Core Tuning)

- **Ticker**: Asset symbol (default BTC-USD).
- **Start Date**: Earliest date for data pull.
- **Lookback (days)**: Training window size for each walk-forward step.
- **Horizon (days)**: Prediction horizon for future returns.
- **Retrain Frequency (days)**: How often the model retrains.
- **Target Vol**: Risk target for position sizing.
- **Max Leverage**: Cap on position sizing.
- **Stop Loss (drawdown)**: Max cumulative drawdown before halting.
- **Transaction Cost**: Round-trip cost in decimal terms.
- **Slippage**: Execution slippage in decimal terms.

## Presets

- **Save Preset**: Stores current slider values to `ui/presets/presets.json`.
- **Load Preset**: Loads a previously saved preset into the UI.

Use presets to compare parameter sets quickly and reproduce results.

## Backtest Results

After running a backtest, the dashboard shows:

- **Total Return**: Cumulative return across test periods.
- **Sharpe**: Risk-adjusted return.
- **Max Drawdown**: Worst peak-to-trough decline.
- **Win Rate**: % of test periods with positive returns.
- **IC / IC p-value**: Correlation between predictions and actual returns.

Charts:

- **Equity Curve**: Growth of $1 over the test periods.
- **Drawdown**: Peak-to-trough losses over time.
- **Latest Results**: Table of the most recent periods.

## Parameter Sweep (1D)

Use this to scan a single parameter over a range:

- Select **Sweep Parameter** (e.g., LOOKBACK).
- Set **Start**, **End**, **Step**.
- Run sweep and review results table.
- **Download Sweep CSV** exports all results.

Use this to find stable ranges where Sharpe and IC remain strong.

## 2D Parameter Grid (Heatmap)

Run a two-parameter grid search and view a heatmap:

- Choose **X Parameter** and **Y Parameter**.
- Define ranges and step sizes.
- The heatmap shows **Sharpe** across the grid.

Interpretation:

- Bright regions indicate strong Sharpe.
- Flat regions indicate stable parameters.
- Sharp spikes suggest overfitting risk.

Download results with **Download Grid CSV**.

## Preset Comparison

Compare saved presets side-by-side:

- Select multiple presets.
- Run comparison.
- Review metrics and export to CSV.

This is the fastest way to compare multiple strategies.

## Tips

- Start with small sweeps to avoid long runtimes.
- Focus on stable parameter regions, not just peak Sharpe.
- Treat IC and p-value as primary signal validation metrics.
- Use presets to lock in working configurations.
