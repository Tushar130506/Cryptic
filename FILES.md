# File Overview

This document summarizes what each Python module does.

## Core pipeline

- backtest.py: walk-forward backtest loop, IC calculation, and result export.
- config.py: central configuration defaults and validation checks.
- data.py: data loading from Yahoo Finance with integrity checks.
- features_ic.py: feature engineering and target construction for IC.
- model_ic_focused.py: Random Forest + XGBoost ensemble model.
- main.py: end-to-end pipeline entry point.

## Portfolio and risk

- portfolio_production.py: position sizing, transaction costs, drawdown stops.

## Visualization

- visualization_dashboard.py: Matplotlib performance dashboard generator.

## UI

- ui/streamlit_app.py: Streamlit tuning dashboard with presets and sweeps.

## Stress testing

- stress_test.py: parameter and multi-asset stress tests with CSV output.

## Utilities

- logger.py: logging setup used across modules.

## Tests

- tests/test_config.py: configuration validation tests.
- tests/test_features_ic.py: feature engineering tests.
- tests/test_model.py: model training and prediction tests.
