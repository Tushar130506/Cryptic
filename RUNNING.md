# Running Cryptic

This guide shows how to run the backtest and the Streamlit dashboard.

## 1) Install Dependencies

```bash
pip install -r requirements.txt
```

## 2) Run the Backtest

```bash
python main.py
```

Outputs:

- results/backtest_results.csv
- results/dashboard.png

## 3) Run the Streamlit Dashboard

```bash
streamlit run ui/streamlit_app.py
```

### What You Can Tune

- Ticker, start date
- Lookback, horizon, retrain frequency
- Target vol, leverage, stop loss
- Transaction cost, slippage

### Advanced Tools

- Preset save/load
- Parameter sweep (1D)
- 2D grid heatmap
- Preset comparison

## 4) Run Tests

```bash
pytest -q
```

## Troubleshooting

- If Streamlit is missing:
  pip install streamlit
- If results folder is missing, run the backtest first.
- If data download fails, check the ticker and start date.
