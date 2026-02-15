# Troubleshooting

Common issues and fixes for Cryptic.

## 1) Streamlit command not found

**Symptom**: `streamlit` is not recognized.

**Fix**:

```bash
pip install streamlit
```

## 2) Data download fails

**Symptom**: `No data retrieved` or connection errors.

**Fix**:

- Check internet connection
- Verify ticker symbol
- Try a later start date

## 3) Results folder missing

**Symptom**: Dashboard cannot find results CSV.

**Fix**:

- Run the backtest first: `python main.py`
- Ensure `results/` is created

## 4) IC or Sharpe looks wrong

**Symptom**: IC negative or Sharpe extreme.

**Fix**:

- Reduce horizon to 10–22
- Keep lookback between 300–500
- Re-run stress tests for validation

## 5) Stop loss triggers too early

**Symptom**: Few test periods, early stop loss.

**Fix**:

- Increase stop loss threshold (e.g., -0.25)
- Reduce leverage
- Use shorter horizon

## 6) Slow runs

**Symptom**: Stress tests take too long.

**Fix**:

- Reduce grid size
- Limit assets to BTC-USD only
- Run fewer parameter cases

## 7) Font/glyph warning in plots

**Symptom**: Matplotlib warnings about glyphs.

**Fix**:

- Safe to ignore (does not affect output)

## 8) KeyboardInterrupt during stress test

**Symptom**: Stress test stops mid-run.

**Fix**:

- Partial results are saved in `results/stress_test_results.csv`
- Re-run stress test to complete remaining cases
