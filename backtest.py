"""
Simplified backtest using macro features to validate IC reconstruction.

Tests macro-aware model on in-sample data to show IC improvement.
"""

import sys
import numpy as np
import pandas as pd
from scipy import stats

from logger import setup_logger
from data import load_data
from features_ic import add_ic_features, get_ic_feature_names
from model_ic_focused import MacroAwareModel
from portfolio_production import (
    compute_position_production,
    apply_transaction_costs_production,
    check_drawdown_stop_production
)

logger = setup_logger(__name__)


def run_backtest_ic_test(df, config):
    """
    Run backtest on macro-aware model to validate IC reconstruction.
    """
    
    logger.info("\n" + "=" * 70)
    logger.info("IC RECONSTRUCTION TEST (Macro-Aware Model)")
    logger.info("=" * 70)
    logger.info(f"Data points: {len(df)}")
    
    df = df.copy()
    feature_names = get_ic_feature_names()
    
    # Initialize model
    model = MacroAwareModel(config)
    
    # Tracking
    test_idx_list = []
    predictions_list = []
    actual_returns_list = []
    positions_list = []
    pnl_list = []
    equity_list = []
    
    position_prev = 0
    equity = 1.0
    last_retrain_idx = 0
    model_trained = False
    
    # ==========================================
    # WALK-FORWARD BACKTEST
    # ==========================================
    
    logger.info("Walk-forward validation:\n")
    
    for i in range(config.LOOKBACK + config.HORIZON, len(df)):
        
        # Retrain periodically
        if (i - last_retrain_idx) >= config.RETRAIN_FREQ or not model_trained:
            
            train_idx_start = max(0, i - config.LOOKBACK)
            train_idx_end = i
            
            X_train = df[feature_names].iloc[train_idx_start:train_idx_end].values
            y_train = df['target'].iloc[train_idx_start:train_idx_end].values
            
            # Clean NaN
            valid_mask = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
            X_train = X_train[valid_mask]
            y_train = y_train[valid_mask]
            
            if len(X_train) < 100:
                continue
            
            model.fit(X_train, y_train)
            last_retrain_idx = i
            model_trained = True
            
            if i % 100 == 0:
                logger.info(f"[{i}] Retrained on {len(X_train)} samples")
        
        # Get prediction
        X_test = df[feature_names].iloc[i:i+1].values
        
        if np.isnan(X_test).any():
            continue
        
        prediction = float(model.predict(X_test)[0])
        
        # Simple position sizing (no regime complexity)
        realized_vol = df['realized_vol'].iloc[i] if 'realized_vol' in df.columns else 0.01
        position = np.tanh(prediction / (realized_vol + 1e-8))
        position = np.clip(position, -1.0, 1.0)
        
        # Transaction cost
        price = df['price'].iloc[i]
        cost = apply_transaction_costs_production(position_prev, position, price, config)
        
        # Execute
        future_return = df['target'].iloc[i]
        pnl_pct = position * future_return * 100 - cost / price * 100
        
        equity = equity * (1 + pnl_pct / 100)
        
        # Track
        test_idx_list.append(i)
        predictions_list.append(prediction)
        actual_returns_list.append(future_return)
        positions_list.append(position)
        pnl_list.append(pnl_pct)
        equity_list.append(equity)
        
        position_prev = position
        
        # Stop loss
        should_stop, _ = check_drawdown_stop_production(pd.Series(equity_list), config)
        if should_stop:
            logger.warning(f"Stop loss triggered at index {i}")
            break
    
    # ==========================================
    # METRICS
    # ==========================================
    
    logger.info(f"\nBacktest complete: {len(test_idx_list)} test periods\n")
    
    returns_array = np.array(pnl_list) / 100
    equity_array = np.array(equity_list)
    predictions_array = np.array(predictions_list)
    actual_array = np.array(actual_returns_list)
    
    # Metrics
    total_return = (equity_array[-1] - 1) if len(equity_array) > 0 else 0
    daily_sharpe = returns_array.mean() / (returns_array.std() + 1e-8)
    sharpe = daily_sharpe * np.sqrt(252)
    
    peak = np.maximum.accumulate(equity_array)
    dd = (equity_array - peak) / peak
    max_dd = dd.min() if len(dd) > 0 else 0
    
    # IC statistics - FIXED: Calculate correlation across full period
    if len(predictions_array) > 2 and np.std(predictions_array) > 1e-8 and np.std(actual_array) > 1e-8:
        # Remove means for correlation
        pred_centered = predictions_array - np.mean(predictions_array)
        actual_centered = actual_array - np.mean(actual_array)
        
        # Calculate correlation coefficient properly
        ic_mean = np.dot(pred_centered, actual_centered) / (np.sqrt(np.sum(pred_centered**2)) * np.sqrt(np.sum(actual_centered**2)) + 1e-8)
        
        # Calculate t-stat and p-value
        r = ic_mean
        n = len(predictions_array)
        t_stat = r * np.sqrt(n - 2) / np.sqrt(1 - r**2 + 1e-8)
        ic_pvalue = 2 * (1 - stats.t.cdf(abs(t_stat), max(n - 2, 1)))
    else:
        ic_mean = 0.0
        ic_pvalue = 1.0
    
    ic_std = np.std(predictions_array - actual_array) if len(predictions_array) > 1 else 1.0
    
    # Win rate
    win_rate = (actual_array > 0).sum() / len(actual_array) if len(actual_array) > 0 else 0
    
    # Profit factor
    gross_profit = max((returns_array[returns_array > 0]).sum(), 1e-8)
    gross_loss = abs(min((returns_array[returns_array < 0]).sum(), 0))
    profit_factor = gross_profit / (gross_loss + 1e-8)
    
    # ==========================================
    # RESULTS
    # ==========================================
    
    logger.info("=" * 70)
    logger.info("IC RECONSTRUCTION RESULTS")
    logger.info("=" * 70)
    logger.info(f"\nTotal Return:           {total_return:+.2%}")
    logger.info(f"Sharpe Ratio:           {sharpe:+.2f}")
    logger.info(f"Max Drawdown:           {max_dd:+.2%}")
    logger.info(f"Win Rate:               {win_rate:.1%}")
    logger.info(f"Profit Factor:          {profit_factor:.2f}")
    logger.info(f"\nInformation Coefficient: {ic_mean:+.4f}")
    logger.info(f"IC Std Dev:             {ic_std:.4f}")
    logger.info(f"IC p-value:             {ic_pvalue:.4f}")
    logger.info(f"IC Significant:         {'YES' if ic_pvalue < 0.05 else 'NO'}")
    logger.info(f"\nTest Periods:           {len(test_idx_list)}")
    
    logger.info("\n" + "=" * 70)
    
    if ic_mean > 0:
        logger.info("SUCCESS: IC > 0 - Signal reconstructed!")
    else:
        logger.warning("WARNING: IC still <= 0 - Need more work")
    
    logger.info("=" * 70 + "\n")
    
    # Save results to CSV
    import os
    os.makedirs('results', exist_ok=True)
    
    results_df = pd.DataFrame({
        'test_idx': test_idx_list,
        'prediction': predictions_list,
        'actual_return': actual_returns_list,
        'position': positions_list,
        'pnl': pnl_list,
        'return': [r for r in pnl_list],
    })
    
    results_df.to_csv('results/backtest_results.csv', index=False)
    logger.info(f"Results saved to results/backtest_results.csv ({len(results_df)} rows)")
    
    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'ic': ic_mean,
        'ic_pvalue': ic_pvalue,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
    }


if __name__ == "__main__":
    import config
    
    config.validate_config()
    df = load_data(config)
    df = add_ic_features(df, config.HORIZON)
    
    run_backtest_ic_test(df, config)
