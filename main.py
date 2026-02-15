"""
Production Main Entry Point

Executes clean, reproducible walk-forward backtest pipeline with:
- Fixed random seeds for reproducibility
- Macro-aware feature engineering
- RF + XGB ensemble model
- Comprehensive backtesting with walk-forward validation
- Transaction costs and risk management
- Dashboard generation
"""

import sys
import os
import warnings
import numpy as np
from logger import setup_logger

# Production imports
import config
from data import load_data
from features_ic import add_ic_features, get_ic_feature_names
from backtest import run_backtest_ic_test
from visualization_dashboard import PerformanceDashboard

logger = setup_logger(__name__)

# Suppress known benign ResourceWarning from sklearn's joblib backend
warnings.filterwarnings(
    "ignore",
    category=ResourceWarning,
    module=r"sklearn\.utils\.parallel"
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"sklearn\.utils\.parallel"
)


def set_random_seed(seed):
    """Set all random seeds for reproducibility"""
    np.random.seed(seed)
    import random
    random.seed(seed)
    logger.info(f"Random seed set to {seed}")


def main():
    """Execute production backtest pipeline"""
    try:
        logger.info("\n" + "="*70)
        logger.info("QUANT BLOCK - PRODUCTION BACKTEST")
        logger.info("="*70)
        
        # Setup
        set_random_seed(config.RANDOM_STATE)
        config.validate_config()
        
        logger.info(f"Configuration:")
        logger.info(f"  - Ticker: {config.TICKER}")
        logger.info(f"  - Period: {config.START_DATE} to 2026-12-31")
        logger.info(f"  - Lookback: {config.LOOKBACK} days")
        logger.info(f"  - Horizon: {config.HORIZON} days")
        logger.info(f"  - Target Vol: {config.TARGET_VOL:.1%}")
        logger.info(f"  - Transaction Cost: {config.TRANSACTION_COST:.2%}")
        
        # Load data
        logger.info("\n[1/4] Loading market data...")
        df = load_data(config)
        logger.info(f"Loaded {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")
        
        # Add features
        logger.info("[2/4] Engineering macro-aware features...")
        df = add_ic_features(df, config.HORIZON)
        logger.info(f"Added {len(get_ic_feature_names())} macro features")
        
        # Run backtest
        logger.info("[3/4] Running walk-forward backtest...")
        results = run_backtest_ic_test(df, config)
        
        # Generate dashboard
        logger.info("[4/4] Generating dashboard...")
        if os.path.exists('results/backtest_results.csv'):
            import pandas as pd
            results_df = pd.read_csv('results/backtest_results.csv')
            dashboard = PerformanceDashboard(results_df)
            dashboard.create_dashboard()
            dashboard.save_dashboard('results/dashboard.png')
            logger.info("Dashboard saved to results/dashboard.png")
        
        logger.info("\n" + "="*70)
        logger.info("BACKTEST COMPLETE")
        logger.info("="*70)
        logger.info(f"Results:")
        logger.info(f"  - Total Return: {results['total_return']:+.2%}")
        logger.info(f"  - Sharpe Ratio: {results['sharpe']:+.2f}")
        logger.info(f"  - Max Drawdown: {results['max_dd']:+.2%}")
        logger.info(f"  - Win Rate: {results['win_rate']:.1%}")
        logger.info(f"  - IC: {results['ic']:+.4f} (p={results['ic_pvalue']:.4f})")
        logger.info("\nOutput files:")
        logger.info("  - results/backtest_results.csv (detailed results)")
        logger.info("  - results/dashboard.png (performance dashboard)")
        logger.info("="*70 + "\n")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

