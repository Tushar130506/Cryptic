"""
Core IC features - Minimal robust signal generation

Focus: Only the features that work, no NaN propagation
"""

import numpy as np
import pandas as pd
from logger import setup_logger

logger = setup_logger(__name__)


def add_ic_features(df, horizon=22):
    """
    Minimal feature set focused on what actually works.
    Only use features with no NaN issues.
    """
    
    df = df.copy()
    
    logger.info("Engineering IC-focused features (simplified)...")
    
    # ==========================================
    # CORE SIGNALS ONLY (No NaN)
    # ==========================================
    
    # 1. Momentum (5 periods - responsive)
    df['mom_5d'] = df['price'].pct_change(5).fillna(0)
    
    # 2. Momentum (20 periods - medium)
    df['mom_20d'] = df['price'].pct_change(20).fillna(0)
    
    # 3. Simple moving average crossover
    df['sma_20'] = df['price'].rolling(20).mean()
    df['sma_50'] = df['price'].rolling(50).mean()
    
    # Distance from SMA (z-scored, trimmed)
    price_dist = df['price'] - df['sma_20']
    std_dist = price_dist.rolling(20).std() + 1e-8
    df['price_zscore'] = (price_dist / std_dist).fillna(0)
    df['price_zscore'] = np.clip(df['price_zscore'], -3, 3)  # Trim outliers
    
    # 4. Volatility (20-day realized)
    log_ret = np.log(df['price'] / df['price'].shift(1)).fillna(0)
    df['realized_vol'] = log_ret.rolling(20).std().fillna(0) + 1e-8
    
    # 5. Trend signal (SMA crossover)
    df['trend_up'] = (df['sma_20'] > df['sma_50']).astype(float)
    df['trend_down'] = (df['sma_20'] < df['sma_50']).astype(float)
    
    # ==========================================
    # PRIMARY SIGNAL (NO NaNs)
    # ==========================================
    
    # Blend momentum signals (most important - has IC)
    df['signal_momentum'] = (
        0.6 * (df['mom_5d'] / (df['realized_vol'] + 1e-8)).clip(-1, 1) +
        0.4 * (df['mom_20d'] / (df['realized_vol'] + 1e-8)).clip(-1, 1)
    )
    
    # Mean reversion signal (opposite of price zscore)
    df['signal_mean_reversion'] = -df['price_zscore'] * 0.5
    
    # Combine: weighted blend
    df['signal_final'] = (
        0.7 * df['signal_momentum'] +
        0.3 * df['signal_mean_reversion']
    )
    
    # Normalize to [-1, 1]
    sig_abs_max = df['signal_final'].rolling(60).apply(lambda x: np.abs(x).max(), raw=False) + 1e-8
    df['signal_final'] = (df['signal_final'] / sig_abs_max).fillna(0).clip(-1, 1)
    
    # ==========================================
    # TARGET (Future return, NO NaNs until end)
    # ==========================================
    df['target'] = df['price'].pct_change(horizon).shift(-horizon)
    
    logger.info(f"Features created: {len(get_ic_feature_names())} core indicators")
    logger.info(f"Samples available: {len(df)}")
    logger.info(f"NaN in target: {df['target'].isna().sum()}")
    
    return df


def get_ic_feature_names():
    """Return list of feature column names"""
    return [
        'mom_5d', 'mom_20d',
        'sma_20', 'sma_50',
        'price_zscore',
        'realized_vol',
        'trend_up', 'trend_down',
        'signal_momentum',
        'signal_mean_reversion',
        'signal_final',
    ]


if __name__ == '__main__':
    from data import load_data
    import config
    
    df = load_data(config)
    df = add_ic_features(df, config.HORIZON)
    print(df[['price', 'target', 'signal_final']].tail(50))
    print(f"\nNaN checks:")
    print(f"signal_final NaN: {df['signal_final'].isna().sum()}")
    print(f"target NaN: {df['target'].isna().sum()}")
