"""
Production portfolio management with regime-aware risk controls.

Enhancements:
1. Adaptive position sizing based on market regime
2. Confidence-weighted position scaling
3. Multi-layer drawdown protection
4. Volatility normalization
5. Better transaction cost modeling
"""

import numpy as np
import pandas as pd
from logger import setup_logger

logger = setup_logger(__name__)


def compute_position_production(signal, realized_vol, regime, config):
    """
    Compute position size with regime awareness.
    
    Args:
        signal: Model prediction (float)
        realized_vol: Realized volatility (float)
        regime: Market regime (int: 0=BULL, 1=BEAR, 2=RANGE, 3=TRANSITION)
        config: Configuration object
    
    Returns:
        Clipped position size (-1 to +1 for long/short, or 0 to 1 for long-only)
    """
    
    # ==========================================
    # 1. BASE SIGNAL (Scale by regime confidence)
    # ==========================================
    
    regime_confidence = {
        0: 0.8,   # BULL: High confidence in momentum signals
        1: 0.6,   # BEAR: Lower confidence, more defensive
        2: 0.5,   # RANGE: Weak signals, reduce size
        3: 0.3    # TRANSITION: Very defensive
    }
    
    confidence = regime_confidence.get(regime, 0.5)
    adj_signal = signal * confidence
    
    # ==========================================
    # 2. VOLATILITY NORMALIZATION
    # ==========================================
    
    # Target volatility: normalize position to achieve consistent vol
    target_vol = config.TARGET_VOL
    
    if realized_vol > 1e-8:
        vol_adjusted_signal = adj_signal * (target_vol / realized_vol)
    else:
        vol_adjusted_signal = adj_signal
    
    # ==========================================
    # 3. TANH SQUASHING (Smooth, non-linear scaling)
    # ==========================================
    
    position = np.tanh(vol_adjusted_signal * 2)  # Scale by 2 to get to [-1, 1]
    
    # ==========================================
    # 4. HARD LEVERAGE CAP
    # ==========================================
    
    max_leverage = config.MAX_LEVERAGE
    position = np.clip(position, -max_leverage, max_leverage)
    
    # ==========================================
    # 5. REGIME-SPECIFIC CAPS
    # ==========================================
    
    per_position_caps = {
        0: 1.0,   # BULL: Full position allowed
        1: 0.5,   # BEAR: Max 50% position (more conservative)
        2: 0.3,   # RANGE: Max 30% position
        3: 0.2    # TRANSITION: Max 20% position
    }
    
    cap = per_position_caps.get(regime, 0.5)
    position = np.clip(position, -cap, cap)
    
    return position


def apply_transaction_costs_production(position_old, position_new, price, config):
    """
    Calculate transaction costs from position changes.
    
    Args:
        position_old: Previous position
        position_new: New position
        price: Current price
        config: Configuration object
    
    Returns:
        Transaction cost in dollars
    """
    
    turnover = abs(position_new - position_old)
    tc = config.TRANSACTION_COST
    slippage = config.SLIPPAGE
    
    cost = turnover * price * (tc + slippage)
    
    return cost


def check_drawdown_stop_production(equity_curve, config):
    """
    Check if stop loss should be triggered.
    
    Args:
        equity_curve: Cumulative equity series
        config: Configuration object
    
    Returns:
        (should_stop, max_dd_pct)
    """
    
    if len(equity_curve) < 2:
        return False, 0
    
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    max_dd = drawdown.min()
    
    stop_loss = config.STOP_LOSS
    
    if max_dd < stop_loss:
        return True, max_dd
    
    return False, max_dd


def compute_regime_aware_position_sizing(signals, regimes, realized_vols, config):
    """
    Vectorized position sizing computation.
    
    Args:
        signals: Array of model predictions
        regimes: Array of regime classifications
        realized_vols: Array of realized volatilities
        config: Configuration object
    
    Returns:
        Array of positions
    """
    
    positions = np.zeros(len(signals))
    
    for i in range(len(signals)):
        positions[i] = compute_position_production(
            signals[i],
            realized_vols[i],
            regimes[i],
            config
        )
    
    return positions


def estimate_performance_confidence(ic_values, win_rate, profit_factor):
    """
    Estimate prediction confidence from performance metrics.
    
    Returns confidence score [0, 1]:
    - 0.0: No confidence (random predictions)
    - 0.5: Some signal
    - 1.0: Perfect predictions
    """
    
    # Information coefficient confidence (correlation-based)
    ic_confidence = min(abs(ic_values.mean()), 0.3) / 0.3 if len(ic_values) > 0 else 0
    
    # Win rate confidence (away from 50%)
    wr_confidence = abs(win_rate - 0.5) * 2
    
    # Profit factor confidence
    pf_confidence = (profit_factor - 1.0) / 3.0 if profit_factor > 1.0 else 0
    
    overall_confidence = (ic_confidence + wr_confidence + pf_confidence) / 3
    
    return np.clip(overall_confidence, 0, 1)


def apply_confidence_adjustment(positions, confidence):
    """
    Scale positions by confidence score.
    
    Low confidence → smaller positions
    High confidence → larger positions
    """
    
    return positions * confidence


if __name__ == "__main__":
    import config
    
    # Test position sizing
    config.validate_config()
    
    signals = np.random.randn(100) * 0.1
    regimes = np.random.randint(0, 4, 100)
    vols = np.abs(np.random.randn(100)) * 0.02 + 0.01
    
    positions = compute_regime_aware_position_sizing(signals, regimes, vols, config)
    
    print(f"Position sizing test:")
    print(f"  Mean position: {positions.mean():.3f}")
    print(f"  Max position: {positions.max():.3f}")
    print(f"  Min position: {positions.min():.3f}")
    print(f"  Position std: {positions.std():.3f}")
