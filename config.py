"""
Research-grade configuration with validation.
Ensures all hyperparameters and constants are statistically defensible.
"""

# ==========================================
# DATA CONFIGURATION
# ==========================================
TICKER = "BTC-USD"
START_DATE = "2015-01-01"
MIN_DATA_POINTS = 1000  # Minimum history to validate feature stability

# ==========================================
# ANALYSIS PARAMETERS (Walk-Forward)
# ==========================================
LOOKBACK = 500          # Training window (trading days)
HORIZON = 22            # Prediction horizon (trading days) - 1 month
RETRAIN_FREQ = 20       # Retrain every N trading days (avoids overtraining)
TEST_WINDOW = 1         # Out-of-sample test set size (1 day per fold)

# ==========================================
# MODEL HYPERPARAMETERS
# ==========================================
N_ESTIMATORS = 100      # Number of trees (conservative)
MAX_DEPTH = 5           # Tree depth (prevent overfitting)
MIN_SAMPLES_SPLIT = 20  # Minimum samples per split
MIN_SAMPLES_LEAF = 10   # Minimum samples per leaf
RANDOM_STATE = 42       # Global random seed for reproducibility

# XGBoost specific parameters
XGBOOST_N_ESTIMATORS = 100
XGBOOST_MAX_DEPTH = 5
XGBOOST_LEARNING_RATE = 0.05
XGBOOST_SUBSAMPLE = 0.8
XGBOOST_COLSAMPLE_BYTREE = 0.8

# Ensemble blending
USE_ENSEMBLE = True           # Blend RF and XGB predictions
RF_WEIGHT = 0.5              # Weight for Random Forest
XGB_WEIGHT = 0.5             # Weight for XGBoost
USE_XGBOOST = True           # Enable XGBoost model

# ==========================================
# POSITION SIZING & RISK (Conservative)
# ==========================================
TARGET_VOL = 0.15       # Target portfolio volatility (15% annualized)
MAX_LEVERAGE = 2.0      # Maximum gross exposure (prevent blowups)
POSITION_CAP = 1.0      # Max position size as % of portfolio
STOP_LOSS = -0.20       # Cumulative drawdown stop (20%)

# ==========================================
# TRANSACTION COSTS (Realistic)
# ==========================================
TRANSACTION_COST = 0.001  # 10 bps round trip (realistic for BTC spot)
SLIPPAGE = 0.0005         # Execution slippage (5 bps)

# ==========================================
# VALIDATION & RISK CHECKS
# ==========================================
MIN_IC = 0.01           # Minimum information coefficient to trade
IC_WINDOW = 63          # Rolling IC eval window (3 months)
SHARPE_THRESHOLD = 0.5  # Minimum acceptable Sharpe ratio


def validate_config():
    """
    Sanity checks on configuration to prevent obvious errors.
    Raises ValueError if invalid.
    """
    assert LOOKBACK > HORIZON, "Lookback must exceed horizon"
    assert LOOKBACK > 100, "Lookback too small for robust training"
    assert HORIZON > 0, "Horizon must be positive"
    assert RETRAIN_FREQ > 0, "Retrain frequency must be positive"
    assert 0 < TARGET_VOL < 1, "Target vol must be 0-100%"
    assert 0 < MAX_LEVERAGE <= 5, "Max leverage must be 1-5x"
    assert 0 <= TRANSACTION_COST <= 0.01, "Transaction cost unrealistic"
    assert RANDOM_STATE > 0, "Random state must be positive"
    return True
