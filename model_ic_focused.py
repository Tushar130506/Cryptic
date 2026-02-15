"""
Simplified IC-focused model using macro features.

Strategy to rebuild IC:
1. Use only 10 strongest features (not 20)
2. Remove regime-specific models (causes overfitting to classification)
3. Use universal RF + XGB blend
4. Focus on signal correlation, not regime adaptation
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from logger import setup_logger

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

logger = setup_logger(__name__)


class MacroAwareModel:
    """
    Simplified model focusing on IC reconstruction.
    Single universal model instead of per-regime models.
    """
    
    def __init__(self, config):
        """Initialize macro-aware model."""
        self.config = config
        self.scaler = StandardScaler()
        self.scaler_fitted = False
        
        # Initialize models
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            bootstrap=True,
            n_jobs=-1,
            random_state=42
        )
        
        if XGBOOST_AVAILABLE:
            self.xgb_model = XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                alpha=0.1,
                lambda_=1.0,
                random_state=42,
                verbosity=0
            )
        else:
            self.xgb_model = None
        
        logger.info("MacroAwareModel initialized: RF + XGB blend")
    
    def fit(self, X_train, y_train):
        """Fit models on training data."""
        
        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.scaler_fitted = True
        
        # Train RF
        logger.info(f"  Training RF on {len(X_train)} samples...")
        self.rf_model.fit(X_train_scaled, y_train)
        
        # Train XGB
        if XGBOOST_AVAILABLE and self.xgb_model is not None:
            logger.info(f"  Training XGB on {len(X_train)} samples...")
            self.xgb_model.fit(X_train_scaled, y_train)
    
    def predict(self, X_test):
        """Generate predictions."""
        
        if not self.scaler_fitted:
            logger.error("Model not fitted!")
            return None
        
        X_test_scaled = self.scaler.transform(X_test)
        
        pred_rf = self.rf_model.predict(X_test_scaled)
        
        if XGBOOST_AVAILABLE and self.xgb_model is not None:
            pred_xgb = self.xgb_model.predict(X_test_scaled)
            # Equal weight blend
            return 0.5 * pred_rf + 0.5 * pred_xgb
        else:
            return pred_rf


if __name__ == "__main__":
    from data import load_data
    from features_ic import add_ic_features, get_ic_feature_names
    import config
    
    config.validate_config()
    df = load_data(config)
    df = add_ic_features(df, config.HORIZON)
    
    feature_names = get_ic_feature_names()
    
    train_end = len(df) // 2
    X_train = df[feature_names].iloc[:train_end].values
    y_train = df['target'].iloc[:train_end].values
    X_test = df[feature_names].iloc[train_end:].values
    y_test = df['target'].iloc[train_end:].values
    
    model = MacroAwareModel(config)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    ic = np.corrcoef(preds, y_test)[0, 1]
    print(f"Test IC: {ic:+.4f}")
