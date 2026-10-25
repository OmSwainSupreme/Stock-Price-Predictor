"""Model training and prediction module for stock price prediction.

Handles chronological splitting, training, test prediction, and next-session prediction.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator

from .features import FEATURE_COLUMNS, TARGET_COLUMN, create_features, prepare_features_and_target, reconstruct_latest_features

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockPriceModel:
    """Linear Regression model for stock price prediction with chronological splitting."""
    
    def __init__(self, test_size: float = 0.2):
        """Initialize the model.
        
        Args:
            test_size: Fraction of data to use for testing (chronological split).
        """
        self.test_size = test_size
        self.model: Optional[LinearRegression] = None
        self.is_fitted = False
        self.feature_columns = FEATURE_COLUMNS
        self.train_size: Optional[int] = None
        self.test_size_actual: Optional[int] = None
    
    def chronological_split(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data chronologically (not randomly).
        
        Args:
            X: Feature DataFrame.
            y: Target Series.
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        n = len(X)
        if n == 0:
            return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float)
        
        split_idx = int(n * (1 - self.test_size))
        # Ensure at least 1 sample in each split
        split_idx = max(1, min(split_idx, n - 1))
        
        X_train = X.iloc[:split_idx].copy()
        X_test = X.iloc[split_idx:].copy()
        y_train = y.iloc[:split_idx].copy()
        y_test = y.iloc[split_idx:].copy()
        
        self.train_size = len(X_train)
        self.test_size_actual = len(X_test)
        
        logger.info(f"Chronological split: train={self.train_size}, test={self.test_size_actual}")
        
        return X_train, X_test, y_train, y_test
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'StockPriceModel':
        """Fit the model on training data only.
        
        Args:
            X_train: Training features.
            y_train: Training targets.
            
        Returns:
            Self for chaining.
        """
        if X_train.empty or y_train.empty:
            raise ValueError("Training data is empty")
        
        if len(X_train) != len(y_train):
            raise ValueError(f"Feature/target length mismatch: {len(X_train)} vs {len(y_train)}")
        
        # Check for non-finite values
        if not np.all(np.isfinite(X_train.values)) or not np.all(np.isfinite(y_train.values)):
            raise ValueError("Training data contains non-finite values")
        
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        logger.info(f"Model fitted on {len(X_train)} training samples")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the fitted model.
        
        Args:
            X: Features to predict on.
            
        Returns:
            Array of predictions.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before prediction")
        
        if X.empty:
            return np.array([])
        
        # Ensure column order matches training
        X = X[self.feature_columns]
        
        predictions = self.model.predict(X)
        return predictions
    
    def get_coefficients(self) -> pd.Series:
        """Get model coefficients.
        
        Returns:
            Series with feature names and coefficients.
        """
        if not self.is_fitted or self.model is None:
            return pd.Series(dtype=float)
        
        return pd.Series(self.model.coef_, index=self.feature_columns)
    
    def get_intercept(self) -> float:
        """Get model intercept."""
        if not self.is_fitted or self.model is None:
            return 0.0
        return float(self.model.intercept_)


def train_and_evaluate(raw_df: pd.DataFrame, test_size: float = 0.2) -> dict:
    """Complete training and evaluation pipeline.
    
    Args:
        raw_df: Raw historical DataFrame from data_loader.
        test_size: Fraction for chronological test split.
        
    Returns:
        Dictionary with model, predictions, metrics, and metadata.
    """
    result = {
        'model': None,
        'X_train': pd.DataFrame(),
        'X_test': pd.DataFrame(),
        'y_train': pd.Series(dtype=float),
        'y_test': pd.Series(dtype=float),
        'y_pred_test': np.array([]),
        'features_df': pd.DataFrame(),
        'latest_features': pd.DataFrame(),
        'next_prediction': None,
        'latest_date': None,
        'latest_close': None,
        'error': None,
    }
    
    try:
        # Create features
        features_df = create_features(raw_df)
        result['features_df'] = features_df
        
        if features_df.empty:
            result['error'] = "Insufficient data for feature engineering (need at least 10 rows after cleaning)"
            return result
        
        # Prepare features and target
        X, y = prepare_features_and_target(features_df)
        
        if len(X) < 2:
            result['error'] = f"Insufficient samples after feature engineering: {len(X)}"
            return result
        
        # Initialize model
        model = StockPriceModel(test_size=test_size)
        
        # Chronological split
        X_train, X_test, y_train, y_test = model.chronological_split(X, y)
        
        if X_train.empty or X_test.empty:
            result['error'] = "Insufficient data for train/test split"
            return result
        
        # Fit on training data only
        model.fit(X_train, y_train)
        
        # Predict on test set
        y_pred_test = model.predict(X_test)
        
        # Reconstruct latest features for next-session prediction
        latest_features = reconstruct_latest_features(raw_df)
        
        # Fit final model on ALL data for deployment prediction
        final_model = LinearRegression()
        final_model.fit(X, y)
        
        # Predict next session
        next_prediction = None
        if not latest_features.empty:
            next_prediction = final_model.predict(latest_features)[0]
        
        # Get latest actual close and date
        price_col = 'Adj Close' if 'Adj Close' in raw_df.columns else 'Close'
        latest_close = raw_df[price_col].iloc[-1]
        latest_date = raw_df['Date'].iloc[-1]
        
        result.update({
            'model': final_model,  # Final model trained on all data
            'eval_model': model,   # Model trained on train split only (for evaluation)
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred_test': y_pred_test,
            'latest_features': latest_features,
            'next_prediction': next_prediction,
            'latest_date': latest_date,
            'latest_close': latest_close,
        })
        
    except Exception as e:
        logger.error(f"Error in train_and_evaluate: {e}")
        result['error'] = str(e)
    
    return result


def predict_next_session(raw_df: pd.DataFrame, model: BaseEstimator) -> Optional[float]:
    """Predict next trading session's closing price.
    
    Args:
        raw_df: Raw historical DataFrame.
        model: Fitted model (should be trained on all available data).
        
    Returns:
        Predicted next close price, or None if prediction fails.
    """
    try:
        latest_features = reconstruct_latest_features(raw_df)
        if latest_features.empty:
            return None
        
        prediction = model.predict(latest_features)
        return float(prediction[0])
    except Exception as e:
        logger.error(f"Error predicting next session: {e}")
        return None