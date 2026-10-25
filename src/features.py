"""Feature engineering module for stock price prediction.

Creates reproducible features from historical prices without future leakage.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple


# Feature column names (must match training order)
FEATURE_COLUMNS = [
    'lag_1',
    'lag_2',
    'lag_3',
    'lag_5',
    'ma_5',
    'ma_10',
    'daily_return',
]

TARGET_COLUMN = 'target_next_close'


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features from historical price data without future leakage.
    
    Args:
        df: DataFrame with columns ['Date', 'Close', 'Adj Close', ...].
            Must be sorted chronologically by Date.
            
    Returns:
        DataFrame with feature columns and target column added.
        Rows with insufficient history for lags/rolling windows are dropped.
    """
    if df.empty:
        return pd.DataFrame()
    
    # Work with adjusted close for feature engineering
    # Use 'Adj Close' if available, otherwise 'Close'
    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    
    if price_col not in df.columns:
        raise ValueError(f"Required price column '{price_col}' not found in DataFrame")
    
    # Ensure sorted by date
    df = df.sort_values('Date').reset_index(drop=True).copy()
    
    # Create feature DataFrame
    features = pd.DataFrame(index=df.index)
    features['Date'] = df['Date']
    
    # Lag features (previous closing prices)
    # lag_1 = previous day's adjusted close
    features['lag_1'] = df[price_col].shift(1)
    # lag_2 = 2 trading sessions earlier
    features['lag_2'] = df[price_col].shift(2)
    # lag_3 = 3 trading sessions earlier
    features['lag_3'] = df[price_col].shift(3)
    # lag_5 = 5 trading sessions earlier
    features['lag_5'] = df[price_col].shift(5)
    
    # Rolling moving averages (using only past data)
    # 5-session moving average of adjusted close
    features['ma_5'] = df[price_col].rolling(window=5, min_periods=5).mean().shift(1)
    # 10-session moving average of adjusted close
    features['ma_10'] = df[price_col].rolling(window=10, min_periods=10).mean().shift(1)
    
    # Daily return (percentage change from previous close to current close)
    # This uses only past information: (today_close - yesterday_close) / yesterday_close
    features['daily_return'] = df[price_col].pct_change(1).shift(1)
    
    # Target: next trading session's adjusted close
    features[TARGET_COLUMN] = df[price_col].shift(-1)
    
    # Drop rows where any feature or target is NaN (due to lagging/rolling)
    features = features.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)
    
    return features


def prepare_features_and_target(features_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split features DataFrame into X (features) and y (target).
    
    Args:
        features_df: DataFrame from create_features().
        
    Returns:
        Tuple of (X, y) where X contains feature columns and y is the target.
    """
    if features_df.empty:
        return pd.DataFrame(), pd.Series(dtype=float)
    
    X = features_df[FEATURE_COLUMNS].copy()
    y = features_df[TARGET_COLUMN].copy()
    
    return X, y


def get_latest_feature_row(features_df: pd.DataFrame) -> pd.DataFrame:
    """Get the latest feature row for next-session prediction.
    
    Args:
        features_df: DataFrame from create_features() (before dropping NaN for target).
        
    Returns:
        Single-row DataFrame with feature columns in correct order.
    """
    if features_df.empty:
        return pd.DataFrame()
    
    # Get the last row with all features available (before target drop)
    # We need to reconstruct features without dropping the last row for target
    latest_row = features_df.iloc[[-1]][FEATURE_COLUMNS].copy()
    
    return latest_row


def reconstruct_latest_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct the latest feature row from raw historical data.
    
    This ensures the exact same feature computation as during training.
    
    Args:
        raw_df: Original historical DataFrame with Date, Close, Adj Close.
        
    Returns:
        Single-row DataFrame with features for next-session prediction.
    """
    if raw_df.empty or len(raw_df) < 10:
        return pd.DataFrame()
    
    # Ensure sorted
    raw_df = raw_df.sort_values('Date').reset_index(drop=True)
    
    price_col = 'Adj Close' if 'Adj Close' in raw_df.columns else 'Close'
    prices = raw_df[price_col].values
    
    # Need at least 10 prices for ma_10
    if len(prices) < 10:
        return pd.DataFrame()
    
    # Compute features exactly as in create_features()
    latest_features = {}
    
    # Lag features (using the most recent prices)
    latest_features['lag_1'] = prices[-1]      # Most recent close
    latest_features['lag_2'] = prices[-2]      # 2 sessions ago
    latest_features['lag_3'] = prices[-3]      # 3 sessions ago
    latest_features['lag_5'] = prices[-5]      # 5 sessions ago
    
    # Moving averages (shifted by 1, so use prices up to -1)
    latest_features['ma_5'] = np.mean(prices[-6:-1])   # Sessions -6 to -2 (5 sessions before most recent)
    latest_features['ma_10'] = np.mean(prices[-11:-1]) # Sessions -11 to -2 (10 sessions before most recent)
    
    # Daily return: (most_recent - previous) / previous
    latest_features['daily_return'] = (prices[-1] - prices[-2]) / prices[-2]
    
    # Create DataFrame with correct column order
    feature_row = pd.DataFrame([latest_features], columns=FEATURE_COLUMNS)
    
    return feature_row


def get_feature_columns() -> List[str]:
    """Return the list of feature column names in training order."""
    return FEATURE_COLUMNS.copy()


def get_target_column() -> str:
    """Return the target column name."""
    return TARGET_COLUMN