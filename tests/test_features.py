"""Tests for feature engineering module."""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    create_features,
    prepare_features_and_target,
    get_latest_feature_row,
    reconstruct_latest_features,
    get_feature_columns,
    get_target_column,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)


def create_synthetic_data(n_rows: int = 50) -> pd.DataFrame:
    """Create synthetic price data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=n_rows, freq='B')  # Business days
    np.random.seed(42)
    # Random walk with drift
    returns = np.random.normal(0.0005, 0.015, n_rows)
    prices = 100 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices * np.random.uniform(0.99, 1.01, n_rows),
        'High': prices * np.random.uniform(1.00, 1.02, n_rows),
        'Low': prices * np.random.uniform(0.98, 1.00, n_rows),
        'Close': prices,
        'Adj Close': prices,
        'Volume': np.random.randint(1000000, 10000000, n_rows),
    })
    return df


class TestCreateFeatures:
    """Tests for create_features function."""
    
    def test_feature_columns_exist(self):
        """Test that all expected feature columns are created."""
        df = create_synthetic_data(30)
        features = create_features(df)
        
        assert not features.empty
        for col in FEATURE_COLUMNS:
            assert col in features.columns, f"Missing feature column: {col}"
        assert TARGET_COLUMN in features.columns
    
    def test_target_alignment(self):
        """Test that target is next session's close."""
        df = create_synthetic_data(20)
        features = create_features(df)
        
        price_col = 'Adj Close'
        # Target at row i should equal price at row i+1 in original data
        for i in range(len(features) - 1):
            expected_target = df[price_col].iloc[i + 1 + 10]  # +10 because first 10 rows dropped
            actual_target = features[TARGET_COLUMN].iloc[i]
            assert np.isclose(actual_target, expected_target, rtol=1e-10), \
                f"Target misalignment at row {i}"
    
    def test_lag_features_correct(self):
        """Test that lag features reference correct past values."""
        df = create_synthetic_data(20)
        features = create_features(df)
        
        price_col = 'Adj Close'
        prices = df[price_col].values
        
        # First row of features corresponds to row 10 of original (after dropping NaN)
        # lag_1 should be price at original row 9
        # lag_2 should be price at original row 8
        # etc.
        offset = 10  # rows dropped due to rolling windows
        
        for i in range(min(5, len(features))):
            orig_idx = i + offset
            assert np.isclose(features['lag_1'].iloc[i], prices[orig_idx - 1])
            assert np.isclose(features['lag_2'].iloc[i], prices[orig_idx - 2])
            assert np.isclose(features['lag_3'].iloc[i], prices[orig_idx - 3])
            assert np.isclose(features['lag_5'].iloc[i], prices[orig_idx - 5])
    
    def test_rolling_averages_no_leakage(self):
        """Test that moving averages don't use future values."""
        df = create_synthetic_data(20)
        features = create_features(df)
        
        price_col = 'Adj Close'
        prices = df[price_col].values
        offset = 10
        
        for i in range(min(5, len(features))):
            orig_idx = i + offset
            # ma_5 at row i: rolling(5).mean().shift(1) at orig_idx
            # = rolling mean at orig_idx-1 = mean of prices[orig_idx-6:orig_idx-1] (indices orig_idx-6 through orig_idx-2, 5 values)
            # Wait: rolling(window=5) at index j gives mean of j-4, j-3, j-2, j-1, j (5 values including j)
            # After shift(1), at index orig_idx we get value from orig_idx-1
            # rolling at orig_idx-1 gives mean of (orig_idx-1)-4 through (orig_idx-1) = orig_idx-5 through orig_idx-1
            expected_ma5 = np.mean(prices[orig_idx-5:orig_idx])
            assert np.isclose(features['ma_5'].iloc[i], expected_ma5, rtol=1e-10), \
                f"MA5 leakage at row {i}: got {features['ma_5'].iloc[i]}, expected {expected_ma5}"
            
            # ma_10 at row i: rolling(10).mean().shift(1) at orig_idx
            # = rolling mean at orig_idx-1 = mean of prices[orig_idx-10:orig_idx] (indices orig_idx-10 through orig_idx-1, 10 values)
            expected_ma10 = np.mean(prices[orig_idx-10:orig_idx])
            assert np.isclose(features['ma_10'].iloc[i], expected_ma10, rtol=1e-10), \
                f"MA10 leakage at row {i}"
    
    def test_daily_return_calculation(self):
        """Test daily return is calculated correctly without future info."""
        df = create_synthetic_data(20)
        features = create_features(df)
        
        price_col = 'Adj Close'
        prices = df[price_col].values
        offset = 10
        
        for i in range(min(5, len(features))):
            orig_idx = i + offset
            # daily_return = (price[t-1] - price[t-2]) / price[t-2]
            expected_return = (prices[orig_idx - 1] - prices[orig_idx - 2]) / prices[orig_idx - 2]
            assert np.isclose(features['daily_return'].iloc[i], expected_return, rtol=1e-10), \
                f"Daily return incorrect at row {i}"
    
    def test_rows_dropped_for_insufficient_history(self):
        """Test that rows with insufficient history are dropped."""
        df = create_synthetic_data(15)  # Only 15 rows
        features = create_features(df)
        
        # Need at least 10 rows for ma_10, plus 1 for lag_1, plus 1 for target shift
        # So at most 15 - 10 - 1 = 4 rows should remain (actually 15 - 11 = 4)
        # But we also need lag_5 (5 rows back), so first valid is at index 10
        # Then target needs index 11, so last valid is at index 13
        # So we should have 15 - 11 = 4 rows
        assert len(features) == 4, f"Expected 4 rows, got {len(features)}"
    
    def test_empty_input_returns_empty(self):
        """Test that empty DataFrame returns empty DataFrame."""
        df = pd.DataFrame()
        features = create_features(df)
        assert features.empty
    
    def test_insufficient_data_returns_empty(self):
        """Test that insufficient data returns empty DataFrame."""
        df = create_synthetic_data(5)  # Too few rows
        features = create_features(df)
        assert features.empty
    
    def test_no_nan_in_output(self):
        """Test that output has no NaN values in feature/target columns."""
        df = create_synthetic_data(30)
        features = create_features(df)
        
        for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
            assert not features[col].isna().any(), f"NaN found in {col}"
    
    def test_uses_adj_close_when_available(self):
        """Test that Adj Close is used when available."""
        df = create_synthetic_data(20)
        df['Adj Close'] = df['Close'] * 1.05  # Different from Close
        
        features = create_features(df)
        
        # Features should be based on Adj Close
        offset = 10
        assert np.isclose(features['lag_1'].iloc[0], df['Adj Close'].iloc[offset - 1])
        assert not np.isclose(features['lag_1'].iloc[0], df['Close'].iloc[offset - 1])


class TestPrepareFeaturesAndTarget:
    """Tests for prepare_features_and_target function."""
    
    def test_correct_split(self):
        """Test that X and y are correctly split."""
        df = create_synthetic_data(30)
        features = create_features(df)
        X, y = prepare_features_and_target(features)
        
        assert list(X.columns) == FEATURE_COLUMNS
        assert y.name == TARGET_COLUMN
        assert len(X) == len(y) == len(features)
    
    def test_empty_input(self):
        """Test empty input returns empty."""
        features = pd.DataFrame()
        X, y = prepare_features_and_target(features)
        assert X.empty
        assert y.empty


class TestReconstructLatestFeatures:
    """Tests for reconstruct_latest_features function."""
    
    def test_reconstruction_for_next_session(self):
        """Test that reconstructed features are correct for next-session prediction.
        
        Given raw data with prices p[0], p[1], ..., p[n-1] (n prices),
        reconstruct_latest_features should produce features to predict p[n] (next session):
        - lag_1 = p[n-1] (most recent)
        - lag_2 = p[n-2]
        - lag_3 = p[n-3]
        - lag_5 = p[n-5]
        - ma_5 = mean(p[n-6:n-1]) = mean of 5 sessions before most recent
        - ma_10 = mean(p[n-11:n-1]) = mean of 10 sessions before most recent
        - daily_return = (p[n-1] - p[n-2]) / p[n-2]
        """
        df = create_synthetic_data(30)
        prices = df['Adj Close'].values
        n = len(prices)
        
        reconstructed = reconstruct_latest_features(df)
        
        assert not reconstructed.empty
        assert np.isclose(reconstructed['lag_1'].values[0], prices[n-1])
        assert np.isclose(reconstructed['lag_2'].values[0], prices[n-2])
        assert np.isclose(reconstructed['lag_3'].values[0], prices[n-3])
        assert np.isclose(reconstructed['lag_5'].values[0], prices[n-5])
        assert np.isclose(reconstructed['ma_5'].values[0], np.mean(prices[n-6:n-1]))
        assert np.isclose(reconstructed['ma_10'].values[0], np.mean(prices[n-11:n-1]))
        assert np.isclose(reconstructed['daily_return'].values[0], (prices[n-1] - prices[n-2]) / prices[n-2])
    
    def test_insufficient_data_returns_empty(self):
        """Test that insufficient data returns empty DataFrame."""
        df = create_synthetic_data(5)
        reconstructed = reconstruct_latest_features(df)
        assert reconstructed.empty


class TestFeatureColumns:
    """Tests for feature column getters."""
    
    def test_get_feature_columns(self):
        cols = get_feature_columns()
        assert cols == FEATURE_COLUMNS
        # Ensure it's a copy, not the original list
        cols.append('test')
        assert 'test' not in get_feature_columns()
    
    def test_get_target_column(self):
        assert get_target_column() == TARGET_COLUMN


if __name__ == '__main__':
    pytest.main([__file__, '-v'])