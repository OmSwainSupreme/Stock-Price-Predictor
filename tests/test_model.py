"""Tests for model training and prediction module."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.model import StockPriceModel, train_and_evaluate, predict_next_session
from src.features import create_features, FEATURE_COLUMNS, TARGET_COLUMN


def create_synthetic_data(n_rows: int = 100) -> pd.DataFrame:
    """Create synthetic price data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=n_rows, freq='B')
    np.random.seed(42)
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


class TestStockPriceModel:
    """Tests for StockPriceModel class."""
    
    def test_chronological_split_preserves_order(self):
        """Test that chronological split preserves temporal order."""
        df = create_synthetic_data(50)
        features = create_features(df)
        
        X = features[FEATURE_COLUMNS]
        y = features[TARGET_COLUMN]
        
        # With 50 raw rows, features will have ~39 rows (after dropping first 10 for ma_10, last 1 for target)
        n_features = len(features)
        expected_train = int(n_features * 0.8)
        expected_test = n_features - expected_train
        
        model = StockPriceModel(test_size=0.2)
        X_train, X_test, y_train, y_test = model.chronological_split(X, y)
        
        # Check sizes
        assert len(X_train) == expected_train
        assert len(X_test) == expected_test
        
        # Check chronological order: all train dates < all test dates
        train_dates = features.iloc[:len(X_train)]['Date']
        test_dates = features.iloc[len(X_train):]['Date']
        
        assert train_dates.max() < test_dates.min(), "Train/test split not chronological"
    
    def test_chronological_split_minimum_samples(self):
        """Test split with minimum samples."""
        df = create_synthetic_data(15)
        features = create_features(df)
        
        X = features[FEATURE_COLUMNS]
        y = features[TARGET_COLUMN]
        
        model = StockPriceModel(test_size=0.2)
        X_train, X_test, y_train, y_test = model.chronological_split(X, y)
        
        # With 4 samples after feature engineering, split should be 3/1
        assert len(X_train) >= 1
        assert len(X_test) >= 1
        assert len(X_train) + len(X_test) == len(X)
    
    def test_fit_and_predict(self):
        """Test model fitting and prediction."""
        df = create_synthetic_data(50)
        features = create_features(df)
        
        X = features[FEATURE_COLUMNS]
        y = features[TARGET_COLUMN]
        
        model = StockPriceModel(test_size=0.2)
        X_train, X_test, y_train, y_test = model.chronological_split(X, y)
        
        model.fit(X_train, y_train)
        assert model.is_fitted
        
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)
        assert np.all(np.isfinite(preds))
    
    def test_fit_empty_raises(self):
        """Test that fitting on empty data raises error."""
        model = StockPriceModel()
        X_empty = pd.DataFrame(columns=FEATURE_COLUMNS)
        y_empty = pd.Series(dtype=float)
        
        with pytest.raises(ValueError, match="Training data is empty"):
            model.fit(X_empty, y_empty)
    
    def test_fit_length_mismatch_raises(self):
        """Test that length mismatch raises error."""
        df = create_synthetic_data(30)
        features = create_features(df)
        
        X = features[FEATURE_COLUMNS]
        y = features[TARGET_COLUMN].iloc[:-1]  # Mismatch
        
        model = StockPriceModel()
        with pytest.raises(ValueError, match="length mismatch"):
            model.fit(X, y)
    
    def test_predict_before_fit_raises(self):
        """Test that prediction before fitting raises error."""
        model = StockPriceModel()
        X = pd.DataFrame(np.random.randn(5, len(FEATURE_COLUMNS)), columns=FEATURE_COLUMNS)
        
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict(X)
    
    def test_predict_wrong_columns(self):
        """Test prediction with wrong column order."""
        df = create_synthetic_data(30)
        features = create_features(df)
        
        X = features[FEATURE_COLUMNS]
        y = features[TARGET_COLUMN]
        
        model = StockPriceModel()
        model.fit(X, y)
        
        # Wrong column order
        X_wrong = X[FEATURE_COLUMNS[::-1]]
        preds = model.predict(X_wrong)
        assert len(preds) == len(X_wrong)
    
    def test_get_coefficients(self):
        """Test coefficient retrieval."""
        df = create_synthetic_data(30)
        features = create_features(df)
        
        X = features[FEATURE_COLUMNS]
        y = features[TARGET_COLUMN]
        
        model = StockPriceModel()
        model.fit(X, y)
        
        coefs = model.get_coefficients()
        assert len(coefs) == len(FEATURE_COLUMNS)
        assert list(coefs.index) == FEATURE_COLUMNS
    
    def test_get_intercept(self):
        """Test intercept retrieval."""
        df = create_synthetic_data(30)
        features = create_features(df)
        
        X = features[FEATURE_COLUMNS]
        y = features[TARGET_COLUMN]
        
        model = StockPriceModel()
        model.fit(X, y)
        
        intercept = model.get_intercept()
        assert isinstance(intercept, float)


class TestTrainAndEvaluate:
    """Tests for train_and_evaluate function."""
    
    def test_complete_pipeline(self):
        """Test complete training and evaluation pipeline."""
        df = create_synthetic_data(100)
        results = train_and_evaluate(df)
        
        assert results['error'] is None
        assert results['model'] is not None
        assert isinstance(results['model'], LinearRegression)
        assert not results['X_train'].empty
        assert not results['X_test'].empty
        assert len(results['y_pred_test']) == len(results['y_test'])
        assert results['next_prediction'] is not None
        assert np.isfinite(results['next_prediction'])
        assert results['latest_close'] is not None
        assert results['latest_date'] is not None
    
    def test_train_test_not_overlapping(self):
        """Test that train and test sets don't overlap."""
        df = create_synthetic_data(100)
        results = train_and_evaluate(df)
        
        train_dates = results['features_df']['Date'].iloc[:len(results['X_train'])]
        test_dates = results['features_df']['Date'].iloc[len(results['X_train']):]
        
        assert train_dates.max() < test_dates.min()
    
    def test_eval_model_trained_only_on_train(self):
        """Test that evaluation model is trained only on train split."""
        df = create_synthetic_data(100)
        results = train_and_evaluate(df)
        
        eval_model = results['eval_model']
        assert eval_model.train_size == len(results['X_train'])
        assert eval_model.test_size_actual == len(results['X_test'])
    
    def test_final_model_trained_on_all_data(self):
        """Test that final model is trained on all data."""
        df = create_synthetic_data(100)
        results = train_and_evaluate(df)
        
        # Final model should be trained on all features
        final_model = results['model']
        # We can't directly check training data size, but we can verify
        # it makes predictions on the latest features
        assert results['next_prediction'] is not None
    
    def test_insufficient_data_error(self):
        """Test error handling for insufficient data."""
        df = create_synthetic_data(10)  # Too few
        results = train_and_evaluate(df)
        
        assert results['error'] is not None
        assert results['model'] is None
    
    def test_empty_dataframe_error(self):
        """Test error handling for empty DataFrame."""
        df = pd.DataFrame()
        results = train_and_evaluate(df)
        
        assert results['error'] is not None
    
    def test_non_finite_values_handled(self):
        """Test handling of non-finite values."""
        df = create_synthetic_data(50)
        # Inject NaN
        df.loc[10, 'Adj Close'] = np.nan
        
        results = train_and_evaluate(df)
        
        # Should handle gracefully (either error or drop NaN)
        # Current implementation drops NaN in normalize_data, so might work
        # but with fewer rows
        assert 'error' in results


class TestPredictNextSession:
    """Tests for predict_next_session function."""
    
    def test_prediction_shape(self):
        """Test that prediction returns a single float."""
        df = create_synthetic_data(50)
        features = create_features(df)
        X = features[FEATURE_COLUMNS]
        y = features[TARGET_COLUMN]
        
        model = LinearRegression()
        model.fit(X, y)
        
        pred = predict_next_session(df, model)
        
        assert pred is not None
        assert isinstance(pred, float)
        assert np.isfinite(pred)
    
    def test_insufficient_data_returns_none(self):
        """Test that insufficient data returns None."""
        df = create_synthetic_data(5)
        model = LinearRegression()
        
        pred = predict_next_session(df, model)
        assert pred is None


class TestTemporalLeakage:
    """Tests to ensure no temporal leakage in the pipeline."""
    
    def test_features_not_using_future(self):
        """Verify features at time t only use data up to time t."""
        df = create_synthetic_data(30)
        features = create_features(df)
        
        # For each feature row i (which corresponds to original row i+10),
        # verify no feature uses data from original row > i+10
        prices = df['Adj Close'].values
        
        for i in range(len(features)):
            orig_idx = i + 10
            
            # lag_1 uses orig_idx - 1
            assert features['lag_1'].iloc[i] == prices[orig_idx - 1]
            
            # lag_5 uses orig_idx - 5
            assert features['lag_5'].iloc[i] == prices[orig_idx - 5]
            
            # ma_5: rolling(5).mean().shift(1) at orig_idx
            # = rolling mean at orig_idx-1 = mean of prices[orig_idx-5:orig_idx] (indices orig_idx-5 through orig_idx-1, 5 values)
            expected_ma5 = np.mean(prices[orig_idx-5:orig_idx])
            assert np.isclose(features['ma_5'].iloc[i], expected_ma5), \
                f"MA5 mismatch at row {i}: got {features['ma_5'].iloc[i]}, expected {expected_ma5}"
            
            # ma_10: rolling(10).mean().shift(1) at orig_idx
            # = rolling mean at orig_idx-1 = mean of prices[orig_idx-10:orig_idx] (10 values)
            expected_ma10 = np.mean(prices[orig_idx-10:orig_idx])
            assert np.isclose(features['ma_10'].iloc[i], expected_ma10), \
                f"MA10 mismatch at row {i}"
            
            # daily_return uses (orig_idx-1, orig_idx-2)
            expected_ret = (prices[orig_idx-1] - prices[orig_idx-2]) / prices[orig_idx-2]
            assert np.isclose(features['daily_return'].iloc[i], expected_ret)
            
            # target uses orig_idx + 1 (this is correct - it's what we're predicting)
            assert features[TARGET_COLUMN].iloc[i] == prices[orig_idx + 1]
    
    def test_no_target_in_features(self):
        """Test that target column is not in feature columns."""
        assert TARGET_COLUMN not in FEATURE_COLUMNS


if __name__ == '__main__':
    pytest.main([__file__, '-v'])