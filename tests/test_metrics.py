"""Tests for metrics calculation module."""

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    calculate_metrics,
    calculate_baseline_mae,
    get_baseline_predictions,
    evaluate_model,
    format_metrics,
    create_prediction_summary,
    create_test_results_df,
)


class TestCalculateMetrics:
    """Tests for calculate_metrics function."""
    
    def test_basic_metrics(self):
        """Test basic metric calculations."""
        y_true = np.array([100, 102, 101, 103, 105])
        y_pred = np.array([101, 101, 102, 104, 104])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'r2' in metrics
        
        # Manual calculations
        expected_mae = np.mean(np.abs(y_true - y_pred))
        expected_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        assert np.isclose(metrics['mae'], expected_mae)
        assert np.isclose(metrics['rmse'], expected_rmse)
        assert np.isclose(metrics['r2'], 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    
    def test_perfect_prediction(self):
        """Test metrics for perfect predictions."""
        y_true = np.array([100, 102, 101, 103, 105])
        y_pred = np.array([100, 102, 101, 103, 105])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert metrics['mae'] == 0.0
        assert metrics['rmse'] == 0.0
        assert metrics['r2'] == 1.0
    
    def test_empty_arrays(self):
        """Test handling of empty arrays."""
        y_true = np.array([])
        y_pred = np.array([])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert np.isnan(metrics['mae'])
        assert np.isnan(metrics['rmse'])
        assert np.isnan(metrics['r2'])
    
    def test_length_mismatch(self):
        """Test handling of length mismatch."""
        y_true = np.array([100, 102, 101])
        y_pred = np.array([101, 101])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert np.isnan(metrics['mae'])
        assert np.isnan(metrics['rmse'])
        assert np.isnan(metrics['r2'])
    
    def test_nan_values_filtered(self):
        """Test that NaN values are filtered."""
        y_true = np.array([100, 102, np.nan, 103, 105])
        y_pred = np.array([101, 101, 102, np.nan, 104])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        # Should compute on 3 valid pairs (indices 0, 1, 4)
        valid_true = np.array([100, 102, 105])
        valid_pred = np.array([101, 101, 104])
        expected_mae = np.mean(np.abs(valid_true - valid_pred))
        
        assert np.isclose(metrics['mae'], expected_mae)
    
    def test_all_nan_returns_nan(self):
        """Test that all NaN returns NaN."""
        y_true = np.array([np.nan, np.nan])
        y_pred = np.array([np.nan, np.nan])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert np.isnan(metrics['mae'])
        assert np.isnan(metrics['rmse'])
        assert np.isnan(metrics['r2'])
    
    def test_constant_target_r2(self):
        """Test R² behavior with constant target."""
        y_true = np.array([100, 100, 100, 100])
        y_pred = np.array([101, 101, 101, 101])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        # R² should be 0 or negative when target is constant and predictions are wrong
        # sklearn returns 0.0 in this case
        assert metrics['r2'] <= 0 or np.isclose(metrics['r2'], 0.0)
    
    def test_negative_r2_possible(self):
        """Test that R² can be negative (worse than horizontal line)."""
        y_true = np.array([100, 102, 101, 103, 105])
        y_pred = np.array([200, 200, 200, 200, 200])  # Very bad predictions
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert metrics['r2'] < 0


class TestCalculateBaselineMAE:
    """Tests for calculate_baseline_mae function."""
    
    def test_baseline_calculation(self):
        """Test baseline MAE calculation."""
        y_true = np.array([102, 101, 103, 105, 104])
        y_prev = np.array([100, 102, 101, 103, 105])  # Previous closes
        
        baseline = calculate_baseline_mae(y_true, y_prev)
        
        expected = np.mean(np.abs(y_true - y_prev))
        assert np.isclose(baseline, expected)
    
    def test_empty_arrays(self):
        """Test empty arrays."""
        baseline = calculate_baseline_mae(np.array([]), np.array([]))
        assert np.isnan(baseline)
    
    def test_length_mismatch(self):
        """Test length mismatch."""
        baseline = calculate_baseline_mae(np.array([1, 2, 3]), np.array([1, 2]))
        assert np.isnan(baseline)
    
    def test_nan_handling(self):
        """Test NaN handling."""
        y_true = np.array([100, np.nan, 102])
        y_prev = np.array([99, 100, np.nan])
        
        baseline = calculate_baseline_mae(y_true, y_prev)
        
        # Only index 0 is valid
        expected = np.abs(100 - 99)
        assert np.isclose(baseline, expected)


class TestGetBaselinePredictions:
    """Tests for get_baseline_predictions function."""
    
    def test_from_features(self):
        """Test getting baseline from features DataFrame."""
        features = pd.DataFrame({
            'lag_1': [100, 102, 101, 103],
            'lag_2': [99, 100, 102, 101],
            'lag_3': [98, 99, 100, 102],
            'lag_5': [95, 98, 99, 100],
            'ma_5': [98, 100, 101, 102],
            'ma_10': [97, 99, 100, 101],
            'daily_return': [0.01, -0.01, 0.02, -0.01],
        })
        
        baseline = get_baseline_predictions(features)
        
        expected = np.array([100, 102, 101, 103])
        assert np.array_equal(baseline, expected)
    
    def test_empty_features(self):
        """Test empty features."""
        features = pd.DataFrame()
        baseline = get_baseline_predictions(features)
        assert len(baseline) == 0
    
    def test_missing_lag_1(self):
        """Test missing lag_1 column."""
        features = pd.DataFrame({'lag_2': [1, 2, 3]})
        baseline = get_baseline_predictions(features)
        assert len(baseline) == 0


class TestEvaluateModel:
    """Tests for evaluate_model function."""
    
    def test_complete_evaluation(self):
        """Test complete evaluation with baseline."""
        y_test = np.array([102, 101, 103, 105, 104])
        y_pred = np.array([101, 102, 102, 104, 105])
        features_test = pd.DataFrame({'lag_1': [100, 102, 101, 103, 105]})
        
        result = evaluate_model(y_test, y_pred, features_test)
        
        assert 'model' in result
        assert 'baseline_mae' in result
        assert 'comparison' in result
        
        # Model metrics
        assert 'mae' in result['model']
        assert 'rmse' in result['model']
        assert 'r2' in result['model']
        
        # Baseline
        assert not np.isnan(result['baseline_mae'])
        
        # Comparison
        assert 'improvement_pct' in result['comparison']
        assert 'model_better' in result['comparison']
    
    def test_no_features_test(self):
        """Test evaluation without features_test."""
        y_test = np.array([102, 101, 103])
        y_pred = np.array([101, 102, 102])
        
        result = evaluate_model(y_test, y_pred, None)
        
        assert 'model' in result
        assert np.isnan(result['baseline_mae'])
        assert np.isnan(result['comparison']['improvement_pct'])
        assert result['comparison']['model_better'] is None
    
    def test_improvement_calculation(self):
        """Test improvement percentage calculation."""
        # Model MAE = 1, Baseline MAE = 2 -> 50% improvement
        y_test = np.array([102, 101, 103, 105])
        y_pred = np.array([101, 102, 102, 104])  # Errors: 1, 1, 1, 1 -> MAE = 1
        features_test = pd.DataFrame({'lag_1': [100, 100, 100, 100]})  # Errors: 2, 1, 3, 5 -> MAE = 2.75
        
        result = evaluate_model(y_test, y_pred, features_test)
        
        assert result['comparison']['model_better'] == True
        assert result['comparison']['improvement_pct'] > 0


class TestFormatMetrics:
    """Tests for format_metrics function."""
    
    def test_formatting(self):
        """Test metric formatting."""
        metrics = {
            'model': {'mae': 1.5, 'rmse': 2.0, 'r2': 0.85},
            'baseline_mae': 2.5,
            'comparison': {'improvement_pct': 40.0, 'model_better': True},
        }
        
        formatted = format_metrics(metrics, '$')
        
        assert formatted['mae'] == '1.50 $'
        assert formatted['rmse'] == '2.00 $'
        assert formatted['r2'] == '0.8500'  # fmt uses 4 decimal places for values < 1
        assert formatted['baseline_mae'] == '2.50 $'
        assert formatted['improvement_pct'] == '40.00%'
        assert formatted['model_better'] == 'Yes'
    
    def test_nan_formatting(self):
        """Test NaN formatting."""
        metrics = {
            'model': {'mae': np.nan, 'rmse': np.nan, 'r2': np.nan},
            'baseline_mae': np.nan,
            'comparison': {'improvement_pct': np.nan, 'model_better': None},
        }
        
        formatted = format_metrics(metrics, '$')
        
        assert formatted['mae'] == 'N/A'
        assert formatted['rmse'] == 'N/A'
        assert formatted['r2'] == 'N/A'
        assert formatted['baseline_mae'] == 'N/A'
        assert formatted['improvement_pct'] == 'N/A'
        assert formatted['model_better'] == 'N/A'


class TestCreatePredictionSummary:
    """Tests for create_prediction_summary function."""
    
    def test_summary_creation(self):
        """Test prediction summary DataFrame creation."""
        latest_close = 150.0
        next_pred = 152.5
        metrics = {
            'model': {'mae': 1.5, 'rmse': 2.0, 'r2': 0.85},
            'baseline_mae': 2.5,
            'comparison': {'improvement_pct': 40.0, 'model_better': True},
        }
        latest_date = pd.Timestamp('2024-01-15')
        
        summary = create_prediction_summary(
            latest_close, next_pred, metrics, latest_date, 'Linear Regression', '$'
        )
        
        assert len(summary) == 1
        assert summary['model'].iloc[0] == 'Linear Regression'
        assert summary['data_through_date'].iloc[0] == '2024-01-15'
        assert summary['latest_close'].iloc[0] == 150.0
        assert summary['estimated_next_close'].iloc[0] == 152.5
        assert summary['absolute_change'].iloc[0] == 2.5
        # percent_change is rounded to 2 decimal places in create_prediction_summary
        assert summary['percent_change'].iloc[0] == round(2.5 / 150.0 * 100, 2)  # 1.67
        assert summary['currency'].iloc[0] == '$'
    
    def test_zero_latest_close(self):
        """Test handling of zero latest close."""
        latest_close = 0.0
        next_pred = 1.0
        metrics = {'model': {}, 'baseline_mae': np.nan, 'comparison': {}}
        latest_date = pd.Timestamp('2024-01-15')
        
        summary = create_prediction_summary(
            latest_close, next_pred, metrics, latest_date, 'Test', '$'
        )
        
        assert np.isnan(summary['percent_change'].iloc[0])


class TestCreateTestResultsDF:
    """Tests for create_test_results_df function."""
    
    def test_results_df_creation(self):
        """Test test results DataFrame creation."""
        dates = pd.date_range('2024-01-01', periods=5, freq='B')
        y_test = np.array([100, 102, 101, 103, 105])
        y_pred = np.array([101, 101, 102, 104, 104])
        
        df = create_test_results_df(pd.Series(dates), y_test, y_pred)
        
        assert len(df) == 5
        assert list(df.columns) == ['Date', 'Actual', 'Predicted']
        assert np.array_equal(df['Actual'].values, y_test)
        assert np.array_equal(df['Predicted'].values, y_pred)
        assert np.array_equal(df['Date'].values, dates.values)
    
    def test_length_mismatch(self):
        """Test length mismatch handling."""
        dates = pd.date_range('2024-01-01', periods=5, freq='B')
        y_test = np.array([100, 102, 101])
        y_pred = np.array([101, 101, 102, 104, 104])
        
        df = create_test_results_df(pd.Series(dates), y_test, y_pred)
        
        assert df.empty


if __name__ == '__main__':
    pytest.main([__file__, '-v'])