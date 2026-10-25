"""Metrics calculation module for stock price prediction.

Computes evaluation metrics and baseline comparisons.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate regression metrics.
    
    Args:
        y_true: Actual values.
        y_pred: Predicted values.
        
    Returns:
        Dictionary with MAE, RMSE, R².
    """
    metrics = {
        'mae': np.nan,
        'rmse': np.nan,
        'r2': np.nan,
    }
    
    if len(y_true) == 0 or len(y_pred) == 0:
        logger.warning("Empty arrays provided for metrics calculation")
        return metrics
    
    if len(y_true) != len(y_pred):
        logger.error(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
        return metrics
    
    # Check for non-finite values
    if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
        logger.warning("Non-finite values in metrics calculation")
        # Filter to finite values only
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if not np.any(mask):
            return metrics
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    
    try:
        metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
        metrics['rmse'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        metrics['r2'] = float(r2_score(y_true, y_pred))
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
    
    return metrics


def calculate_baseline_mae(y_true: np.ndarray, y_prev: np.ndarray) -> float:
    """Calculate baseline MAE using previous close as prediction.
    
    Args:
        y_true: Actual next-session closing prices.
        y_prev: Previous session closing prices (lag_1).
        
    Returns:
        Baseline MAE, or NaN if calculation fails.
    """
    if len(y_true) == 0 or len(y_prev) == 0:
        return np.nan
    
    if len(y_true) != len(y_prev):
        logger.warning(f"Baseline length mismatch: {len(y_true)} vs {len(y_prev)}")
        return np.nan
    
    # Filter to finite values
    mask = np.isfinite(y_true) & np.isfinite(y_prev)
    if not np.any(mask):
        return np.nan
    
    y_true_f = y_true[mask]
    y_prev_f = y_prev[mask]
    
    try:
        baseline_mae = float(mean_absolute_error(y_true_f, y_prev_f))
        return baseline_mae
    except Exception as e:
        logger.error(f"Error calculating baseline MAE: {e}")
        return np.nan


def get_baseline_predictions(features_df: pd.DataFrame) -> np.ndarray:
    """Get baseline predictions (lag_1) from features DataFrame.
    
    Args:
        features_df: DataFrame from features.create_features().
        
    Returns:
        Array of baseline predictions (previous close).
    """
    if features_df.empty or 'lag_1' not in features_df.columns:
        return np.array([])
    
    return features_df['lag_1'].values


def evaluate_model(y_test: np.ndarray, y_pred_test: np.ndarray, 
                   features_test: Optional[pd.DataFrame] = None) -> Dict:
    """Complete model evaluation including baseline comparison.
    
    Args:
        y_test: Actual test targets.
        y_pred_test: Model predictions on test set.
        features_test: Test features DataFrame (for baseline).
        
    Returns:
        Dictionary with all metrics and comparison.
    """
    # Model metrics
    model_metrics = calculate_metrics(y_test, y_pred_test)
    
    # Baseline metrics
    baseline_mae = np.nan
    if features_test is not None and not features_test.empty:
        baseline_preds = get_baseline_predictions(features_test)
        baseline_mae = calculate_baseline_mae(y_test, baseline_preds)
    else:
        # If no features_test, we can't compute baseline properly
        logger.warning("features_test not provided, cannot compute baseline MAE")
    
    # Comparison
    comparison = {}
    if not np.isnan(model_metrics['mae']) and not np.isnan(baseline_mae):
        if baseline_mae > 0:
            improvement_pct = ((baseline_mae - model_metrics['mae']) / baseline_mae) * 100
            comparison['improvement_pct'] = improvement_pct
            comparison['model_better'] = model_metrics['mae'] < baseline_mae
        else:
            comparison['improvement_pct'] = 0.0
            comparison['model_better'] = False
    else:
        comparison['improvement_pct'] = np.nan
        comparison['model_better'] = None
    
    return {
        'model': model_metrics,
        'baseline_mae': baseline_mae,
        'comparison': comparison,
    }


def format_metrics(metrics: Dict, currency_symbol: str = '$') -> Dict[str, str]:
    """Format metrics for display.
    
    Args:
        metrics: Dictionary from evaluate_model().
        currency_symbol: Currency symbol for display.
        
    Returns:
        Dictionary with formatted strings.
    """
    model = metrics.get('model', {})
    baseline_mae = metrics.get('baseline_mae', np.nan)
    comparison = metrics.get('comparison', {})
    
    def fmt(val: float, suffix: str = '') -> str:
        if np.isnan(val):
            return 'N/A'
        if abs(val) >= 100:
            return f'{val:,.2f}{suffix}'
        elif abs(val) >= 1:
            return f'{val:.2f}{suffix}'
        else:
            return f'{val:.4f}{suffix}'
    
    return {
        'mae': fmt(model.get('mae', np.nan), f' {currency_symbol}'),
        'rmse': fmt(model.get('rmse', np.nan), f' {currency_symbol}'),
        'r2': fmt(model.get('r2', np.nan)),
        'baseline_mae': fmt(baseline_mae, f' {currency_symbol}'),
        'improvement_pct': fmt(comparison.get('improvement_pct', np.nan), '%'),
        'model_better': 'Yes' if comparison.get('model_better') else ('No' if comparison.get('model_better') is not None else 'N/A'),
    }


def create_prediction_summary(
    latest_close: float,
    next_prediction: float,
    metrics: Dict,
    latest_date: pd.Timestamp,
    model_name: str = 'Linear Regression',
    currency_symbol: str = '$'
) -> pd.DataFrame:
    """Create a concise prediction summary DataFrame for download.
    
    Args:
        latest_close: Most recent observed closing price.
        next_prediction: Estimated next-session closing price.
        metrics: Evaluation metrics dictionary.
        latest_date: Date of latest observation.
        model_name: Name of the model used.
        currency_symbol: Currency symbol.
        
    Returns:
        Single-row DataFrame with summary information.
    """
    model_metrics = metrics.get('model', {})
    baseline_mae = metrics.get('baseline_mae', np.nan)
    comparison = metrics.get('comparison', {})
    
    abs_diff = next_prediction - latest_close
    pct_diff = (abs_diff / latest_close * 100) if latest_close != 0 else np.nan
    
    summary = {
        'model': model_name,
        'data_through_date': latest_date.strftime('%Y-%m-%d'),
        'latest_close': round(latest_close, 2),
        'estimated_next_close': round(next_prediction, 2),
        'absolute_change': round(abs_diff, 2),
        'percent_change': round(pct_diff, 2) if not np.isnan(pct_diff) else np.nan,
        'currency': currency_symbol,
        'mae': round(model_metrics.get('mae', np.nan), 4),
        'rmse': round(model_metrics.get('rmse', np.nan), 4),
        'r2': round(model_metrics.get('r2', np.nan), 4),
        'baseline_mae': round(baseline_mae, 4) if not np.isnan(baseline_mae) else np.nan,
        'improvement_pct': round(comparison.get('improvement_pct', np.nan), 2) if not np.isnan(comparison.get('improvement_pct', np.nan)) else np.nan,
    }
    
    return pd.DataFrame([summary])


def create_test_results_df(
    test_dates: pd.Series,
    y_test: np.ndarray,
    y_pred_test: np.ndarray
) -> pd.DataFrame:
    """Create DataFrame of actual vs predicted for test period.
    
    Args:
        test_dates: Dates for test observations.
        y_test: Actual values.
        y_pred_test: Predicted values.
        
    Returns:
        DataFrame with Date, Actual, Predicted columns.
    """
    if len(test_dates) != len(y_test) or len(y_test) != len(y_pred_test):
        logger.error("Length mismatch in test results")
        return pd.DataFrame()
    
    df = pd.DataFrame({
        'Date': test_dates.values,
        'Actual': y_test,
        'Predicted': y_pred_test,
    })
    
    return df