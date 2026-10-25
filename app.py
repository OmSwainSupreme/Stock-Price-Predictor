"""Stock Price Predictor - Streamlit Web Application.

Interactive dashboard for stock price prediction using linear regression.
"""

import io
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import (
    fetch_historical_data,
    get_currency_symbol,
    get_latest_market_date,
    PERIOD_MAP,
    validate_sufficient_data,
    validate_ticker,
)
from src.features import create_features, FEATURE_COLUMNS, TARGET_COLUMN
from src.metrics import (
    calculate_metrics,
    calculate_baseline_mae,
    create_prediction_summary,
    create_test_results_df,
    evaluate_model,
    format_metrics,
    get_baseline_predictions,
)
from src.model import StockPriceModel, train_and_evaluate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #6b7280;
        font-weight: 500;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1f2937;
    }
    .metric-delta-positive {
        color: #059669;
    }
    .metric-delta-negative {
        color: #dc2626;
    }
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1f2937;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
    }
    .info-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fef3c7;
        border: 1px solid #fcd34d;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .disclaimer {
        font-size: 0.8rem;
        color: #9ca3af;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #e5e7eb;
    }
    .stButton > button {
        background-color: #2563eb;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
    }
    .stButton > button:hover {
        background-color: #1d4ed8;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'last_ticker' not in st.session_state:
        st.session_state.last_ticker = ''
    if 'last_period' not in st.session_state:
        st.session_state.last_period = '1 year'


def render_sidebar() -> Tuple[str, str, bool]:
    """Render sidebar and return user inputs.
    
    Returns:
        Tuple of (ticker, period, run_analysis).
    """
    with st.sidebar:
        st.markdown("### 📊 Stock Selection")
        
        # Ticker input
        ticker = st.text_input(
            "Stock Ticker",
            value="AAPL",
            help="Enter a valid stock ticker (e.g., AAPL, MSFT, GOOGL, RELIANCE.NS, TCS.NS, INFY.NS)"
        ).strip().upper()
        
        # Example tickers
        st.markdown("**Example tickers:**")
        example_cols = st.columns(2)
        examples = ["AAPL", "MSFT", "GOOGL", "RELIANCE.NS", "TCS.NS", "INFY.NS"]
        for i, ex in enumerate(examples):
            with example_cols[i % 2]:
                if st.button(ex, key=f"ex_{ex}", use_container_width=True):
                    ticker = ex
                    st.rerun()
        
        st.divider()
        
        # Period selection
        period_label = st.selectbox(
            "Historical Period",
            options=list(PERIOD_MAP.keys()),
            index=3,  # Default to "1 year"
            help="Select the historical data period to download"
        )
        period = PERIOD_MAP[period_label]
        
        st.divider()
        
        # Run button
        run_analysis = st.button("🔍 Fetch Data & Run Analysis", type="primary", use_container_width=True)
        
        # Cache info
        st.markdown("---")
        st.caption("Data cached for 1 hour. Click button to refresh.")
        
        # Info about data source
        with st.expander("ℹ️ Data Source Info"):
            st.markdown("""
            - **Provider:** Yahoo Finance (via yfinance)
            - **Price type:** Adjusted Close (accounts for splits/dividends)
            - **Frequency:** Daily
            - **Availability:** Depends on market and Yahoo Finance coverage
            - **Note:** Not all tickers may be available
            """)
    
    return ticker, period, run_analysis


def render_header():
    """Render main header."""
    st.markdown('<div class="main-header">📈 Stock Price Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Linear Regression model for next-session closing price estimation using historical price features</div>', unsafe_allow_html=True)


def render_loading_state(message: str = "Fetching data and training model..."):
    """Render loading state."""
    with st.spinner(message):
        st.empty()


def render_error(message: str):
    """Render error message."""
    st.error(f"❌ {message}")


def render_warning(message: str):
    """Render warning message."""
    st.markdown(f'<div class="warning-box">⚠️ {message}</div>', unsafe_allow_html=True)


def render_info(message: str):
    """Render info message."""
    st.markdown(f'<div class="info-box">ℹ️ {message}</div>', unsafe_allow_html=True)


def render_metric_cards(latest_close: float, next_prediction: float, 
                        metrics: Dict, currency_symbol: str):
    """Render metric cards for key values."""
    col1, col2, col3, col4 = st.columns(4)
    
    # Latest Close
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Latest Close</div>
            <div class="metric-value">{currency_symbol}{latest_close:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Estimated Next Close
    with col2:
        abs_diff = next_prediction - latest_close
        pct_diff = (abs_diff / latest_close * 100) if latest_close != 0 else 0
        delta_class = "metric-delta-positive" if abs_diff >= 0 else "metric-delta-negative"
        delta_sign = "+" if abs_diff >= 0 else ""
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Est. Next Close</div>
            <div class="metric-value">{currency_symbol}{next_prediction:,.2f}</div>
            <div class="{delta_class}">{delta_sign}{abs_diff:,.2f} ({delta_sign}{pct_diff:.2f}%)</div>
        </div>
        """, unsafe_allow_html=True)
    
    # MAE
    with col3:
        mae_str = metrics.get('mae', 'N/A')
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Test MAE</div>
            <div class="metric-value">{mae_str}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # RMSE
    with col4:
        rmse_str = metrics.get('rmse', 'N/A')
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Test RMSE</div>
            <div class="metric-value">{rmse_str}</div>
        </div>
        """, unsafe_allow_html=True)


def render_historical_chart(data: pd.DataFrame, currency_symbol: str):
    """Render historical price chart with moving averages."""
    st.markdown('<div class="section-header">📊 Historical Price Trends</div>', unsafe_allow_html=True)
    
    if data.empty:
        st.warning("No data to display")
        return
    
    price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
    
    # Compute moving averages for display
    data = data.copy()
    data['MA_5'] = data[price_col].rolling(window=5, min_periods=1).mean()
    data['MA_10'] = data[price_col].rolling(window=10, min_periods=1).mean()
    
    fig = go.Figure()
    
    # Closing price
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data[price_col],
        mode='lines',
        name='Closing Price',
        line=dict(color='#1f2937', width=1.5),
        hovertemplate='%{x|%Y-%m-%d}<br>Close: %{y:.2f}<extra></extra>'
    ))
    
    # 5-session MA
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['MA_5'],
        mode='lines',
        name='5-Session MA',
        line=dict(color='#2563eb', width=1.2, dash='dash'),
        hovertemplate='%{x|%Y-%m-%d}<br>MA5: %{y:.2f}<extra></extra>'
    ))
    
    # 10-session MA
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['MA_10'],
        mode='lines',
        name='10-Session MA',
        line=dict(color='#dc2626', width=1.2, dash='dot'),
        hovertemplate='%{x|%Y-%m-%d}<br>MA10: %{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Historical Closing Price with Moving Averages",
        xaxis_title="Date",
        yaxis_title=f"Price ({currency_symbol})",
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        height=500,
        template='plotly_white',
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e5e7eb')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e5e7eb')
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption("Chart shows adjusted closing price with 5-session and 10-session simple moving averages.")


def render_actual_vs_predicted_chart(results: Dict, currency_symbol: str):
    """Render actual vs predicted chart for test period."""
    st.markdown('<div class="section-header">🎯 Model Evaluation: Actual vs Predicted (Test Period)</div>', unsafe_allow_html=True)
    
    y_test = results.get('y_test')
    y_pred_test = results.get('y_pred_test')
    features_df = results.get('features_df')
    
    if y_test is None or len(y_test) == 0 or y_pred_test is None or len(y_pred_test) == 0:
        st.warning("No test predictions available")
        return
    
    # Get test dates from features_df (last n rows where n = len(y_test))
    if features_df is not None and not features_df.empty:
        test_dates = features_df['Date'].iloc[-len(y_test):].values
    else:
        test_dates = np.arange(len(y_test))
    
    fig = go.Figure()
    
    # Actual
    fig.add_trace(go.Scatter(
        x=test_dates,
        y=y_test,
        mode='lines+markers',
        name='Actual Close',
        line=dict(color='#1f2937', width=2),
        marker=dict(size=6),
        hovertemplate='%{x|%Y-%m-%d}<br>Actual: %{y:.2f}<extra></extra>'
    ))
    
    # Predicted
    fig.add_trace(go.Scatter(
        x=test_dates,
        y=y_pred_test,
        mode='lines+markers',
        name='Predicted Close',
        line=dict(color='#2563eb', width=2, dash='dash'),
        marker=dict(size=6, symbol='diamond'),
        hovertemplate='%{x|%Y-%m-%d}<br>Predicted: %{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Actual vs Predicted Closing Prices (Held-Out Test Period)",
        xaxis_title="Date",
        yaxis_title=f"Price ({currency_symbol})",
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        height=500,
        template='plotly_white',
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e5e7eb')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e5e7eb')
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption("Model predictions on chronologically held-out test data (approximately 20% most recent observations).")


def render_evaluation_metrics(results: Dict, currency_symbol: str):
    """Render detailed evaluation metrics."""
    st.markdown('<div class="section-header">📊 Evaluation Metrics</div>', unsafe_allow_html=True)
    
    features_df = results.get('features_df')
    y_test = results.get('y_test')
    y_pred_test = results.get('y_pred_test')
    
    if y_test is None or len(y_test) == 0:
        st.warning("No test data available for evaluation")
        return
    
    # Get test features for baseline
    if features_df is not None and not features_df.empty:
        test_features = features_df.iloc[-len(y_test):]
    else:
        test_features = None
    
    # Evaluate
    evaluation = evaluate_model(y_test, y_pred_test, test_features)
    formatted = format_metrics(evaluation, currency_symbol)
    
    # Display metrics in a table
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Model Performance**")
        st.markdown(f"- **MAE**: {formatted['mae']}")
        st.markdown(f"- **RMSE**: {formatted['rmse']}")
        st.markdown(f"- **R²**: {formatted['r2']}")
    
    with col2:
        st.markdown("**Baseline Comparison**")
        st.markdown(f"- **Baseline MAE (Prev Close)**: {formatted['baseline_mae']}")
        st.markdown(f"- **Improvement**: {formatted['improvement_pct']}")
        st.markdown(f"- **Model beats baseline**: {formatted['model_better']}")
    
    # Explanations
    with st.expander("📖 Metric Explanations"):
        st.markdown(f"""
        - **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual prices. 
          Measured in {currency_symbol}. Lower is better.
        - **RMSE (Root Mean Squared Error)**: Square root of average squared errors. 
          Penalizes larger errors more than MAE. Measured in {currency_symbol}. Lower is better.
        - **R² (Coefficient of Determination)**: Proportion of variance explained by the model. 
          Can be negative (worse than horizontal line). 1.0 = perfect fit.
        - **Baseline MAE**: Error from simply predicting "tomorrow = today" (previous close). 
          A reasonable model should beat this baseline.
        - **Improvement**: Percentage reduction in MAE vs baseline. Positive = model is better.
        """)


def render_next_session_estimate(results: Dict, currency_symbol: str):
    """Render next trading session estimate."""
    st.markdown('<div class="section-header">🔮 Next Trading Session Estimate</div>', unsafe_allow_html=True)
    
    next_prediction = results.get('next_prediction')
    latest_close = results.get('latest_close')
    latest_date = results.get('latest_date')
    
    if next_prediction is None or latest_close is None:
        st.warning("Unable to generate next-session estimate")
        return
    
    abs_diff = next_prediction - latest_close
    pct_diff = (abs_diff / latest_close * 100) if latest_close != 0 else 0
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="padding: 1.5rem;">
            <div class="metric-label">Latest Observed Close ({latest_date.strftime('%Y-%m-%d')})</div>
            <div class="metric-value" style="font-size: 2rem;">{currency_symbol}{latest_close:,.2f}</div>
            <br>
            <div class="metric-label">Estimated Next Session Close</div>
            <div class="metric-value" style="font-size: 2rem;">{currency_symbol}{next_prediction:,.2f}</div>
            <div class="{'metric-delta-positive' if abs_diff >= 0 else 'metric-delta-negative'}" style="font-size: 1.2rem;">
                {"+" if abs_diff >= 0 else ""}{abs_diff:,.2f} ({"+" if pct_diff >= 0 else ""}{pct_diff:.2f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**Model Details**")
        st.markdown(f"- **Model**: Linear Regression")
        st.markdown(f"- **Features**: {len(FEATURE_COLUMNS)} price-based features")
        st.markdown(f"- **Training data**: All available history")
        st.markdown(f"- **Data through**: {latest_date.strftime('%Y-%m-%d')}")
    
    # Important disclaimers
    st.markdown("""
    <div class="warning-box">
    <strong>⚠️ Important Disclaimers:</strong>
    <ul>
    <li>This is a <strong>model estimate</strong>, not a guaranteed outcome or financial advice.</li>
    <li>Prediction is based solely on historical price patterns (lag features, moving averages, returns).</li>
    <li>No fundamental data, news, or market sentiment is incorporated.</li>
    <li>Next trading session date may vary due to market holidays and weekends.</li>
    <li>Past-price regression does not establish profitable trading performance.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)


def render_downloads(results: Dict, raw_data: pd.DataFrame, ticker: str, currency_symbol: str):
    """Render CSV download buttons."""
    st.markdown('<div class="section-header">📥 Download Results</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    # 1. Historical data
    with col1:
        if not raw_data.empty:
            csv_hist = raw_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📊 Historical Data",
                data=csv_hist,
                file_name=f"{ticker}_historical_data.csv",
                mime="text/csv",
                use_container_width=True,
            )
    
    # 2. Test results
    with col2:
        y_test = results.get('y_test')
        y_pred_test = results.get('y_pred_test')
        features_df = results.get('features_df')
        
        if y_test is not None and len(y_test) > 0 and features_df is not None and not features_df.empty:
            test_dates = features_df['Date'].iloc[-len(y_test):]
            test_df = create_test_results_df(test_dates, y_test, y_pred_test)
            if not test_df.empty:
                csv_test = test_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="🎯 Test Results (Actual vs Predicted)",
                    data=csv_test,
                    file_name=f"{ticker}_test_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
    
    # 3. Prediction summary
    with col3:
        next_prediction = results.get('next_prediction')
        latest_close = results.get('latest_close')
        latest_date = results.get('latest_date')
        
        if next_prediction is not None and latest_close is not None:
            # Recompute metrics for summary
            y_test = results.get('y_test')
            y_pred_test = results.get('y_pred_test')
            if y_test is not None and len(y_test) > 0:
                test_features = features_df.iloc[-len(y_test):] if features_df is not None else None
                metrics = evaluate_model(y_test, y_pred_test, test_features)
            else:
                metrics = {'model': {}, 'baseline_mae': np.nan, 'comparison': {}}
            
            summary_df = create_prediction_summary(
                latest_close, next_prediction, metrics, latest_date,
                model_name='Linear Regression', currency_symbol=currency_symbol
            )
            csv_summary = summary_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="🔮 Prediction Summary",
                data=csv_summary,
                file_name=f"{ticker}_prediction_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )


def render_feature_info():
    """Render feature engineering information."""
    with st.expander("🔧 Feature Engineering Details"):
        st.mark("""
        **Features used (all available at prediction time):**
        - **lag_1**: Previous session's adjusted close
        - **lag_2**: 2 sessions ago adjusted close
        - **lag_3**: 3 sessions ago adjusted close
        - **lag_5**: 5 sessions ago adjusted close
        - **ma_5**: 5-session moving average (shifted by 1)
        - **ma_10**: 10-session moving average (shifted by 1)
        - **daily_return**: Previous session's return (shifted by 1)
        
        **Target:** Next session's adjusted close
        
        **Methodology:**
        - All features computed without future information (shifted appropriately)
        - Rows with insufficient history dropped
        - Chronological train/test split (80/20) - no random shuffling
        - Final deployment model trained on ALL available labeled data
        """)


def main():
    """Main application entry point."""
    init_session_state()
    render_header()
    
    # Sidebar
    ticker, period, run_analysis = render_sidebar()
    
    # Main content area
    if run_analysis:
        # Validate ticker
        is_valid, error_msg = validate_ticker(ticker)
        if not is_valid:
            render_error(error_msg)
            return
        
        # Fetch data with loading state
        with st.spinner(f"Fetching data for {ticker}..."):
            raw_data = fetch_historical_data(ticker, period)
        
        if raw_data.empty:
            render_error(f"No data found for {ticker}. Please check the ticker symbol and try a different period.")
            return
        
        # Validate sufficient data
        is_sufficient, error_msg = validate_sufficient_data(raw_data)
        if not is_sufficient:
            render_error(error_msg)
            return
        
        # Train and evaluate
        with st.spinner("Training model and evaluating..."):
            results = train_and_evaluate(raw_data)
        
        if results.get('error'):
            render_error(f"Model training failed: {results['error']}")
            return
        
        # Store in session state
        st.session_state.data = raw_data
        st.session_state.results = results
        st.session_state.last_ticker = ticker
        st.session_state.last_period = period
        
        st.success(f"✅ Analysis complete for {ticker} ({period})")
    
    # Display results if available
    if st.session_state.results is not None and st.session_state.data is not None:
        results = st.session_state.results
        raw_data = st.session_state.data
        currency_symbol = get_currency_symbol(st.session_state.last_ticker)
        
        # Show data info
        latest_date = get_latest_market_date(raw_data)
        st.info(f"📅 Data retrieved: {len(raw_data)} trading sessions | Latest market date: {latest_date.strftime('%Y-%m-%d') if latest_date else 'N/A'} | Price type: Adjusted Close")
        
        # Metric cards
        if results.get('next_prediction') is not None and results.get('latest_close') is not None:
            # Recompute metrics for display
            y_test = results.get('y_test')
            y_pred_test = results.get('y_pred_test')
            features_df = results.get('features_df')
            if y_test is not None and len(y_test) > 0:
                test_features = features_df.iloc[-len(y_test):] if features_df is not None else None
                metrics = evaluate_model(y_test, y_pred_test, test_features)
                formatted = format_metrics(metrics, currency_symbol)
            else:
                formatted = {'mae': 'N/A', 'rmse': 'N/A'}
            
            render_metric_cards(
                results['latest_close'],
                results['next_prediction'],
                formatted,
                currency_symbol
            )
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Historical Trends", "🎯 Model Evaluation", "🔮 Next Session Estimate", "🔧 Features"])
        
        with tab1:
            render_historical_chart(raw_data, currency_symbol)
        
        with tab2:
            render_actual_vs_predicted_chart(results, currency_symbol)
            render_evaluation_metrics(results, currency_symbol)
        
        with tab3:
            render_next_session_estimate(results, currency_symbol)
        
        with tab4:
            render_feature_info()
        
        # Downloads
        render_downloads(results, raw_data, st.session_state.last_ticker, currency_symbol)
    
    else:
        # Initial state
        render_info("Enter a stock ticker in the sidebar and click **'Fetch Data & Run Analysis'** to begin.")
        
        # Show example of what the app does
        st.markdown("### What this app does")
        st.markdown("""
        1. **Downloads** historical daily price data from Yahoo Finance
        2. **Engineers** features from past prices (lags, moving averages, returns) without future leakage
        3. **Trains** a Linear Regression model on ~80% of data (chronological split)
        4. **Evaluates** on held-out ~20% test set with MAE, RMSE, R² vs baseline
        5. **Estimates** the next trading session's closing price
        6. **Provides** interactive charts and downloadable CSV results
        """)
    
    # Footer disclaimer
    st.markdown("""
    <div class="disclaimer">
    <strong>Disclaimer:</strong> This application is for educational and demonstration purposes only. 
    It does not constitute financial advice, investment recommendations, or trading signals. 
    Past performance does not guarantee future results. The model uses only historical price data 
    and ignores fundamentals, news, market conditions, and risk factors. 
    Data sourced from Yahoo Finance via yfinance; accuracy and availability not guaranteed.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()