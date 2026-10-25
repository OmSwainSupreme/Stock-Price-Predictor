# Stock Price Predictor

An interactive Streamlit web application that retrieves historical stock market data, trains a Linear Regression model on engineered price features, evaluates performance on a chronological holdout set, and estimates the next trading session's closing price.

## Features

- **Stock Selection**: Enter any valid ticker symbol (e.g., AAPL, MSFT, GOOGL, RELIANCE.NS, TCS.NS, INFY.NS) with period selection (1 month to 5 years)
- **Data Retrieval**: Downloads historical daily data from Yahoo Finance via yfinance using adjusted closing prices
- **Feature Engineering**: Creates 7 price-based features (lags, moving averages, returns) without future leakage
- **Model Training**: Linear Regression with chronological 80/20 train/test split (no random shuffling)
- **Evaluation**: MAE, RMSE, R² with baseline comparison (previous-close benchmark)
- **Visualizations**: Interactive Plotly charts for historical trends and actual vs predicted prices
- **Next-Session Estimate**: Predicts next trading day's close using model trained on all available data
- **CSV Downloads**: Historical data, test results, and prediction summary
- **Professional UI**: Clean financial dashboard with metric cards, tooltips, and responsive charts

## Technology Stack

- **Python**: 3.11+
- **Streamlit**: Web interface
- **yfinance**: Market data retrieval
- **pandas/numpy**: Data processing
- **scikit-learn**: Linear Regression and metrics
- **plotly**: Interactive charts
- **pytest**: Automated testing

## Project Structure

```
stock-price-predictor/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .gitignore            # Git ignore rules
├── AGENTS.md             # Agent instructions
├── src/
│   ├── __init__.py
│   ├── data_loader.py    # Data fetching, validation, caching
│   ├── features.py       # Feature engineering & target alignment
│   ├── model.py          # Model training, splitting, prediction
│   └── metrics.py        # Evaluation metrics & baseline comparison
└── tests/
    ├── test_features.py  # Feature engineering tests
    ├── test_model.py     # Model training/prediction tests
    └── test_metrics.py   # Metrics calculation tests
```

## Installation

### Prerequisites
- Python 3.11 or newer
- Git (optional, for version control)

### Windows Setup

```powershell
# Navigate to project directory
cd C:\codecproject1

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Linux/macOS Setup

```bash
# Navigate to project directory
cd /path/to/codecproject1

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

## Running the Application

```bash
# From project root with virtual environment activated
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

## Usage

1. **Select a Stock**: Enter a ticker symbol in the sidebar (default: AAPL). Examples: AAPL, MSFT, GOOGL, RELIANCE.NS, TCS.NS, INFY.NS
2. **Choose Period**: Select historical data period (1 month to 5 years)
3. **Run Analysis**: Click "Fetch Data & Run Analysis" to download data, train model, and generate predictions
4. **Explore Results**:
   - **Historical Trends**: Closing price with 5/10-session moving averages
   - **Model Evaluation**: Actual vs predicted on held-out test set with metrics
   - **Next Session Estimate**: Model's prediction for next trading day
   - **Features**: Details on feature engineering methodology
5. **Download Results**: CSV exports for historical data, test results, and prediction summary

## Feature Engineering

All features are computed using only past information (no future leakage):

| Feature | Description |
|---------|-------------|
| `lag_1` | Previous session's adjusted close |
| `lag_2` | 2 sessions ago adjusted close |
| `lag_3` | 3 sessions ago adjusted close |
| `lag_5` | 5 sessions ago adjusted close |
| `ma_5` | 5-session moving average (shifted by 1) |
| `ma_10` | 10-session moving average (shifted by 1) |
| `daily_return` | Previous session's return (shifted by 1) |

**Target**: Next trading session's adjusted close

Rows with insufficient history for lags/rolling windows are dropped.

## Chronological Train/Test Split

The dataset is split **chronologically** (not randomly):
- **Training**: First ~80% of observations (oldest data)
- **Testing**: Last ~20% of observations (most recent data)

This is critical for time-series data because:
1. Random splitting would leak future information into training
2. Chronological split simulates real-world forecasting (train on past, predict future)
3. Prevents overly optimistic performance estimates

## Evaluation Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **MAE** | Mean Absolute Error | Average absolute prediction error in price units. Lower = better. |
| **RMSE** | Root Mean Squared Error | Square root of average squared errors. Penalizes large errors more. Lower = better. |
| **R²** | Coefficient of Determination | Proportion of variance explained. Can be negative (worse than naive). 1.0 = perfect. |
| **Baseline MAE** | Previous-close benchmark | MAE from predicting "tomorrow = today". Model should beat this. |

## Known Limitations

1. **Data Dependency**: Requires internet connection and Yahoo Finance availability. Some tickers may not be supported.
2. **Price-Only Model**: Uses only historical prices; ignores fundamentals, news, sentiment, macro factors.
3. **Linear Assumption**: Linear Regression assumes linear relationships; financial markets are non-linear.
4. **No Uncertainty Quantification**: Point estimates only; no confidence intervals.
5. **Market Holidays**: Next trading session date uncertain without exchange calendar.
6. **Educational Only**: Not financial advice. Past patterns don't guarantee future performance.
7. **Cache Duration**: Data cached for 1 hour; may not reflect latest market data.
8. **Minimum Data**: Requires ~30+ trading sessions for reliable training.

## Data Source Attribution

- Market data provided by **Yahoo Finance** via the `yfinance` library
- Data includes adjusted closing prices (accounting for splits and dividends)
- Availability and accuracy depend on Yahoo Finance coverage

## Disclaimer

> **This application is for educational and demonstration purposes only. It does not constitute financial advice, investment recommendations, or trading signals. Past performance does not guarantee future results. The model uses only historical price data and ignores fundamentals, news, market conditions, and risk factors. Data sourced from Yahoo Finance via yfinance; accuracy and availability not guaranteed.**

## Development

### Running Tests

```bash
# From project root with virtual environment activated
pytest tests/ -v
```

### Code Structure Principles

- Modular design with single-responsibility modules
- No temporal leakage in feature engineering
- Chronological splitting enforced
- Comprehensive error handling
- Unit tests with synthetic data (no internet required)

## License

MIT License - Feel free to use for learning and demonstration purposes.