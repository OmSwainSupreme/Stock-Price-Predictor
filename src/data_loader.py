"""Data loading module for stock price prediction.

Handles downloading, normalizing, validating, and caching historical market data.
"""

import logging
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd
import streamlit as st
import yfinance as yf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Period mapping for user-friendly selection
PERIOD_MAP = {
    "1 month": "1mo",
    "3 months": "3mo",
    "6 months": "6mo",
    "1 year": "1y",
    "2 years": "2y",
    "5 years": "5y",
}

# Valid periods for yfinance
VALID_PERIODS = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]


def validate_ticker(ticker: str) -> Tuple[bool, str]:
    """Validate ticker input.
    
    Args:
        ticker: Stock ticker symbol.
        
    Returns:
        Tuple of (is_valid, error_message).
    """
    if not ticker or not ticker.strip():
        return False, "Ticker symbol cannot be empty."
    
    ticker = ticker.strip().upper()
    
    # Basic format validation
    if len(ticker) > 20:
        return False, "Ticker symbol is too long."
    
    # Allow alphanumeric, dots, hyphens, carets (for indices)
    import re
    if not re.match(r'^[A-Z0-9.\-^]+$', ticker):
        return False, "Ticker contains invalid characters."
    
    return True, ""


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_historical_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Download historical daily market data using yfinance.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'RELIANCE.NS').
        period: Data period (e.g., '1y', '6mo', '5y').
        
    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Adj Close, Volume.
        Returns empty DataFrame on failure.
    """
    ticker = ticker.strip().upper()
    
    if period not in VALID_PERIODS:
        logger.warning(f"Invalid period '{period}', defaulting to '1y'")
        period = "1y"
    
    try:
        logger.info(f"Fetching data for {ticker} with period {period}")
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, auto_adjust=False)
        
        if df.empty:
            logger.warning(f"No data returned for {ticker} with period {period}")
            return pd.DataFrame()
        
        # Normalize the data
        df = normalize_data(df, ticker)
        
        logger.info(f"Successfully fetched {len(df)} rows for {ticker}")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()


def normalize_data(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize downloaded data into a consistent DataFrame.
    
    Args:
        df: Raw DataFrame from yfinance.
        ticker: Stock ticker symbol for logging.
        
    Returns:
        Normalized DataFrame with consistent columns and index.
    """
    # Reset index to get Date as column
    df = df.reset_index()
    
    # Ensure Date column exists and is datetime
    if 'Date' not in df.columns:
        if 'Datetime' in df.columns:
            df = df.rename(columns={'Datetime': 'Date'})
        else:
            logger.error(f"No date column found for {ticker}")
            return pd.DataFrame()
    
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort chronologically and remove duplicates
    df = df.sort_values('Date').drop_duplicates(subset='Date', keep='last')
    
    # Ensure required columns exist
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            logger.error(f"Missing required column '{col}' for {ticker}")
            return pd.DataFrame()
    
    # Handle Adj Close - use it if available, otherwise use Close
    if 'Adj Close' not in df.columns:
        df['Adj Close'] = df['Close']
        logger.info(f"Adj Close not available for {ticker}, using Close")
    
    # Select and order columns
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    
    # Remove rows with missing Close or Adj Close
    df = df.dropna(subset=['Close', 'Adj Close'])
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df


def validate_sufficient_data(df: pd.DataFrame, min_rows: int = 30) -> Tuple[bool, str]:
    """Validate that sufficient usable historical data exists.
    
    Args:
        df: Historical data DataFrame.
        min_rows: Minimum number of rows required.
        
    Returns:
        Tuple of (is_sufficient, error_message).
    """
    if df.empty:
        return False, "No historical data available for this ticker and period."
    
    if len(df) < min_rows:
        return False, f"Insufficient data: {len(df)} rows available, minimum {min_rows} required."
    
    # Check for non-finite values in key columns
    key_cols = ['Close', 'Adj Close']
    for col in key_cols:
        if df[col].isna().any() or not df[col].apply(lambda x: pd.notna(x) and np.isfinite(x)).all():
            return False, f"Data contains missing or non-finite values in '{col}'."
    
    return True, ""


def get_latest_market_date(df: pd.DataFrame) -> Optional[pd.Timestamp]:
    """Get the most recent available market date from the data.
    
    Args:
        df: Historical data DataFrame.
        
    Returns:
        Latest date as Timestamp, or None if DataFrame is empty.
    """
    if df.empty or 'Date' not in df.columns:
        return None
    return df['Date'].max()


def get_currency_symbol(ticker: str) -> str:
    """Get currency symbol for display based on ticker.
    
    Args:
        ticker: Stock ticker symbol.
        
    Returns:
        Currency symbol string.
    """
    ticker = ticker.upper()
    if ticker.endswith('.NS') or ticker.endswith('.BO'):
        return '₹'  # Indian Rupee
    elif ticker.endswith('.L'):
        return '£'  # British Pound
    elif ticker.endswith('.TO'):
        return 'C$'  # Canadian Dollar
    elif ticker.endswith('.AX'):
        return 'A$'  # Australian Dollar
    elif ticker.endswith('.HK'):
        return 'HK$'  # Hong Kong Dollar
    elif ticker.endswith('.T'):
        return '¥'  # Japanese Yen
    else:
        return '$'  # Default to USD


# Import numpy for validation
import numpy as np