# AGENTS.md - Agent Instructions for Stock Price Predictor

## Project Overview
This is a complete Stock Price Predictor application built with Streamlit, designed for AI/ML internship demonstration. The application retrieves historical stock data, engineers features, trains a Linear Regression model with chronological splitting, evaluates on holdout data, and predicts the next trading session's close.

## Key Architectural Decisions

### 1. No Temporal Leakage
- All features computed with explicit shifts (`.shift(1)` for lags, rolling windows shifted)
- Target is `t+1` close, features from `t, t-1, ...`
- Chronological train/test split (80/20) - NO random shuffling
- Final deployment model trained on ALL data AFTER evaluation

### 2. Data Flow
```
yfinance -> normalize_data() -> create_features() -> chronological_split() 
-> fit(train) -> predict(test) -> evaluate -> fit(all) -> predict_next()
```

### 3. Price Series
- Uses **Adjusted Close** (accounts for splits/dividends) when available
- Falls back to **Close** if Adj Close unavailable
- Documents which series is used throughout

### 4. Error Handling
- Validates ticker format, data sufficiency, non-finite values
- Graceful degradation with user-friendly messages
- Never fabricates data or silently uses stale cache

## Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `data_loader.py` | yfinance download, normalization, validation, caching |
| `features.py` | Feature engineering, target alignment, latest feature reconstruction |
| `model.py` | Chronological split, LinearRegression training, prediction |
| `metrics.py` | MAE/RMSE/R², baseline MAE, formatting, CSV creation |
| `app.py` | Streamlit UI, orchestration, charts, downloads |

## Testing Strategy

- **Unit tests only** - no internet, no live data
- Synthetic deterministic data (seeded random walk)
- Test categories:
  - Feature column generation & alignment
  - No future leakage in rolling features
  - Chronological split preserves order
  - Train/test separation enforced
  - Metrics calculations (including edge cases)
  - Next-session feature reconstruction matches training
  - Error handling for invalid/empty inputs

## Running Commands (Verified)

```powershell
# Windows - Setup
cd C:\codecproject1
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run application
streamlit run app.py
```

## Common Issues & Fixes

| Issue | Resolution |
|-------|------------|
| yfinance fails | Check internet, try different ticker/period |
| Insufficient data | Need ~30+ sessions; use longer period |
| Module import errors | Ensure venv activated, run from project root |
| Streamlit not found | `pip install streamlit` in venv |
| Port 8501 in use | `streamlit run app.py --server.port 8502` |

## Extension Points

1. **Add models**: Extend `model.py` with new estimator classes (keep same interface)
2. **Add features**: Modify `features.py` `FEATURE_COLUMNS` and `create_features()`
3. **Add metrics**: Extend `metrics.py` `calculate_metrics()`
4. **Change data source**: Replace `data_loader.py` fetch logic
5. **Add charts**: Add Plotly figures in `app.py` tabs

## Do Not

- ❌ Add random train/test split
- ❌ Use future data in features (no `shift(-1)` except for target)
- ❌ Train on test data
- ❌ Fabricate confidence intervals
- ❌ Claim financial accuracy
- ❌ Add deep learning without clear need
- ❌ Commit virtual environments or cache

## Git Workflow

```bash
git init
git add .
git commit -m "feat: implement stock price prediction dashboard"
# Push to remote if needed
```