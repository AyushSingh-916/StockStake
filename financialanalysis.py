!pip install plotly yfinance scikit-learn pandas numpy

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import datetime
import plotly.graph_objs as go
from plotly.offline import iplot

# -----------------------------
# 1. Data Collection
# -----------------------------
def load_stock_data(symbol, years=2):
    """Download historical stock data. Defaults to 2 years as per resume."""
    try:
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=years*365)
        
        df = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)
        
        # FIX: Flatten MultiIndex columns introduced in recent yfinance updates
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        stock_info = yf.Ticker(symbol)
        return df, stock_info
    except Exception as err:
        print(f"Could not retrieve data for {symbol}: {err}")
        return None, None

# -----------------------------
# 2. Fundamental Ratios (Fixed Extraction)
# -----------------------------
def safe_extract(df, keys):
    if df is None or df.empty: return np.nan
    for key in keys:
        if key in df.index: 
            # FIX: Added to iloc to grab the most recent year's value
            return df.loc[key].iloc 
    return np.nan

def extract_financial_ratios(stock_info):
    """Compute key financial ratios for holistic insights."""
    metrics = {}
    try:
        bs = stock_info.balance_sheet
        fs = stock_info.financials

        debt = safe_extract(bs, ['Total Debt', 'Total Liab', 'Total Liabilities Net Minority Interest'])
        equity = safe_extract(bs, ['Stockholders Equity', 'Total Stockholder Equity'])
        metrics['de_ratio'] = debt / equity if equity and equity != 0 else np.nan

        net_income = safe_extract(fs, ['Net Income Common Stockholders', 'Net Income'])
        metrics['roe'] = net_income / equity if equity and equity != 0 else np.nan
        
        current_assets = safe_extract(bs, ['Current Assets', 'Total Current Assets'])
        current_liabilities = safe_extract(bs, ['Current Liabilities', 'Total Current Liabilities'])
        metrics['current_ratio'] = current_assets / current_liabilities if current_liabilities and current_liabilities != 0 else np.nan

    except Exception as err:
        pass # Silently pass missing data to keep output clean
    return metrics

# -----------------------------
# 3. Feature Engineering (The Secret Sauce)
# -----------------------------
def create_lagged_features(df, lags=5):
    """Convert prices to daily returns and create sliding window features."""
    data = pd.DataFrame(index=df.index)
    data['Close'] = df['Close'].values.flatten()
    
    # We predict Daily Returns (percentage change), not raw prices
    data['Return'] = data['Close'].pct_change()
    
    # Create Lagged Returns (sliding window)
    for i in range(1, lags + 1):
        data[f'Lag_{i}'] = data['Return'].shift(i)
        
    data.dropna(inplace=True)
    return data

# -----------------------------
# 4. Model Training (Random Forest)
# -----------------------------
def train_rf_model(data, lags=5):
    """Train Random Forest on lagged returns."""
    features = [f'Lag_{i}' for i in range(1, lags + 1)]
    X = data[features]
    y = data['Return']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # Calculate RMSE on the Test Set (Returns)
    predictions = model.predict(X_test)
    test_rmse = sqrt(mean_squared_error(y_test, predictions))
    
    # Predict over the entire dataset for visualization
    data['Predicted_Return'] = model.predict(X)
    
    # Reconstruct prices from predicted returns
    # Price_today = Price_yesterday * (1 + Predicted_Return)
    data['Predicted_Close'] = data['Close'].shift(1) * (1 + data['Predicted_Return'])
    
    return model, data, test_rmse, features

# -----------------------------
# 5. Future Forecasting (Recursive)
# -----------------------------
def project_future_prices(model, data, features, days_ahead=30):
    """Recursively forecast future returns and convert to prices."""
    # Get the last known lagged returns as our starting window
    current_window = data[features].iloc[-1].values.tolist()
    last_known_price = data['Close'].iloc[-1]
    
    forecast_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=days_ahead, freq='B')
    future_prices = []
    
    current_price = last_known_price
    
    for _ in range(days_ahead):
        # Predict the return for the next day
        pred_return = model.predict(np.array([current_window]))
        
        # Calculate new price
        next_price = current_price * (1 + pred_return)
        future_prices.append(next_price)
        
        # Update the sliding window for the next loop iteration
        current_window.insert(0, pred_return)
        current_window.pop() # Drop the oldest lag
        
        current_price = next_price
        
    return forecast_dates, future_prices

# -----------------------------
# 6. Plotly Visualizations
# -----------------------------
def plot_full_analysis(data, forecast_dates, future_prices, symbol):
    """Combines historical, predicted, and future forecasts into one interactive chart."""
    actual_trace = go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Actual Price', line=dict(color='blue'))
    
    # Start predicted trace from the second day to align with shifted prices
    pred_trace = go.Scatter(x=data.index[1:], y=data['Predicted_Close'][1:], mode='lines', name='RF Fitted', line=dict(color='orange', width=1))
    
    future_trace = go.Scatter(x=forecast_dates, y=future_prices, mode='lines', name='30-Day Forecast', line=dict(color='red', dash='dash'))
    
    layout = go.Layout(
        title=f'{symbol} - AI Price Prediction & Forecast (Random Forest)',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Stock Price'),
        hovermode='x unified'
    )
    fig = go.Figure(data=[actual_trace, pred_trace, future_trace], layout=layout)
    iplot(fig)

# -----------------------------
# 7. Recommendation Engine
# -----------------------------
def make_investment_decision(ratios, data, future_prices, rmse):
    """Blends ML trajectory with Fundamental Ratios to reduce risk."""
    clean_ratios = [val for val in ratios.values() if not np.isnan(val)]
    fundamental_score = np.mean([np.tanh(val) for val in clean_ratios]) if clean_ratios else 0
    
    # ML Score based on projected 30-day trajectory
    current_price = data['Close'].iloc[-1]
    projected_price = future_prices[-1]
    expected_return = (projected_price - current_price) / current_price
    
    # Combine signals
    confidence = min(100, max(50, 50 + (fundamental_score * 25) + (expected_return * 500)))
    
    decision = "BUY" if expected_return > 0.02 and fundamental_score > -0.2 else "SELL/HOLD"
    
    return f"Recommendation: {decision} | Confidence: {confidence:.1f}% | Proj. 30-Day Return: {expected_return*100:.2f}% | Model RMSE (Returns): {rmse:.4f}"

# -----------------------------
# 8. Main Execution Pipeline
# -----------------------------
def analyze_stock(symbol):
    print(f"\nEvaluating {symbol}...")
    df, stock_info = load_stock_data(symbol)

    if df is not None and not df.empty:
        ratios = extract_financial_ratios(stock_info)
        
        # Machine Learning Pipeline
        data = create_lagged_features(df, lags=5)
        model, data, rmse, features = train_rf_model(data, lags=5)
        
        # Future Projection
        forecast_dates, future_prices = project_future_prices(model, data, features, days_ahead=30)
        
        # Visualization & Output
        plot_full_analysis(data, forecast_dates, future_prices, symbol)
        rec = make_investment_decision(ratios, data, future_prices, rmse)
        return rec
    else:
        return f"No valid data found for {symbol}"

# -----------------------------
# Run the Platform
# -----------------------------
stock_list = [
    'ITC.NS',   # ITC Ltd
    'TCS.NS'    # Tata Consultancy Services
]

for symbol in stock_list:
    print(analyze_stock(symbol))
