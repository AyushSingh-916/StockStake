# !pip install plotly

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
# Data Collection
# -----------------------------
def load_stock_data(symbol):
    """Download historical stock data for a given symbol."""
    try:
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        df = yf.download(symbol, start="2010-01-01", end=today)
        stock_info = yf.Ticker(symbol)
        return df, stock_info
    except Exception as err:
        print(f"Could not retrieve data for {symbol}: {err}")
        return None, None


# -----------------------------
# Financial Ratios Calculation
# -----------------------------
def extract_financial_ratios(stock_info):
    """Compute key financial ratios using Yahoo Finance data."""
    metrics = {}
    try:
        info = stock_info.info
        bs = stock_info.balance_sheet
        cf = stock_info.cashflow
        fs = stock_info.financials

        # Valuation
        metrics['pe_ratio'] = info.get('trailingPE', np.nan)

        # Leverage
        debt = bs.get('Total Liab', pd.Series([np.nan])).iloc[-1]
        equity = bs.get('Total Stockholder Equity', pd.Series([np.nan])).iloc[-1]
        metrics['de_ratio'] = debt / equity if equity else np.nan

        # Profitability
        net_income = fs.get('Net Income', pd.Series([np.nan])).iloc[-1]
        metrics['roe'] = net_income / equity if equity else np.nan

        # Liquidity
        current_assets = bs.get('Total Current Assets', pd.Series([np.nan])).iloc[-1]
        current_liabilities = bs.get('Total Current Liabilities', pd.Series([np.nan])).iloc[-1]
        metrics['current_ratio'] = current_assets / current_liabilities if current_liabilities else np.nan

        # Dividend
        metrics['dividend_yield'] = info.get('dividendYield', np.nan)

        # Interest coverage
        operating_cash = cf.get('Total Cash From Operating Activities', pd.Series([np.nan])).iloc[-1]
        interest = cf.get('Interest Expense', pd.Series([0])).iloc[-1]
        metrics['interest_coverage'] = operating_cash / -interest if interest else np.nan

        # Efficiency
        revenue = fs.get('Total Revenue', pd.Series([np.nan])).iloc[-1]
        avg_assets = (bs.get('Total Assets', pd.Series([np.nan])).iloc[-1] + bs.get('Total Assets', pd.Series([np.nan])).iloc[0]) / 2
        metrics['asset_turnover'] = revenue / avg_assets if avg_assets else np.nan

    except Exception as err:
        print(f"Error computing ratios: {err}")

    return metrics


# -----------------------------
# Model Training & Evaluation
# -----------------------------
def train_price_model(df):
    """Train a Random Forest model to predict stock prices."""
    if df is None or df.empty:
        return None, None, None, float('nan')

    df['Index_Day'] = np.arange(len(df))
    features = df[['Index_Day']]
    target = df['Close']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    test_rmse = sqrt(mean_squared_error(y_test, rf_model.predict(X_test)))
    return rf_model, features, target, test_rmse


# -----------------------------
# Visualization
# -----------------------------
def plot_historical_vs_predicted(df, model, features, target):
    """Visualize actual vs predicted stock prices."""
    actual_trace = go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Actual')
    predicted_trace = go.Scatter(x=features.index, y=model.predict(features), mode='markers', name='Predicted', marker=dict(color='red'))
    
    layout = go.Layout(title='Stock Price Estimation', xaxis=dict(title='Date'), yaxis=dict(title='Price'))
    fig = go.Figure(data=[actual_trace, predicted_trace], layout=layout)
    iplot(fig)


def project_future_prices(model, df, days_ahead=30):
    """Forecast future prices based on trained model."""
    if model is None or df.empty:
        print("Forecasting not possible due to missing model/data.")
        return None, None

    last_index = df['Index_Day'].iloc[-1]
    upcoming_days = np.arange(last_index + 1, last_index + days_ahead + 1)
    forecast_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=days_ahead)
    forecast_input = pd.DataFrame(upcoming_days, columns=['Index_Day'], index=forecast_dates)

    future_predictions = model.predict(forecast_input)
    return forecast_input.index, future_predictions


def plot_future_forecast(df, forecast_dates, forecast_prices):
    """Plot historical prices with forecasted future values."""
    hist_trace = go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Historical')
    future_trace = go.Scatter(x=forecast_dates, y=forecast_prices, mode='lines', name='Forecast', line=dict(color='red', dash='dash'))
    
    layout = go.Layout(title='Future Stock Price Forecast', xaxis=dict(title='Date'), yaxis=dict(title='Price'))
    fig = go.Figure(data=[hist_trace, future_trace], layout=layout)
    iplot(fig)


# -----------------------------
# Recommendation Logic
# -----------------------------
def make_investment_decision(ratios, model, rmse):
    """Provide buy/sell recommendation based on model & ratios."""
    if model is None:
        return "Recommendation unavailable: insufficient data."

    ratio_score = np.nanmean([np.tanh(val) for val in ratios.values()])
    confidence_level = 100 * np.tanh(model.feature_importances_[0]) * ratio_score if not np.isnan(ratio_score) else 50

    decision = "Buy" if confidence_level > 50 else "Sell"
    return f"Suggested Action: {decision} with {confidence_level:.2f}% confidence. Model RMSE: {rmse:.2f}"


# -----------------------------
# Main Pipeline
# -----------------------------
def analyze_stock(symbol):
    df, stock_info = load_stock_data(symbol)

    if df is not None and not df.empty:
        ratios = extract_financial_ratios(stock_info)
        model, features, target, rmse = train_price_model(df)

        plot_historical_vs_predicted(df, model, features, target)
        recommendation = make_investment_decision(ratios, model, rmse)

        forecast_dates, forecast_prices = project_future_prices(model, df, days_ahead=60)
        if forecast_dates is not None:
            plot_future_forecast(df, forecast_dates, forecast_prices)

        return recommendation
    else:
        return f"No valid data found for {symbol}"


# -----------------------------
# Execution Example
# -----------------------------
stock_list = [
    'ITC.NS',      # ITC Ltd
    'TCS.NS',      # Tata Consultancy Services
    'HDFCBANK.NS', # HDFC Bank
    'INFY.NS',     # Infosys
    'RELIANCE.NS', # Reliance Industries
    'ICICIBANK.NS',# ICICI Bank
    'HINDUNILVR.NS',# Hindustan Unilever
    'KOTAKBANK.NS',# Kotak Mahindra Bank
    'LT.NS',       # Larsen & Toubro
    'SBIN.NS'      # State Bank of India
]

for symbol in stock_list:
    print(analyze_stock(symbol))