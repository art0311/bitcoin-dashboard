import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import requests
import plotly.graph_objects as go

st.set_page_config(page_title="Bitcoin Live Dashboard", layout="wide")

# -----------------------------
# Helper function to check data
# -----------------------------
def has_data(df, cols):
    return df is not None and not df.empty and all(col in df.columns for col in cols)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("⚙️ Settings")
symbol = st.sidebar.selectbox("Asset", ["BTC-USD"], index=0)

period = st.sidebar.selectbox(
    "Time Range",
    ["1d", "7d", "1mo", "1y", "2y", "5y", "max"],
    index=3
)

predict_days = st.sidebar.slider("Prediction Days", 7, 90, 30)

# -----------------------------
# Crypto Fear & Greed Gauge
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Market Sentiment")

@st.cache_data(ttl=300)
def fetch_fng():
    try:
        url = "https://api.alternative.me/fng/"
        r = requests.get(url, timeout=5)
        data = r.json()['data'][0]
        value = int(data['value'])
        label = data['value_classification']
        return value, label
    except:
        return None, "N/A"

sentiment_score, sentiment_label = fetch_fng()

if sentiment_score is not None:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sentiment_score,
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 24], 'color': 'red'},
                {'range': [25, 49], 'color': 'orange'},
                {'range': [50, 50], 'color': 'yellow'},
                {'range': [51, 74], 'color': 'lightgreen'},
                {'range': [75, 100], 'color': 'green'}
            ],
        },
        title={'text': sentiment_label}
    ))
    st.sidebar.plotly_chart(fig_gauge, use_container_width=True)
else:
    st.sidebar.info("Could not fetch sentiment data.")

# -----------------------------
# Data Loader
# -----------------------------
@st.cache_data(ttl=120)
def load_data(symbol, period):
    try:
        if period in ["1d", "7d"]:
            df = yf.download(symbol, period=period, interval="5m", progress=False)
        else:
            df = yf.download(symbol, period=period, progress=False)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Ensure OHLCV exists
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col not in df.columns:
                df[col] = np.nan

        df.dropna(subset=['Close', 'High', 'Low', 'Volume'], inplace=True)
        return df
    except Exception:
        return pd.DataFrame()

btc = load_data(symbol, period)

# -----------------------------
# Indicators (SMA, RSI, VWAP, OBV, MACD, Bollinger Bands)
# -----------------------------
if has_data(btc, ['Close', 'High', 'Low', 'Volume']):
    close = btc['Close'].to_numpy(dtype=float)
    high = btc['High'].to_numpy(dtype=float)
    low = btc['Low'].to_numpy(dtype=float)
    volume = btc['Volume'].to_numpy(dtype=float)

    # SMA
    btc['SMA_50'] = pd.Series(close).rolling(50).mean() if len(close) >= 50 else np.nan
    btc['SMA_200'] = pd.Series(close).rolling(200).mean() if len(close) >= 200 else np.nan

    # RSI
    if len(close) >= 15:
        delta = pd.Series(close).diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        rs = gain.rolling(14).mean() / loss.rolling(14).mean()
        btc['RSI'] = 100 - (100 / (1 + rs))
    else:
        btc['RSI'] = np.nan

    # VWAP
    typical_price = (high + low + close) / 3
    btc['VWAP'] = (typical_price * volume).cumsum() / volume.cumsum()

    # OBV
    obv = np.zeros(len(close))
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - volume[i]
        else:
            obv[i] = obv[i-1]
    btc['OBV'] = obv

    # MACD
    if len(close) >= 26:
        exp12 = pd.Series(close).ewm(span=12, adjust=False).mean()
        exp26 = pd.Series(close).ewm(span=26, adjust=False).mean()
        btc['MACD'] = exp12 - exp26
        btc['MACD_Signal'] = btc['MACD'].ewm(span=9, adjust=False).mean()
        btc['MACD_Hist'] = btc['MACD'] - btc['MACD_Signal']
    else:
        btc['MACD'] = btc['MACD_Signal'] = btc['MACD_Hist'] = np.nan

    # Bollinger Bands
    if len(close) >= 20:
        sma20 = pd.Series(close).rolling(20).mean()
        std20 = pd.Series(close).rolling(20).std()
        btc['BB_upper'] = sma20 + 2 * std20
        btc['BB_lower'] = sma20 - 2 * std20
    else:
        btc['BB_upper'] = btc['BB_lower'] = np.nan
else:
    # Ensure all columns exist to avoid errors
    for col in ['SMA_50','SMA_200','RSI','VWAP','OBV','MACD','MACD_Signal','MACD_Hist','BB_upper','BB_lower']:
        btc[col] = np.nan

# -----------------------------
# Support & Resistance
# -----------------------------
def detect_levels(series, window=20, tolerance=0.015):
    prices = series.to_numpy(dtype=float)
    levels = []
    for i in range(window, len(prices) - window):
        low_range = prices[i-window:i+window]
        current = prices[i]
        if current == low_range.min():
            levels.append(current)
        elif current == low_range.max():
            levels.append(current)
    levels = sorted(levels)
    clustered = []
    for level in levels:
        if not clustered or abs(level - clustered[-1]) / clustered[-1] > tolerance:
            clustered.append(level)
    return clustered

levels = detect_levels(btc['Close']) if has_data(btc, ['Close']) else []

# -----------------------------
# Dashboard Layout
# -----------------------------
st.title("📊 Bitcoin Live Market Dashboard")
st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

# Metrics








