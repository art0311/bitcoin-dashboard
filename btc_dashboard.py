import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Bitcoin Live Dashboard", layout="wide")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("⚙️ Settings")
symbol = st.sidebar.selectbox("Asset", ["BTC-USD"], index=0)

period = st.sidebar.selectbox(
    "Time Range",
    ["1d", "7d", "1mo", "1y", "2y", "5y", "max"],  # Added 1d, 7d, 1mo
    index=3
)

predict_days = st.sidebar.slider("Prediction Days", 7, 90, 30)

# -----------------------------
# Crypto Fear & Greed Index (Gauge)
# -----------------------------
import requests
import plotly.graph_objects as go

st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Market Sentiment")

def fetch_fng():
    try:
        url = "https://api.alternative.me/fng/"
        r = requests.get(url)
        data = r.json()['data'][0]
        value = int(data['value'])
        label = data['value_classification']
        return value, label
    except:
        return None, "N/A"

sentiment_score, sentiment_label = fetch_fng()

if sentiment_score is not None:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sentiment_score,
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 24], 'color': 'red'},         # Extreme Fear
                {'range': [25, 49], 'color': 'orange'},     # Fear
                {'range': [50, 50], 'color': 'yellow'},     # Neutral
                {'range': [51, 74], 'color': 'lightgreen'}, # Greed
                {'range': [75, 100], 'color': 'green'}      # Extreme Greed
            ],
        },
        title={'text': sentiment_label}
    ))
    st.sidebar.plotly_chart(fig, use_container_width=True)
else:
    st.sidebar.info("Could not fetch sentiment data.")

# -----------------------------
# Data Loader
# -----------------------------
@st.cache_data(ttl=120)
def load_data(symbol, period):
    try:
        df = yf.download(symbol, period=period, progress=False)
        df = df[['Close']]
        df.dropna(inplace=True)
        return df
    except Exception:
        return pd.DataFrame()

btc = load_data(symbol, period)

# -----------------------------
# Indicators (Crash-proof)
# -----------------------------
if not btc.empty and len(btc) > 1:
    if len(btc) >= 50:
        btc['SMA_50'] = btc['Close'].rolling(50).mean()
    if len(btc) >= 200:
        btc['SMA_200'] = btc['Close'].rolling(200).mean()

    if len(btc) >= 15:
        delta = btc['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        rs = gain.rolling(14).mean() / loss.rolling(14).mean()
        btc['RSI'] = 100 - (100 / (1 + rs))

# -----------------------------
# Support & Resistance Detection
# -----------------------------
def detect_levels(series, window=20, tolerance=0.015):
    prices = series.values
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
        if not clustered:
            clustered.append(level)
        elif abs(level - clustered[-1]) / clustered[-1] > tolerance:
            clustered.append(level)

    return clustered

levels = detect_levels(btc['Close']) if not btc.empty else []

# -----------------------------
# Header
# -----------------------------
st.title("📊 Bitcoin Live Market Dashboard")
st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

# -----------------------------
# Metrics
# -----------------------------
col1, col2, col3 = st.columns(3)

if btc.empty or len(btc) < 2:
    col1.metric("BTC Price (USD)", "Loading...", "")
    col2.metric("50D SMA", "Loading...")
    col3.metric("200D SMA", "Loading...")
else:
    price = float(btc['Close'].iloc[-1])
    prev = float(btc['Close'].iloc[-2])
    change = ((price - prev) / prev) * 100

    col1.metric("BTC Price (USD)", f"${price:,.0f}", f"{change:.2f}%")
    col2.metric("50D SMA", f"${btc['SMA_50'].iloc[-1]:,.0f}" if 'SMA_50' in btc else "N/A")
    col3.metric("200D SMA", f"${btc['SMA_200'].iloc[-1]:,.0f}" if 'SMA_200' in btc else "N/A")

# -----------------------------
# Price Chart with Support & Resistance
# -----------------------------
st.subheader("📈 Price Chart + Support & Resistance")

if not btc.empty:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(btc.index, btc['Close'], label='Price')

    if 'SMA_50' in btc:
        ax.plot(btc.index, btc['SMA_50'], label='SMA 50')
    if 'SMA_200' in btc:
        ax.plot(btc.index, btc['SMA_200'], label='SMA 200')

    for level in levels[-8:]:
        ax.axhline(level, linestyle='--', alpha=0.4)

    ax.legend()
    ax.set_ylabel("USD")
    st.pyplot(fig)
else:
    st.info("Loading price data...")

# -----------------------------
# RSI Chart
# -----------------------------
st.subheader("📉 RSI Indicator")

if not btc.empty and 'RSI' in btc:
    fig2, ax2 = plt.subplots(figsize=(12, 3))
    ax2.plot(btc.index, btc['RSI'])
    ax2.axhline(70, linestyle='--')
    ax2.axhline(30, linestyle='--')
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI")
    st.pyplot(fig2)
else:
    st.info("Not enough data for RSI.")

# -----------------------------
# AI Prediction Models
# -----------------------------
st.subheader("🤖 AI Price Predictions")

if len(btc) > 100:
    df = btc.copy()
    df['t'] = np.arange(len(df))

    X = df[['t']]
    y = df['Close']

    lr = LinearRegression()
    lr.fit(X, y)

    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X, y)

    future_t = np.arange(len(df), len(df) + predict_days).reshape(-1, 1)

    lr_pred = lr.predict(future_t)
    rf_pred = rf.predict(future_t)

    future_dates = pd.date_range(df.index[-1], periods=predict_days+1, freq='D')[1:]

    fig4, ax4 = plt.subplots(figsize=(12, 4))
    ax4.plot(df.index, df['Close'], label='Historical')
    ax4.plot(future_dates, lr_pred, label='Linear Regression Forecast')
    ax4.plot(future_dates, rf_pred, label='Random Forest Forecast')
    ax4.legend()
    st.pyplot(fig4)
    st.caption("⚠️ AI forecasts are experimental and not financial advice.")
else:
    st.info("Not enough data for AI predictions yet.")

# -----------------------------
# Trend Projection
# -----------------------------
st.subheader("🔮 Short-Term Trend Projection")

if len(btc) > 30:
    y = btc['Close'].values
    X = np.arange(len(y))
    coef = np.polyfit(X[-30:], y[-30:], 1)
    trend = coef[0] * X + coef[1]

    fig3, ax3 = plt.subplots(figsize=(12, 4))
    ax3.plot(btc.index, btc['Close'], label='Actual')
    ax3.plot(btc.index, trend, label='Trend')
    ax3.legend()
    st.pyplot(fig3)
else:
    st.info("Not enough data for projection.")




