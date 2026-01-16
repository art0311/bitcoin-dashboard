import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title="Bitcoin Live Dashboard", layout="wide")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("⚙️ Settings")
symbol = st.sidebar.selectbox("Asset", ["BTC-USD"], index=0)
period = st.sidebar.selectbox("Time Range", ["1y", "2y", "5y", "max"], index=0)

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
# Indicators
# -----------------------------
if not btc.empty:
    btc['SMA_50'] = btc['Close'].rolling(50).mean()
    btc['SMA_200'] = btc['Close'].rolling(200).mean()

    delta = btc['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    btc['RSI'] = 100 - (100 / (1 + rs))

# -----------------------------
# Header
# -----------------------------
st.title("📊 Bitcoin Live Market Dashboard")
st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

# -----------------------------
# Metrics (CRASH-PROOF)
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
    col2.metric("50D SMA", f"${btc['SMA_50'].iloc[-1]:,.0f}")
    col3.metric("200D SMA", f"${btc['SMA_200'].iloc[-1]:,.0f}")

# -----------------------------
# Price Chart
# -----------------------------
st.subheader("📈 Price Chart")

if not btc.empty:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(btc.index, btc['Close'], label='Price')
    ax.plot(btc.index, btc['SMA_50'], label='SMA 50')
    ax.plot(btc.index, btc['SMA_200'], label='SMA 200')
    ax.legend()
    ax.set_ylabel("USD")
    st.pyplot(fig)
else:
    st.info("Loading price data...")

# -----------------------------
# RSI Chart
# -----------------------------
st.subheader("📉 RSI Indicator")

if not btc.empty:
    fig2, ax2 = plt.subplots(figsize=(12, 3))
    ax2.plot(btc.index, btc['RSI'])
    ax2.axhline(70, linestyle='--')
    ax2.axhline(30, linestyle='--')
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI")
    st.pyplot(fig2)
else:
    st.info("Loading RSI...")

# -----------------------------
# Simple Forecast (Trend Projection)
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

