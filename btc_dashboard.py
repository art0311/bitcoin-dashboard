# Bitcoin Live Dashboard using Streamlit
# Run with: streamlit run btc_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(page_title="Bitcoin Live Dashboard", layout="wide")

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.title("⚙️ Settings")

symbol = st.sidebar.selectbox("Asset", ["BTC-USD"], index=0)
period = st.sidebar.selectbox("Time Range", ["1y", "2y", "5y", "max"], index=0)
refresh = st.sidebar.slider("Refresh (seconds)", 30, 300, 60)

# -----------------------------
# Data Loader
# -----------------------------
@st.cache_data(ttl=60)
def load_data(symbol, period):
    df = yf.download(symbol, period=period)
    df = df[['Close']]
    df.dropna(inplace=True)
    return df

btc = load_data(symbol, period)

# -----------------------------
# Indicators
# -----------------------------
btc['SMA_50'] = btc['Close'].rolling(50).mean()
btc['SMA_200'] = btc['Close'].rolling(200).mean()

# RSI
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
# Metrics
# -----------------------------
col1, col2, col3 = st.columns(3)

price = btc['Close'].iloc[-1]
change_24h = btc['Close'].pct_change().iloc[-1] * 100

col1.metric("BTC Price (USD)", f"${price:,.0f}", f"{change_24h:.2f}%")
col2.metric("50D SMA", f"${btc['SMA_50'].iloc[-1]:,.0f}")
col3.metric("200D SMA", f"${btc['SMA_200'].iloc[-1]:,.0f}")

# -----------------------------
# Price Chart
# -----------------------------
st.subheader("📈 Price Chart")

fig, ax = plt.subplots(figsize=(12,5))
ax.plot(btc.index, btc['Close'], label='Price')
ax.plot(btc.index, btc['SMA_50'], label='SMA 50')
ax.plot(btc.index, btc['SMA_200'], label='SMA 200')
ax.legend()
ax.set_ylabel("USD")
st.pyplot(fig)

# -----------------------------
# RSI Chart
# -----------------------------
st.subheader("📉 RSI Indicator")

fig2, ax2 = plt.subplots(figsize=(12,3))
ax2.plot(btc.index, btc['RSI'])
ax2.axhline(70, linestyle='--')
ax2.axhline(30, linestyle='--')
ax2.set_ylim(0,100)
ax2.set_ylabel("RSI")
st.pyplot(fig2)

# -----------------------------
# Simple Forecast (Linear Regression)
# -----------------------------
st.subheader("🔮 Short-Term Forecast (Experimental)")

window = 30
X = np.arange(len(btc)).reshape(-1,1)
y = btc['Close'].values

coef = np.polyfit(X.flatten()[-window:], y[-window:], 1)
trend = coef[0] * X.flatten() + coef[1]

fig3, ax3 = plt.subplots(figsize=(12,4))
ax3.plot(btc.index, btc['Close'], label='Actual')
ax3.plot(btc.index, trend, label='Trend Projection')
ax3.legend()
st.pyplot(fig3)

# -----------------------------
# Auto Refresh
# -----------------------------
st.experimental_rerun()
