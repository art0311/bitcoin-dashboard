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
# Helper function
# -----------------------------
def has_data(df, cols):
    return df is not None and not df.empty and all(col in df.columns for col in cols)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("⚙️ Settings")
symbol = st.sidebar.selectbox("Asset", ["BTC-USD"], index=0)
period = st.sidebar.selectbox("Time Range", ["1d", "7d", "1mo", "1y", "2y", "5y", "max"], index=3)
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
        return int(data['value']), data['value_classification']
    except:
        return None, "N/A"

sentiment_score, sentiment_label = fetch_fng()
if sentiment_score is not None:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sentiment_score,
        gauge={'axis': {'range': [0,100]},
               'bar': {'color': "darkblue"},
               'steps':[{'range':[0,24],'color':'red'},
                        {'range':[25,49],'color':'orange'},
                        {'range':[50,50],'color':'yellow'},
                        {'range':[51,74],'color':'lightgreen'},
                        {'range':[75,100],'color':'green'}]},
        title={'text':sentiment_label}
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
        if period == "1d":
            df = yf.download(symbol, period=period, interval="5m", progress=False)
        elif period == "7d":
            df = yf.download(symbol, period=period, interval="15m", progress=False)
        else:
            df = yf.download(symbol, period=period, progress=False)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        for col in ['Open','High','Low','Close','Volume']:
            if col not in df.columns:
                df[col] = np.nan

        df.dropna(subset=['Close','High','Low','Volume'], inplace=True)
        return df
    except:
        return pd.DataFrame()

btc = load_data(symbol, period)

# -----------------------------
# Adaptive Indicators
# -----------------------------
if has_data(btc, ['Close','High','Low','Volume']) and len(btc) > 1:
    close = btc['Close'].to_numpy(dtype=float)
    high = btc['High'].to_numpy(dtype=float)
    low = btc['Low'].to_numpy(dtype=float)
    volume = btc['Volume'].to_numpy(dtype=float)

    # Adaptive SMA
    sma_short = min(5, len(close))
    sma_medium = min(20, len(close))
    sma_long = min(50, len(close))
    btc['SMA_short'] = pd.Series(close).rolling(sma_short).mean()
    btc['SMA_medium'] = pd.Series(close).rolling(sma_medium).mean()
    btc['SMA_long'] = pd.Series(close).rolling(sma_long).mean()

    # Adaptive Bollinger Bands
    bb_window = min(20,len(close))
    sma20 = pd.Series(close).rolling(bb_window).mean()
    std20 = pd.Series(close).rolling(bb_window).std()
    btc['BB_upper'] = sma20 + 2*std20
    btc['BB_lower'] = sma20 - 2*std20

    # Adaptive RSI
    rsi_window = min(14,len(close)-1)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(rsi_window).mean() / loss.rolling(rsi_window).mean()
    btc['RSI'] = 100 - (100 / (1 + rs))

    # VWAP
    typical_price = (high+low+close)/3
    btc['VWAP'] = (typical_price*volume).cumsum() / volume.cumsum()

    # OBV
    obv = np.zeros(len(close))
    for i in range(1,len(close)):
        if close[i]>close[i-1]: obv[i]=obv[i-1]+volume[i]
        elif close[i]<close[i-1]: obv[i]=obv[i-1]-volume[i]
        else: obv[i]=obv[i-1]
    btc['OBV']=obv

    # Adaptive MACD
    span_fast = min(12,len(close))
    span_slow = min(26,len(close))
    btc['MACD'] = pd.Series(close).ewm(span=span_fast,adjust=False).mean() - pd.Series(close).ewm(span=span_slow,adjust=False).mean()
    btc['MACD_Signal'] = btc['MACD'].ewm(span=min(9,len(close)),adjust=False).mean()
    btc['MACD_Hist'] = btc['MACD'] - btc['MACD_Signal']

else:
    for col in ['SMA_short','SMA_medium','SMA_long','BB_upper','BB_lower','RSI','VWAP','OBV','MACD','MACD_Signal','MACD_Hist']:
        btc[col] = np.nan

# -----------------------------
# Support & Resistance
# -----------------------------
def detect_levels(series, window=20, tolerance=0.015):
    prices = series.to_numpy(dtype=float)
    levels=[]
    for i in range(window,len(prices)-window):
        low_range=prices[i-window:i+window]
        current=prices[i]
        if current==low_range.min() or current==low_range.max(): levels.append(current)
    clustered=[]
    for l in levels:
        if not clustered or abs(l-clustered[-1])/clustered[-1]>tolerance: clustered.append(l)
    return clustered

levels = detect_levels(btc['Close']) if has_data(btc,['Close']) else []

# -----------------------------
# Dashboard Layout
# -----------------------------
st.title("📊 Bitcoin Live Dashboard")
st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

# Metrics
col1,col2,col3 = st.columns(3)
if has_data(btc,['Close']):
    price=float(btc['Close'].iloc[-1])
    prev=float(btc['Close'].iloc[-2])
    change=(price-prev)/prev*100
    col1.metric("BTC Price",f"${price:,.0f}",f"{change:.2f}%")
    col2.metric("SMA Short",f"${btc['SMA_short'].iloc[-1]:,.0f}")
    col3.metric("SMA Medium",f"${btc['SMA_medium'].iloc[-1]:,.0f}")
else:
    col1.metric("BTC Price","Loading..."); col2.metric("SMA Short","Loading..."); col3.metric("SMA Medium","Loading...")

# Price + Indicators
st.subheader("📈 Price + Indicators")
if has_data(btc,['Close']):
    fig,ax = plt.subplots(figsize=(12,5))
    ax.plot(btc.index,btc['Close'],label='Price')
    if 'SMA_short' in btc: ax.plot(btc.index,btc['SMA_short'],label='SMA Short')
    if 'SMA_medium' in btc: ax.plot(btc.index,btc['SMA_medium'],label='SMA Medium')
    if 'SMA_long' in btc: ax.plot(btc.index,btc['SMA_long'],label='SMA Long')
    if 'VWAP' in btc: ax.plot(btc.index,btc['VWAP'],label='VWAP',linestyle='--')
    if 'BB_upper' in btc and 'BB_lower' in btc:
        ax.plot(btc.index,btc['BB_upper'],label='BB Upper',linestyle='--',alpha=0.5)
        ax.plot(btc.index,btc['BB_lower'],label='BB Lower',linestyle='--',alpha=0.5)
    for lvl in levels[-8:]: ax.axhline(lvl,linestyle='--',alpha=0.4)
    ax.legend(); ax.set_ylabel("USD")
    st.pyplot(fig)
else: st.info("Not enough data to plot price.")

# Volume
st.subheader("📊 Volume")
if has_data(btc,['Volume']):
    figv,axv = plt.subplots(figsize=(12,3))
    axv.bar(btc.index,btc['Volume'],alpha=0.6)
    axv.set_ylabel("Volume")
    st.pyplot(figv)
else: st.info("Not enough data for volume.")

# RSI
st.subheader("📉 RSI")
if has_data(btc,['RSI']):
    fig_rsi,ax_rsi = plt.subplots(figsize=(12,3))
    ax_rsi.plot(btc.index,btc['RSI'])
    ax_rsi.axhline(70,linestyle='--'); ax_rsi.axhline(30,linestyle='--')
    ax_rsi.set_ylim(0,100); ax_rsi.set_ylabel("RSI")
    st.pyplot(fig_rsi)
else: st.info("Not enough data for RSI.")

# OBV
st.subheader("💰 OBV")
if has_data(btc,['OBV']):
    fig_obv,ax_obv = plt.subplots(figsize=(12,3))
    ax_obv.plot(btc.index,btc['OBV'])
    ax_obv.set_ylabel("OBV")
    st.pyplot(fig_obv)
else: st.info("Not enough data for OBV.")

# MACD
st.subheader("📊 MACD")
if has_data(btc,['MACD','MACD_Signal','MACD_Hist']):
    fig_macd,ax_macd = plt.subplots(figsize=(12,3))
    ax_macd.plot(btc.index,btc['MACD'],label='MACD')
    ax_macd.plot(btc.index,btc['MACD_Signal'],label='Signal')
    ax_macd.bar(btc.index,btc['MACD_Hist'],alpha=0.3,color='grey',label='Histogram')
    ax_macd.legend()
    st.pyplot(fig_macd)
else: st.info("Not enough data for MACD.")


