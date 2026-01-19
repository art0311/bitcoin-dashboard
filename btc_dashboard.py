import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import requests
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Bitcoin Live Dashboard", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
def has_data(df, cols):
    return df is not None and not df.empty and all(c in df.columns for c in cols)

def safe_last(x):
    try:
        v = float(x.iloc[-1])
        return v if np.isfinite(v) else None
    except Exception:
        return None

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
# Chart Overlay Controls (ONLY affects main chart)
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("📌 Chart Overlays")

OVERLAY_OPTIONS = [
    "SMA Short", "SMA Medium", "SMA Long",
    "VWAP",
    "Bollinger Bands",
    "Support/Resistance",
]

# initialize once
if "overlay_selected" not in st.session_state:
    st.session_state.overlay_selected = ["SMA Short", "SMA Medium", "VWAP", "Bollinger Bands"]

b1, b2 = st.sidebar.columns(2)
if b1.button("Select all"):
    st.session_state.overlay_selected = OVERLAY_OPTIONS
if b2.button("Clear all"):
    st.session_state.overlay_selected = []

overlay_selected = st.sidebar.multiselect(
    "Show overlays on chart",
    options=OVERLAY_OPTIONS,
    default=st.session_state.overlay_selected,
    key="overlay_selected",
)

# -----------------------------
# Fear & Greed (Sidebar Gauge)
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Market Sentiment")

@st.cache_data(ttl=300)
def fetch_fng():
    try:
        r = requests.get("https://api.alternative.me/fng/", timeout=5)
        data = r.json()["data"][0]
        return int(data["value"]), data["value_classification"]
    except Exception:
        return None, "N/A"

sentiment_score, sentiment_label = fetch_fng()
if sentiment_score is not None:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sentiment_score,
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0, 24], "color": "red"},
                {"range": [25, 49], "color": "orange"},
                {"range": [50, 50], "color": "yellow"},
                {"range": [51, 74], "color": "lightgreen"},
                {"range": [75, 100], "color": "green"},
            ],
        },
        title={"text": sentiment_label}
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

        needed = ["Open", "High", "Low", "Close", "Volume"]
        for c in needed:
            if c not in df.columns:
                df[c] = np.nan

        df = df[needed].copy()
        df.dropna(subset=["Close", "High", "Low", "Volume"], inplace=True)
        return df
    except Exception:
        return pd.DataFrame()

btc = load_data(symbol, period)

# -----------------------------
# Indicators (INDEX-SAFE)
# -----------------------------
if has_data(btc, ["Close", "High", "Low", "Volume"]) and len(btc) >= 2:
    close = btc["Close"].astype(float)
    high = btc["High"].astype(float)
    low = btc["Low"].astype(float)
    vol = btc["Volume"].astype(float)

    n = len(btc)

    # Adaptive windows
    sma_short_w = max(3, min(5, n))
    sma_med_w   = max(5, min(20, n))
    sma_long_w  = max(10, min(50, n))

    btc["SMA_short"]  = close.rolling(sma_short_w, min_periods=1).mean()
    btc["SMA_medium"] = close.rolling(sma_med_w,   min_periods=1).mean()
    btc["SMA_long"]   = close.rolling(sma_long_w,  min_periods=1).mean()

    # Bollinger Bands
    bb_w = max(5, min(20, n))
    bb_mid = close.rolling(bb_w, min_periods=1).mean()
    bb_std = close.rolling(bb_w, min_periods=2).std()
    btc["BB_upper"] = bb_mid + 2 * bb_std
    btc["BB_lower"] = bb_mid - 2 * bb_std

    # RSI
    rsi_w = max(5, min(14, n - 1))
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(rsi_w, min_periods=1).mean()
    avg_loss = loss.rolling(rsi_w, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    btc["RSI"] = 100 - (100 / (1 + rs))

    # VWAP
    typical = (high + low + close) / 3.0
    denom = vol.replace(0, np.nan).cumsum()
    btc["VWAP"] = (typical * vol).cumsum() / denom

    # OBV
    direction = np.sign(close.diff()).fillna(0)
    btc["OBV"] = (direction * vol).fillna(0).cumsum()

    # MACD
    if period in ["1d", "7d"]:
        fast, slow, sig = 6, 13, 4
    else:
        fast, slow, sig = 12, 26, 9

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    btc["MACD"] = ema_fast - ema_slow
    btc["MACD_Signal"] = btc["MACD"].ewm(span=sig, adjust=False).mean()
    btc["MACD_Hist"] = btc["MACD"] - btc["MACD_Signal"]

else:
    for c in ["SMA_short", "SMA_medium", "SMA_long", "BB_upper", "BB_lower", "RSI",
              "VWAP", "OBV", "MACD", "MACD_Signal", "MACD_Hist"]:
        btc[c] = np.nan

# -----------------------------
# Support & Resistance
# -----------------------------
def detect_levels(series, window=20, tolerance=0.015):
    prices = series.to_numpy(dtype=float)
    if len(prices) < (2 * window + 1):
        return []
    levels = []
    for i in range(window, len(prices) - window):
        seg = prices[i-window:i+window]
        cur = prices[i]
        if cur == np.min(seg) or cur == np.max(seg):
            levels.append(cur)
    levels = sorted(levels)
    clustered = []
    for lvl in levels:
        if not clustered:
            clustered.append(lvl)
        else:
            if abs(lvl - clustered[-1]) / max(clustered[-1], 1e-9) > tolerance:
                clustered.append(lvl)
    return clustered

levels = detect_levels(btc["Close"]) if (has_data(btc, ["Close"]) and "Support/Resistance" in overlay_selected) else []

# -----------------------------
# Header + Metrics
# -----------------------------
st.title("📊 Bitcoin Live Dashboard")
st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

col1, col2, col3, col4 = st.columns(4)

if has_data(btc, ["Close"]) and len(btc) >= 2:
    price = float(btc["Close"].iloc[-1])
    prev = float(btc["Close"].iloc[-2])
    change = (price - prev) / prev * 100 if prev != 0 else 0.0
    col1.metric("BTC Price", f"${price:,.0f}", f"{change:.2f}%")

    s1 = safe_last(btc["SMA_short"])
    s2 = safe_last(btc["SMA_medium"])
    s3 = safe_last(btc["SMA_long"])

    col2.metric("SMA Short", f"${s1:,.0f}" if s1 is not None else "N/A")
    col3.metric("SMA Medium", f"${s2:,.0f}" if s2 is not None else "N/A")
    col4.metric("SMA Long", f"${s3:,.0f}" if s3 is not None else "N/A")
else:
    col1.metric("BTC Price", "Loading...", "")
    col2.metric("SMA Short", "Loading...")
    col3.metric("SMA Medium", "Loading...")

# -----------------------------
# Price + Overlays (controlled by sidebar)
# -----------------------------
st.subheader("📈 Price + Indicators")

if has_data(btc, ["Close"]):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(btc.index, btc["Close"], label="Price")

    # Only plot what is selected
    if "SMA Short" in overlay_selected:
        ax.plot(btc.index, btc["SMA_short"], label="SMA Short")
    if "SMA Medium" in overlay_selected:
        ax.plot(btc.index, btc["SMA_medium"], label="SMA Medium")
    if "SMA Long" in overlay_selected:
        ax.plot(btc.index, btc["SMA_long"], label="SMA Long")

    if "VWAP" in overlay_selected:
        ax.plot(btc.index, btc["VWAP"], label="VWAP", linestyle="--")

    if "Bollinger Bands" in overlay_selected:
        ax.plot(btc.index, btc["BB_upper"], label="BB Upper", linestyle="--", alpha=0.5)
        ax.plot(btc.index, btc["BB_lower"], label="BB Lower", linestyle="--", alpha=0.5)

    if "Support/Resistance" in overlay_selected:
        for lvl in levels[-8:]:
            ax.axhline(lvl, linestyle="--", alpha=0.35)

    ax.set_ylabel("USD")
    ax.legend()
    st.pyplot(fig)
else:
    st.info("Not enough data to plot price.")

# -----------------------------
# Volume
# -----------------------------
st.subheader("📊 Volume")
if has_data(btc, ["Volume"]):
    figv, axv = plt.subplots(figsize=(12, 3))
    axv.bar(btc.index, btc["Volume"], alpha=0.6)
    axv.set_ylabel("Volume")
    st.pyplot(figv)
else:
    st.info("Not enough data for volume.")

# -----------------------------
# RSI
# -----------------------------
st.subheader("📉 RSI")
if has_data(btc, ["RSI"]) and btc["RSI"].notna().sum() >= 2:
    figr, axr = plt.subplots(figsize=(12, 3))
    axr.plot(btc.index, btc["RSI"])
    axr.axhline(70, linestyle="--")
    axr.axhline(30, linestyle="--")
    axr.set_ylim(0, 100)
    axr.set_ylabel("RSI")
    st.pyplot(figr)
else:
    st.info("Not enough data for RSI.")

# -----------------------------
# OBV
# -----------------------------
st.subheader("💰 OBV")
if has_data(btc, ["OBV"]) and btc["OBV"].notna().sum() >= 2:
    figo, axo = plt.subplots(figsize=(12, 3))
    axo.plot(btc.index, btc["OBV"])
    axo.set_ylabel("OBV")
    st.pyplot(figo)
else:
    st.info("Not enough data for OBV.")

# -----------------------------
# MACD
# -----------------------------
st.subheader("📊 MACD")
if has_data(btc, ["MACD", "MACD_Signal", "MACD_Hist"]) and btc["MACD"].notna().sum() >= 2:
    figm, axm = plt.subplots(figsize=(12, 3))
    axm.plot(btc.index, btc["MACD"], label="MACD")
    axm.plot(btc.index, btc["MACD_Signal"], label="Signal")
    axm.bar(btc.index, btc["MACD_Hist"], alpha=0.3, label="Histogram")
    axm.legend()
    st.pyplot(figm)
else:
    st.info("Not enough data for MACD.")

# -----------------------------
# AI Forecasts
# -----------------------------
st.subheader("🤖 AI Forecasts")
if has_data(btc, ["Close"]) and len(btc) > 100:
    df = btc[["Close"]].dropna().copy()
    df["t"] = np.arange(len(df))

    X = df[["t"]]
    y = df["Close"]

    lr = LinearRegression().fit(X, y)
    rf = RandomForestRegressor(n_estimators=200, random_state=42).fit(X, y)

    future_t = np.arange(len(df), len(df) + predict_days).reshape(-1, 1)
    lr_pred = lr.predict(future_t)
    rf_pred = rf.predict(future_t)

    future_dates = pd.date_range(df.index[-1], periods=predict_days + 1, freq="D")[1:]

    figa, axa = plt.subplots(figsize=(12, 4))
    axa.plot(df.index, df["Close"], label="Historical")
    axa.plot(future_dates, lr_pred, label="Linear Regression")
    axa.plot(future_dates, rf_pred, label="Random Forest")
    axa.legend()
    st.pyplot(figa)
    st.caption("⚠️ Forecasts are experimental, not financial advice.")
else:
    st.info("Not enough data for AI predictions yet.")



