import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import requests
import plotly.graph_objects as go
import re

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

def market_regime(btc: pd.DataFrame):
    if btc is None or btc.empty or len(btc) < 15:
        return "Unknown", "⚪", {}
    if not all(c in btc.columns for c in ["Close", "SMA_long", "MACD"]):
        return "Unknown", "⚪", {}

    close = btc["Close"].dropna()
    sma_long = btc["SMA_long"].dropna()
    macd = btc["MACD"].dropna()
    if close.empty or sma_long.empty or macd.empty:
        return "Unknown", "⚪", {}

    price = float(close.iloc[-1])
    smaL = float(sma_long.iloc[-1])
    macd_last = float(macd.iloc[-1])

    lookback = min(10, len(sma_long) - 1)
    slope = float(sma_long.iloc[-1] - sma_long.iloc[-1 - lookback]) if lookback >= 1 else 0.0

    if price > smaL and slope > 0 and macd_last > 0:
        return "Bull", "🟢", {"slope": slope, "macd": macd_last}
    if price < smaL and slope < 0 and macd_last < 0:
        return "Bear", "🔴", {"slope": slope, "macd": macd_last}
    return "Range / Transition", "🟡", {"slope": slope, "macd": macd_last}

def distance_from_sma_long_pct(btc: pd.DataFrame):
    if not has_data(btc, ["Close", "SMA_long"]):
        return None
    c = btc["Close"].dropna()
    s = btc["SMA_long"].dropna()
    if c.empty or s.empty:
        return None
    price = float(c.iloc[-1])
    smaL = float(s.iloc[-1])
    if smaL == 0:
        return None
    return (price - smaL) / smaL * 100.0

def market_strength_score(btc: pd.DataFrame):
    """
    Simple 0-100 score using signals you already have:
    +20 Price > SMA Long
    +20 SMA Long slope up
    +20 MACD > 0
    +20 OBV rising (vs 10 bars ago)
    +20 RSI >= 50
    """
    if btc is None or btc.empty or len(btc) < 15:
        return None, "Unknown"

    needed = ["Close", "SMA_long", "MACD", "OBV", "RSI"]
    if not all(c in btc.columns for c in needed):
        return None, "Unknown"

    close = btc["Close"].dropna()
    smaL = btc["SMA_long"].dropna()
    macd = btc["MACD"].dropna()
    obv = btc["OBV"].dropna()
    rsi = btc["RSI"].dropna()

    if close.empty or smaL.empty or macd.empty or obv.empty or rsi.empty:
        return None, "Unknown"

    price = float(close.iloc[-1])
    sma_last = float(smaL.iloc[-1])
    macd_last = float(macd.iloc[-1])
    rsi_last = float(rsi.iloc[-1])

    lookback = min(10, len(smaL) - 1)
    sma_slope = float(smaL.iloc[-1] - smaL.iloc[-1 - lookback]) if lookback >= 1 else 0.0

    obv_lookback = min(10, len(obv) - 1)
    obv_trend = float(obv.iloc[-1] - obv.iloc[-1 - obv_lookback]) if obv_lookback >= 1 else 0.0

    score = 0
    score += 20 if price > sma_last else 0
    score += 20 if sma_slope > 0 else 0
    score += 20 if macd_last > 0 else 0
    score += 20 if obv_trend > 0 else 0
    score += 20 if rsi_last >= 50 else 0

    if score >= 80:
        label = "Strong Bullish"
    elif score >= 60:
        label = "Bullish"
    elif score >= 40:
        label = "Neutral"
    elif score >= 20:
        label = "Bearish"
    else:
        label = "Strong Bearish"

    return score, label

# -----------------------------
# Spot BTC ETF Daily Net Flow (US$mm) - robust (no read_html)
# -----------------------------
@st.cache_data(ttl=900)
def fetch_spot_btc_etf_flow_usdm_debug():
    """
    Returns (flow_usdm, debug_dict)
    flow_usdm: float USD millions, or None
    """
    urls = [
        "https://defillama2.llamao.fi/etfs",
        "https://defillama.com/etfs",
    ]

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/html",
        "Accept-Language": "en-US,en;q=0.9",
    }

    # Handles: Flows-$394.7m, Flows$4.7m, Flows: -$394.7m, etc.
    patterns = [
        r"Bitcoin[\s\S]{0,2500}?Flows\s*-?\s*:?\s*\$?\s*([+-])?\s*([0-9][0-9,\.]*)\s*([mMbB])",
        r"Flows\s*-?\s*:?\s*\$?\s*([+-])?\s*([0-9][0-9,\.]*)\s*([mMbB])",
    ]

    debug = {"tried": []}

    for url in urls:
        entry = {"url": url}
        try:
            r = requests.get(url, headers=headers, timeout=15)
            entry["status"] = r.status_code
            entry["len"] = len(r.text or "")
            html = r.text or ""

            entry["has_Bitcoin"] = ("Bitcoin" in html)
            entry["has_Flows"] = ("Flows" in html)

            # store a small snippet around first "Flows" if present
            idx = html.lower().find("flows")
            if idx != -1:
                start = max(0, idx - 120)
                end = min(len(html), idx + 200)
                entry["flows_snippet"] = html[start:end]
            else:
                entry["flows_snippet"] = None

            match = None
            used = None
            for p in patterns:
                match = re.search(p, html, flags=re.IGNORECASE)
                if match:
                    used = p
                    break

            entry["matched"] = bool(match)
            entry["pattern_used"] = used

            debug["tried"].append(entry)

            if not match:
                continue

            sign = -1.0 if (match.group(1) == "-") else 1.0
            num = float(match.group(2).replace(",", ""))
            unit = match.group(3).lower()

            flow_usdm = sign * num * (1000.0 if unit == "b" else 1.0)
            debug["selected_url"] = url
            debug["parsed_flow"] = flow_usdm
            return flow_usdm, debug

        except Exception as e:
            entry["error"] = str(e)
            debug["tried"].append(entry)
            continue

    return None, debug



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

st.sidebar.markdown("---")
st.sidebar.subheader("🏦 Spot BTC ETF Flows (US$mm)")

flow, etf_debug = fetch_spot_btc_etf_flow_usdm_debug()

# Store a small history in session_state so we can do 7-entry rolling trend
# (persists while the app stays running)
if "etf_flow_hist" not in st.session_state:
    st.session_state.etf_flow_hist = []

# Update history if we got a valid number
if flow is not None and np.isfinite(flow):
    hist = st.session_state.etf_flow_hist
    # prevent duplicate repeats on reruns if the value hasn't changed
    if not hist or hist[-1] != float(flow):
        hist.append(float(flow))
    # keep last 30 values
    st.session_state.etf_flow_hist = hist[-30:]

if flow is None or (isinstance(flow, (int, float)) and not np.isfinite(flow)):
    st.sidebar.info("ETF flow data unavailable.")
else:
    label = "🟢 Net Inflow" if flow > 0 else ("🔴 Net Outflow" if flow < 0 else "🟡 Flat")
    st.sidebar.metric("Latest Daily ETF Flow", f"{flow:,.1f} US$mm", label)
    st.sidebar.caption("Source: DefiLlama (flows attributed to Farside on page)")

# --- 7-entry rolling trend label (based on stored history) ---
hist = st.session_state.get("etf_flow_hist", [])
if len(hist) >= 7:
    last7_sum = float(np.sum(hist[-7:]))
    st.sidebar.metric("7-entry rolling net flow", f"{last7_sum:,.1f} US$mm")

    if len(hist) >= 14:
        prev7_sum = float(np.sum(hist[-14:-7]))
        delta = last7_sum - prev7_sum

        rel = abs(delta) / max(abs(prev7_sum), 1e-9)
        if abs(delta) < 50 and rel < 0.25:
            trend = "🟡 Flat / Mixed"
        elif delta > 0:
            trend = "🟢 Rising (more inflow)"
        else:
            trend = "🔴 Falling (more outflow)"

        st.sidebar.metric("7-entry trend", trend, f"Δ vs prior 7: {delta:+,.1f} US$mm")
    else:
        st.sidebar.caption("7-entry trend: Need 14 values for comparison.")
else:
    st.sidebar.caption("7-entry trend: collecting data (need 7 refreshes).")

with st.sidebar.expander("ETF debug (temporary)"):
    st.write(etf_debug)

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
# Indicators (INDEX-SAFE) + Adaptive SMA windows
# -----------------------------
if has_data(btc, ["Close", "High", "Low", "Volume"]) and len(btc) >= 2:
    close = btc["Close"].astype(float)
    high = btc["High"].astype(float)
    low = btc["Low"].astype(float)
    vol = btc["Volume"].astype(float)

    n = len(btc)

    # Better adaptive SMA windows (scale with data length)
    sma_short_w = int(max(5, round(n * 0.05)))
    sma_med_w   = int(max(10, round(n * 0.15)))
    sma_long_w  = int(max(20, round(n * 0.35)))
    sma_long_w = min(sma_long_w, max(30, n - 2))
    if sma_med_w <= sma_short_w:
        sma_med_w = sma_short_w + 1
    if sma_long_w <= sma_med_w:
        sma_long_w = sma_med_w + 1
    sma_short_w = min(sma_short_w, n)
    sma_med_w   = min(sma_med_w, n)
    sma_long_w  = min(sma_long_w, n)

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

levels = detect_levels(btc["Close"]) if has_data(btc, ["Close"]) else []

# -----------------------------
# Header + Top Metrics (now includes Score + Distance)
# -----------------------------
st.title("📊 Bitcoin Live Dashboard")
st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

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

    regime, emoji, _ = market_regime(btc)
    col5.metric("Market Regime", f"{emoji} {regime}")

    dist = distance_from_sma_long_pct(btc)
    col6.metric("Dist vs SMA Long", f"{dist:+.1f}%" if dist is not None else "N/A")

    score, score_label = market_strength_score(btc)
    col7.metric("Strength Score", f"{score}/100" if score is not None else "N/A", score_label)

else:
    for c in [col1, col2, col3, col4, col5, col6, col7]:
        c.metric("Loading...", "")

# -----------------------------
# Price + Overlays (SMA Long slope coloring)
# -----------------------------
st.subheader("📈 Price + Indicators")

if has_data(btc, ["Close"]):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(btc.index, btc["Close"], label="Price")

    ax.plot(btc.index, btc["SMA_short"], label="SMA Short")
    ax.plot(btc.index, btc["SMA_medium"], label="SMA Medium")

    # SMA Long colored by slope
    sma_long_series = btc["SMA_long"].dropna()
    sma_color = "grey"
    if len(sma_long_series) >= 11:
        slope = float(sma_long_series.iloc[-1] - sma_long_series.iloc[-11])
        if slope > 0:
            sma_color = "green"
        elif slope < 0:
            sma_color = "red"

    ax.plot(btc.index, btc["SMA_long"], label="SMA Long", linewidth=2.4, color=sma_color)

    ax.plot(btc.index, btc["VWAP"], label="VWAP", linestyle="--")
    ax.plot(btc.index, btc["BB_upper"], label="BB Upper", linestyle="--", alpha=0.5)
    ax.plot(btc.index, btc["BB_lower"], label="BB Lower", linestyle="--", alpha=0.5)

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




