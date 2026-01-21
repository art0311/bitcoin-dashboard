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

st.set_page_config(page_title="Crypto Live Dashboard", layout="wide")

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
def fetch_spot_etf_flow_usdm_debug(asset_label: str):
    """
    Returns (flow_usdm, debug_dict)
    flow_usdm = float in USD millions (e.g., 3000.0 means $3.0B)
    """
    urls = [
        "https://defillama2.llamao.fi/etfs",  # mirror (usually works on Streamlit Cloud)
        "https://defillama.com/etfs",
    ]

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/html",
        "Accept-Language": "en-US,en;q=0.9",
    }

    dbg = {"asset": asset_label, "tried": []}

    if asset_label not in ("Bitcoin", "Ethereum"):
        dbg["error"] = "asset_label must be 'Bitcoin' or 'Ethereum'"
        return None, dbg

    # Find asset section, then find Flows value near it
    value_pat = re.compile(
        r"Flows[\s\S]{0,250}?([+-])?\$?\s*([0-9][0-9,\.]*)\s*([mMbB])",
        re.IGNORECASE
    )

    for url in urls:
        info = {"url": url, "status": None, "len": None, "matched": False, "flows_snippet": None}
        try:
            r = requests.get(url, headers=headers, timeout=15)
            info["status"] = r.status_code
            html = r.text or ""
            info["len"] = len(html)

            if r.status_code != 200:
                dbg["tried"].append(info)
                continue

            idx = html.lower().find(asset_label.lower())
            if idx == -1:
                dbg["tried"].append(info)
                continue

            window = html[idx: idx + 8000]  # big enough window
            fidx = window.lower().find("flows")
            if fidx != -1:
                info["flows_snippet"] = window[max(0, fidx - 150): fidx + 220]

            m = value_pat.search(window)
            if not m:
                dbg["tried"].append(info)
                continue

            sign = -1.0 if (m.group(1) == "-") else 1.0
            num = float(m.group(2).replace(",", ""))
            unit = m.group(3).lower()

            flow_usdm = sign * num * (1000.0 if unit == "b" else 1.0)

            info["matched"] = True
            dbg["tried"].append(info)
            return flow_usdm, dbg

        except Exception as e:
            info["error"] = str(e)
            dbg["tried"].append(info)
            continue

    return None, dbg




# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("⚙️ Settings")

symbol = st.sidebar.selectbox("Asset", ["BTC-USD", "ETH-USD"], index=0)


ASSET_TICKER = symbol.replace("-USD", "")
ASSET_NAME = "Bitcoin" if symbol == "BTC-USD" else "Ethereum"
ETF_ASSET_LABEL = ASSET_NAME

st.sidebar.markdown("---")
st.sidebar.subheader(f"🏦 Spot {ASSET_TICKER} ETF Flows (US$mm)")

flow, etf_debug = fetch_spot_etf_flow_usdm_debug(ETF_ASSET_LABEL)

# Debug expander (super important)
with st.sidebar.expander("ETF debug", expanded=False):
    st.write(etf_debug)

if flow is None or (isinstance(flow, (int, float)) and not np.isfinite(flow)):
    st.sidebar.info("ETF flow data unavailable.")
else:
    label = (
        "🟢 Net Inflow" if flow > 0 else
        "🔴 Net Outflow" if flow < 0 else
        "🟡 Flat"
    )

    st.sidebar.metric(
        "Latest Daily ETF Flow",
        f"{flow:,.1f} US$mm",
        label
    )

    st.sidebar.caption("Source: DefiLlama (flows attributed to Farside)")

btc = load_data(symbol, period)

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
st.sidebar.subheader(f"🏦 Spot {ASSET_TICKER} ETF Flows (US$mm)")

flow, etf_debug = fetch_spot_etf_flow_usdm_debug(ETF_ASSET_LABEL)

# Store a small history in session_state so we can do 7-entry rolling trend
# (persists while the app stays running)
hist_key = f"etf_flow_hist_{ASSET_TICKER}"
if hist_key not in st.session_state:
    st.session_state[hist_key] = []

# Update history if we got a valid number
if flow is not None and np.isfinite(flow):
    hist = st.session_state[hist_key]
    # prevent duplicate repeats on reruns if the value hasn't changed
    if not hist or hist[-1] != float(flow):
        hist.append(float(flow))
    # keep last 30 values
    st.session_state[hist_key] = hist[-30:]

if flow is None or (isinstance(flow, (int, float)) and not np.isfinite(flow)):
    st.sidebar.info("ETF flow data unavailable.")
else:
    label = "🟢 Net Inflow" if flow > 0 else ("🔴 Net Outflow" if flow < 0 else "🟡 Flat")
    st.sidebar.metric("Latest Daily ETF Flow", f"{flow:,.1f} US$mm", label)
    st.sidebar.caption("Source: DefiLlama (flows attributed to Farside on page)")

# --- 7-entry rolling trend label (based on stored history) ---
hist = st.session_state.get(hist_key, [])
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
# Support & Resistance (separate supports/resistances)
# -----------------------------
def detect_support_resistance(series, window=20, tolerance=0.015, pivot_eps=0.002):
    prices = pd.Series(series).dropna().to_numpy(dtype=float)
    n = len(prices)
    if n < (2 * window + 1):
        return [], []

    supports = []
    resistances = []

    for i in range(window, n - window):
        seg = prices[i - window : i + window + 1]
        cur = prices[i]
        seg_min = float(np.min(seg))
        seg_max = float(np.max(seg))

        # Near-min / near-max pivots
        if cur <= seg_min * (1 + pivot_eps):
            supports.append(cur)
        if cur >= seg_max * (1 - pivot_eps):
            resistances.append(cur)

    def cluster(levels):
        if not levels:
            return []
        levels = sorted(levels)
        clustered = [levels[0]]
        for lvl in levels[1:]:
            if abs(lvl - clustered[-1]) / max(abs(clustered[-1]), 1e-9) > tolerance:
                clustered.append(lvl)
        return clustered

    return cluster(supports), cluster(resistances)


# Adaptive window per range (works across all)
supports, resistances = [], []
sr_window = None

if has_data(btc, ["Close"]):
    n = len(btc)
    sr_window = int(np.clip(n * 0.12, 4, 40))  # good across 1mo -> max
    sr_tol = 0.02 if n < 60 else 0.015

    supports, resistances = detect_support_resistance(
        btc["Close"],
        window=sr_window,
        tolerance=sr_tol,
        pivot_eps=0.002
    )



# -----------------------------
# Header + Top Metrics (now includes Score + Distance)
# -----------------------------
st.title(f"📊 {ASSET_NAME} Live Dashboard")
st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

if has_data(btc, ["Close"]) and len(btc) >= 2:
    price = float(btc["Close"].iloc[-1])
    prev = float(btc["Close"].iloc[-2])
    change = (price - prev) / prev * 100 if prev != 0 else 0.0

    col1.metric(f"{ASSET_TICKER} Price", f"${price:,.0f}", f"{change:.2f}%")

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

# Debug (optional)
# st.caption(f"S/R debug — n={len(btc)}, window={sr_window}, supports={len(supports)}, resistances={len(resistances)}")

if has_data(btc, ["Close"]):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(btc.index, btc["Close"], label="Price")

    # Overlays (only plot if column exists)
    if "SMA_short" in btc.columns:  ax.plot(btc.index, btc["SMA_short"], label="SMA Short")
    if "SMA_medium" in btc.columns: ax.plot(btc.index, btc["SMA_medium"], label="SMA Medium")
    if "SMA_long" in btc.columns:   ax.plot(btc.index, btc["SMA_long"], label="SMA Long", linewidth=2.0)

    if "VWAP" in btc.columns:       ax.plot(btc.index, btc["VWAP"], label="VWAP", linestyle="--")
    if "BB_upper" in btc.columns:   ax.plot(btc.index, btc["BB_upper"], label="BB Upper", linestyle="--", alpha=0.5)
    if "BB_lower" in btc.columns:   ax.plot(btc.index, btc["BB_lower"], label="BB Lower", linestyle="--", alpha=0.5)

    # -----------------------------
    # Support / Resistance with labels
    # -----------------------------
    price_now = float(btc["Close"].iloc[-1])

    # Supports: take closest supports below price
    support_below = [s for s in supports if s < price_now]
    support_below = sorted(support_below)[-6:]  # up to 6 supports

    for lvl in support_below:
        ax.axhline(lvl, linestyle="--", linewidth=1.3, alpha=0.85)
        ax.text(
            btc.index[-1], lvl, f"S: {lvl:,.0f}",
            va="center", ha="left", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2)
        )

    # ONE Resistance: nearest above price, with fallback
    res_above = [r for r in resistances if r > price_now]
    r1 = min(res_above) if res_above else (max(resistances) if resistances else None)

    if r1 is not None:
        ax.axhline(r1, linestyle="--", linewidth=2.0, alpha=0.95)
        ax.text(
            btc.index[-1], r1, f"R: {r1:,.0f}",
            va="center", ha="left", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2)
        )

    ax.set_ylabel("USD")
    ax.legend()
    st.pyplot(fig)

else:
    st.info("Not enough data to plot price.")



# -----------------------------
# Draw Supports (multiple) + ONE Resistance (nearest above price)
# -----------------------------
price_now = float(btc["Close"].iloc[-1])

# Supports: keep only those below price, choose a few closest ones
support_below = [s for s in supports if s < price_now]
support_below = sorted(support_below)[-6:]  # last 6 closest supports

for lvl in support_below:
    ax.axhline(lvl, linestyle="--", linewidth=1.4, alpha=0.85, color="green")
    ax.text(
        btc.index[-1], lvl, f"{lvl:,.0f}",
        va="center", ha="left", fontsize=9, color="green",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2)
    )

# ONE Resistance: nearest above current price
res_above = [r for r in resistances if r > price_now]

if res_above:
    # Normal case — resistance above price
    r1 = min(res_above)

else:
    # Fallback — price already above all resistances
    r1 = max(resistances) if resistances else None

if r1 is not None:
    ax.axhline(
        r1,
        linestyle="--",
        linewidth=2.2,
        alpha=0.95,
        color="red"
    )

    ax.text(
        btc.index[-1],
        r1,
        f"R: {r1:,.0f}",
        va="center",
        ha="left",
        fontsize=9,
        color="red",
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.80,
            pad=2
        )
    )



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
st.subheader("📉 RSI (Relative Strength Index)")
st.caption(
    "Measures how fast and how far price has moved recently. "
    "Values above 70 suggest the asset may be overbought, "
    "while values below 30 suggest it may be oversold."
)

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
st.subheader("💰 OBV (On-Balance Volume)")
st.caption(
    "Tracks whether volume is flowing into or out of the asset. "
    "Rising OBV suggests buyers are in control, "
    "while falling OBV indicates selling pressure."
)

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
st.subheader("📊 MACD (Moving Average Convergence Divergence)")
st.caption(
    "Shows trend direction and momentum by comparing two moving averages. "
    "When MACD crosses above its signal line, momentum is turning bullish; "
    "a cross below suggests weakening or bearish momentum."
)

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
# AI Forecasts + Summary Panel (interpretable + action hint)
# -----------------------------
st.subheader("🤖 AI Forecasts")

def _trend_slope(y_vals: np.ndarray) -> float:
    y_vals = np.asarray(y_vals, dtype=float)
    if len(y_vals) < 3:
        return 0.0
    x = np.arange(len(y_vals), dtype=float)
    m, _b = np.polyfit(x, y_vals, 1)
    return float(m)

def _clamp(x, lo, hi):
    return max(lo, min(hi, x))

def action_hint(bias_score, conf_score, agree, regime, flow, dist_pct):
    # classify
    if bias_score >= 60:
        bias = "bull"
    elif bias_score <= 40:
        bias = "bear"
    else:
        bias = "neutral"

    conf = "high" if conf_score >= 70 else ("med" if conf_score >= 45 else "low")

    etf = None
    if flow is not None and np.isfinite(flow):
        etf = "in" if flow > 0 else ("out" if flow < 0 else "flat")

    headline = "🧭 Action hint: Wait for clearer confirmation."
    badge = "🟡"
    bullets = [
        "Signals are mixed. Use support/resistance and wait for a clean break or bounce.",
        "Consider smaller position size (or paper trade) until confidence improves."
    ]

    if bias == "bull" and conf == "high" and agree and regime == "Bull":
        badge = "🟢"
        headline = "🧭 Action hint: Trend-following environment (bullish)."
        bullets = [
            "Prefer buy-the-dip setups near support or the SMA Long; avoid chasing spikes.",
            "If price holds above the nearest support and AI stays bullish, trend continuation is more likely.",
            "Use the nearest resistance (R1) as a first target/decision point."
        ]
        if etf == "in":
            bullets.append("ETF flows are positive: institutional demand supports the uptrend.")
        elif etf == "out":
            bullets.append("ETF flows are negative: be more selective (expect pullbacks).")

    elif bias == "bear" and conf == "high" and agree and regime == "Bear":
        badge = "🔴"
        headline = "🧭 Action hint: Downtrend environment (bearish)."
        bullets = [
            "Avoid adding risk on rallies; rallies often retrace back down in strong downtrends.",
            "Look for lower highs near resistance; treat support breaks as risk warnings.",
            "If price is below SMA Long and AI stays bearish, downside continuation is more likely."
        ]
        if etf == "out":
            bullets.append("ETF flows are negative: distribution pressure can amplify downside moves.")
        elif etf == "in":
            bullets.append("ETF flows are positive: counter-trend bounce risk—be cautious.")

    elif regime == "Range / Transition" or (conf == "low") or (not agree):
        badge = "🟡"
        headline = "🧭 Action hint: Choppy/range conditions—focus on risk control."
        bullets = [
            "In chop, indicators whipsaw—signals flip quickly. Smaller size and tighter risk limits help.",
            "Use support as a bounce zone and resistance as a fade zone; avoid mid-range entries.",
            "Wait for confirmation: a break + hold above resistance (bull) or below support (bear)."
        ]
        if etf == "in":
            bullets.append("ETF flows are positive: chop could resolve upward—wait for breakout confirmation.")
        elif etf == "out":
            bullets.append("ETF flows are negative: chop could resolve downward—wait for breakdown confirmation.")

    if dist_pct is not None and np.isfinite(dist_pct):
        if bias == "bull" and dist_pct > 8:
            bullets.append("Price is extended above SMA Long: consider waiting for a pullback rather than buying immediately.")
        if bias == "bear" and dist_pct < -8:
            bullets.append("Price is far below SMA Long: downside may be crowded—watch for sharp bounces near support.")

    return headline, bullets, badge


if has_data(btc, ["Close"]) and len(btc) >= 25:

    df = btc[["Close"]].dropna().copy()
    df["t"] = np.arange(len(df))
    X = df[["t"]]
    y = df["Close"]

    # Fit models
    lr = LinearRegression().fit(X, y)
    rf = RandomForestRegressor(n_estimators=200, random_state=42).fit(X, y)

    # Forecast horizon (future candles)
    steps = int(predict_steps)
    future_t = np.arange(len(df), len(df) + steps).reshape(-1, 1)
    lr_pred = lr.predict(future_t)
    rf_pred = rf.predict(future_t)

    # Match spacing to selected timeframe
    if period == "1d":
        freq = "5min"
    elif period == "7d":
        freq = "15min"
    else:
        freq = "1D"

    future_dates = pd.date_range(df.index[-1], periods=steps + 1, freq=freq)[1:]

    # Confidence band (LR residual std)
    residuals = (y - lr.predict(X)).to_numpy(dtype=float)
    resid_std = float(np.std(residuals)) if len(residuals) >= 3 else 0.0
    upper = lr_pred + resid_std
    lower = lr_pred - resid_std

    # ---- Bias / confidence / agreement ----
    tail = max(10, min(50, steps))
    lr_slope = _trend_slope(lr_pred[-tail:])
    rf_slope = _trend_slope(rf_pred[-tail:])

    last_price = float(df["Close"].iloc[-1])
    lr_slope_pct = (lr_slope / max(last_price, 1e-9)) * 100.0
    rf_slope_pct = (rf_slope / max(last_price, 1e-9)) * 100.0

    agree = (lr_slope_pct >= 0 and rf_slope_pct >= 0) or (lr_slope_pct <= 0 and rf_slope_pct <= 0)
    agreement_label = "High ✅" if agree else "Low ❌"

    recent_returns = df["Close"].pct_change().dropna().to_numpy(dtype=float)
    vol = float(np.std(recent_returns)) if len(recent_returns) >= 3 else 0.0
    vol_penalty = _clamp(1.0 - (vol * 80.0), 0.25, 1.0)
    agreement_boost = 1.0 if agree else 0.7

    macd_last = None
    if has_data(btc, ["MACD"]) and btc["MACD"].notna().sum() >= 2:
        macd_last = float(btc["MACD"].dropna().iloc[-1])
    macd_bonus = 0.15 if (macd_last is not None and macd_last > 0) else (-0.15 if (macd_last is not None and macd_last < 0) else 0.0)

    slope_combo = 0.6 * lr_slope_pct + 0.4 * rf_slope_pct
    raw_bias = 50.0 + (slope_combo * 120.0) + (macd_bonus * 100.0)
    bias_score = _clamp(raw_bias, 0.0, 100.0)

    if bias_score >= 60:
        bias_label = f"🟢 Bullish ({bias_score:.0f}%)"
    elif bias_score <= 40:
        bias_label = f"🔴 Bearish ({bias_score:.0f}%)"
    else:
        bias_label = f"🟡 Neutral ({bias_score:.0f}%)"

    resid_pct = (resid_std / max(last_price, 1e-9)) * 100.0
    resid_penalty = _clamp(1.0 - (resid_pct * 8.0), 0.25, 1.0)
    conf_score = _clamp(100.0 * vol_penalty * resid_penalty * agreement_boost, 0.0, 100.0)

    if conf_score >= 70:
        conf_label = f"High ({conf_score:.0f}%)"
        conf_emoji = "🟢"
    elif conf_score >= 45:
        conf_label = f"Medium ({conf_score:.0f}%)"
        conf_emoji = "🟡"
    else:
        conf_label = f"Low ({conf_score:.0f}%)"
        conf_emoji = "🔴"

    # Expected range (simple)
    exp_low = float(np.min(lower)) if len(lower) else np.nan
    exp_high = float(np.max(upper)) if len(upper) else np.nan

    # Pull other dashboard context if available
    regime_val = regime if "regime" in locals() else "Unknown"
    flow_val = flow if "flow" in locals() else None
    dist_val = dist if "dist" in locals() else None

    hint_headline, hint_bullets, hint_badge = action_hint(
        bias_score=bias_score,
        conf_score=conf_score,
        agree=agree,
        regime=regime_val,
        flow=flow_val,
        dist_pct=dist_val
    )

    # -----------------------------
    # Render summary panel
    # -----------------------------
    with st.expander("📌 AI Forecast Summary (how to read this)", expanded=True):

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Bias", bias_label)
        c2.metric("Confidence", f"{conf_emoji} {conf_label}")
        c3.metric("Model Agreement", agreement_label)
        c4.metric("Horizon", f"{steps} steps ({freq})")

        if np.isfinite(exp_low) and np.isfinite(exp_high):
            st.markdown(
                f"**Expected range (next {steps} steps):** "
                f"Low **${exp_low:,.0f}**  •  High **${exp_high:,.0f}**"
            )
        else:
            st.caption("Expected range unavailable.")

        st.markdown("---")
        st.markdown(f"### {hint_badge} Action Hint")
        st.markdown(f"**{hint_headline}**")
        for bullet in hint_bullets:
            st.markdown(f"- {bullet}")
        st.caption("Educational only — not financial advice.")

    # -----------------------------
    # Plot forecast (OUTSIDE expander)
    # -----------------------------
    figa, axa = plt.subplots(figsize=(12, 4))
    axa.plot(df.index, df["Close"], label="Historical", linewidth=2)
    axa.plot(future_dates, lr_pred, label="Linear Regression")
    axa.plot(future_dates, rf_pred, label="Random Forest")

    if resid_std > 0:
        axa.fill_between(future_dates, lower, upper, alpha=0.15, label="Confidence Range (±1σ)")

    axa.legend()
    st.pyplot(figa)

else:
    st.info("Not enough data for AI predictions yet (need ~25+ data points).")
                                                                          






