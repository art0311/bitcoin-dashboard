import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timezone
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

def nearest_sr_from_lists(price: float, supports, resistances):
    """Return nearest valid support below price and resistance above price."""
    if price is None or not np.isfinite(price):
        return None, None

    valid_supports = sorted(float(x) for x in supports if np.isfinite(x) and float(x) < price)
    valid_resistances = sorted(float(x) for x in resistances if np.isfinite(x) and float(x) > price)

    nearest_support = valid_supports[-1] if valid_supports else None
    nearest_resistance = valid_resistances[0] if valid_resistances else None
    return nearest_support, nearest_resistance

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
# US spot Bitcoin ETF daily flows (USD millions).
# Free JSON from The Commoner Bitcoin ETF Tracker by TFTC (CC BY 4.0).
# For ETH, retain the legacy DefiLlama lookup for now rather than display BTC data as ETH.
# -----------------------------
ETF_BTC_FEED = "https://www.tftc.io/bitcoin-etf-flows/data.json"
ETF_BTC_SOURCE = "https://www.tftc.io/bitcoin-etf-flows"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_bitcoin_etf_history():
    """Return [(trading_date, net_flow_USD_millions), ...] oldest first."""
    try:
        response = requests.get(
            ETF_BTC_FEED,
            headers={"User-Agent": "CryptoDashboard/1.0", "Accept": "application/json"},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("days"), list):
            return []

        records = {}
        for row in payload["days"]:
            if not isinstance(row, dict):
                continue
            date = str(row.get("date", ""))[:10]
            value = row.get("netFlowUsd")
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date) or value is None:
                continue
            try:
                flow_usd = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(flow_usd):
                continue
            # The provider marks a zero row without a fund breakdown as a holiday.
            if flow_usd == 0 and not row.get("perEtfUsd"):
                continue
            records[date] = flow_usd / 1_000_000.0
        return sorted(records.items())
    except (requests.RequestException, ValueError, TypeError):
        return []


ETF_ETH_FEED = "https://growyourethereum.com/datasets/eth-etf-flows.json"
ETF_ETH_SOURCE = "https://growyourethereum.com/data/ethereum-etf-flows"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ethereum_etf_history():
    """Return dated US spot ETH ETF daily flow totals in USD millions, oldest first."""
    try:
        response = requests.get(
            ETF_ETH_FEED,
            headers={"User-Agent": "CryptoDashboard/1.0", "Accept": "application/json"},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("daily"), list):
            return []

        records = {}
        for row in payload["daily"]:
            if not isinstance(row, dict):
                continue
            date = str(row.get("date", ""))[:10]
            value = row.get("total")  # Feed already uses USD millions.
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date) or value is None:
                continue
            try:
                flow_usdm = float(value)
            except (ValueError, TypeError):
                continue
            if np.isfinite(flow_usdm):
                records[date] = flow_usdm
        return sorted(records.items())
    except (requests.RequestException, ValueError, TypeError):
        return []


# -----------------------------
# Derivatives-market context — Coinalyze authenticated read-only data.
# The API key is read ONLY from Streamlit Secrets, never placed in the repo.
# One supported perpetual contract is selected using provider metadata.
# No guessed liquidation heatmap values and no trading activity.
# -----------------------------
def _market_float(value):
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _coinalyze_time(seconds):
    """Format provider timestamps safely; absent/invalid values must never crash the app.

    Coinalyze may return no `update` field on some current-data endpoints.
    Date-only strings and Unix seconds/milliseconds are accepted, but anything
    implausible is treated as an unknown reporting timestamp.
    """
    if seconds is None or isinstance(seconds, bool):
        return None
    try:
        if isinstance(seconds, str) and not seconds.strip():
            return None
        timestamp = float(seconds)
        if not np.isfinite(timestamp):
            return None
        if timestamp > 1e11:  # Unix milliseconds, if a provider uses them
            timestamp /= 1000.0
        parsed = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        if not (2000 <= parsed.year <= 2100):
            return None
        return parsed.strftime("%Y-%m-%d %H:%M UTC")
    except (TypeError, ValueError, OverflowError, OSError, NotImplementedError):
        return None


def _coinalyze_key():
    try:
        return str(st.secrets.get("COINALYZE_API_KEY", "")).strip()
    except Exception:
        return ""


class CoinalyzeDataError(Exception):
    pass


def _coinalyze_get(endpoint, key, params=None):
    try:
        response = requests.get(
            "https://api.coinalyze.net/v1/" + endpoint,
            params=params or {},
            headers={"api_key": key, "Accept": "application/json"},
            timeout=12,
        )
        if response.status_code == 401:
            raise CoinalyzeDataError("API key missing or rejected (HTTP 401)")
        if response.status_code == 429:
            raise CoinalyzeDataError("API rate limit reached (HTTP 429)")
        if response.status_code != 200:
            raise CoinalyzeDataError(f"{endpoint}: HTTP {response.status_code}")
        payload = response.json()
        if not isinstance(payload, list):
            raise CoinalyzeDataError(f"{endpoint}: unexpected response format")
        return payload
    except requests.RequestException as exc:
        raise CoinalyzeDataError(f"{endpoint}: network unavailable ({type(exc).__name__})") from exc
    except (ValueError, TypeError) as exc:
        raise CoinalyzeDataError(f"{endpoint}: invalid JSON response") from exc


@st.cache_data(ttl=21600, show_spinner=False)
def _coinalyze_market(ticker, key):
    markets = _coinalyze_get("future-markets", key)
    options = [m for m in markets if isinstance(m, dict)
               and m.get("base_asset", "").upper() == ticker
               and m.get("quote_asset", "").upper() == "USDT"
               and m.get("is_perpetual") is True
               and m.get("has_long_short_ratio_data") is True
               and m.get("symbol")]
    if not options:
        raise CoinalyzeDataError(f"No {ticker}/USDT perpetual with long/short data listed")
    # Prefer a known large exchange; never assume suffix letters identify venues.
    def sort_key(m):
        ex = str(m.get("exchange", "")).lower()
        priority = 0 if "binance" in ex else 1 if "bybit" in ex else 2
        return (priority, str(m.get("symbol")))
    return sorted(options, key=sort_key)[0]


def _coinalyze_latest_history(entries):
    if not isinstance(entries, list) or not entries:
        return []
    first = entries[0]
    if not isinstance(first, dict) or not isinstance(first.get("history"), list):
        return []
    return sorted((row for row in first["history"] if isinstance(row, dict)),
                  key=lambda row: row.get("t", 0))


@st.cache_data(ttl=300, show_spinner=False)
def fetch_coinalyze_context(ticker, key):
    result = {"exchange": "Coinalyze", "contract": None, "funding_pct": None,
              "long_pct": None, "short_pct": None, "oi_usd": None,
              "oi_change_pct": None, "oi_period": "1h", "asof": None,
              "ratio_asof": None, "oi_asof": None, "available": 0,
              "notes": [], "fatal": None}
    try:
        market = _coinalyze_market(ticker, key)
    except CoinalyzeDataError as exc:
        result["fatal"] = str(exc)
        return result
    symbol = str(market["symbol"])
    result["contract"] = symbol
    result["exchange"] = str(market.get("exchange") or "Exchange")
    result["oi_basis"] = str(market.get("oi_lq_vol_denominated_in") or "UNKNOWN")
    now = int(pd.Timestamp.now(tz="UTC").timestamp())
    common = {"symbols": symbol}

    try:
        rows = _coinalyze_get("funding-rate", key, common)
        if rows:
            value = _market_float(rows[0].get("value"))
            if value is not None:
                # Coinalyze presents funding rates as percentages, not fractions.
                result["funding_pct"] = value
                result["available"] += 1
            result["asof"] = _coinalyze_time(rows[0].get("update"))
    except CoinalyzeDataError as exc:
        result["notes"].append("Funding: " + str(exc))

    try:
        rows = _coinalyze_get("open-interest", key,
                              {**common, "convert_to_usd": "true"})
        if rows:
            value = _market_float(rows[0].get("value"))
            if value is not None:
                result["oi_usd"] = value
                result["available"] += 1
            result["oi_asof"] = _coinalyze_time(rows[0].get("update"))
    except CoinalyzeDataError as exc:
        result["notes"].append("Open interest: " + str(exc))

    history_params = {**common, "interval": "1hour", "from": now - 6 * 3600, "to": now}
    try:
        entries = _coinalyze_get("open-interest-history", key,
                                 {**history_params, "convert_to_usd": "true"})
        history = _coinalyze_latest_history(entries)
        if len(history) >= 2:
            current = _market_float(history[-1].get("c"))
            previous = _market_float(history[-2].get("c"))
            if current is not None and previous is not None and previous > 0:
                result["oi_change_pct"] = (current / previous - 1) * 100.0
    except CoinalyzeDataError as exc:
        result["notes"].append("OI change: " + str(exc))

    try:
        rows = _coinalyze_get("long-short-ratio-history", key, history_params)
        history = _coinalyze_latest_history(rows)
        if history:
            row = history[-1]
            longs, shorts = _market_float(row.get("l")), _market_float(row.get("s"))
            if longs is not None and shorts is not None and longs >= 0 and shorts >= 0 and (longs+shorts)>0:
                result["long_pct"] = 100 * longs / (longs+shorts)
                result["short_pct"] = 100 * shorts / (longs+shorts)
            else:
                ratio = _market_float(row.get("r"))
                if ratio is not None and ratio >= 0:
                    result["long_pct"] = 100 * ratio / (1+ratio)
                    result["short_pct"] = 100 / (1+ratio)
            if result["long_pct"] is not None:
                result["available"] += 1
                result["ratio_asof"] = _coinalyze_time(row.get("t"))
    except CoinalyzeDataError as exc:
        result["notes"].append("Long/short ratio: " + str(exc))

    return result



# -----------------------------
# Independent, public model-estimated liquidation clusters.
# No exchange publishes centralized users' true liquidation levels.
# Do not mix these estimates with Coinalyze contract-specific OI.
# -----------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_modeled_liquidation_zones(ticker):
    result = {"valid": False, "reason": "Unavailable", "price": None,
              "long": None, "short": None, "asof": None, "version": None}
    if ticker not in ("BTC", "ETH"):
        return result
    try:
        response = requests.get(
            f"https://swolecharts.com/api/liquidation-levels/{ticker}",
            params={"range": "24h"},
            headers={"Accept": "application/json", "User-Agent": "CryptoDashboard/1.0"},
            timeout=12,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict) or data.get("estimated") is not True or data.get("has_data") is not True:
            result["reason"] = "Provider has no usable model snapshot."
            return result
        if str(data.get("coin", "")).upper() != ticker:
            result["reason"] = "Provider coin mismatch."
            return result
        reference = _market_float(data.get("price"))
        stamp = _market_float(data.get("as_of"))
        if reference is None or reference <= 0 or stamp is None:
            result["reason"] = "Missing price or timestamp."
            return result
        asof = datetime.fromtimestamp(stamp / 1000 if stamp > 1e11 else stamp, tz=timezone.utc)
        age = (datetime.now(timezone.utc) - asof).total_seconds()
        if age < -300 or age > 2 * 3600:
            result["reason"] = "Snapshot is older than 2 hours or its timestamp is invalid."
            return result
        levels = data.get("levels", {})
        if not isinstance(levels, dict):
            result["reason"] = "No levels in provider response."
            return result
        for side in ("long", "short"):
            candidates = []
            for item in levels.get(side, []):
                if not isinstance(item, dict):
                    continue
                mid = _market_float(item.get("mid"))
                usd = _market_float(item.get("usd"))
                if mid is None or usd is None or usd <= 0:
                    continue
                if (side == "long" and mid >= reference) or (side == "short" and mid <= reference):
                    continue
                # Select the most intense reported zone, not the nearest by price.
                candidates.append({"price": mid, "usd": usd,
                                   "distance_pct": 100 * (mid / reference - 1)})
            if candidates:
                result[side] = max(candidates, key=lambda x: x["usd"])
        result.update({"valid": bool(result["long"] or result["short"]),
                       "price": reference, "asof": asof.strftime("%Y-%m-%d %H:%M UTC"),
                       "version": str(data.get("model_version") or "unknown")[:70],
                       "reason": "No qualifying zones in the current estimate."})
        return result
    except (requests.RequestException, ValueError, TypeError, OverflowError, OSError) as exc:
        result["reason"] = f"Liquidation data request unavailable ({type(exc).__name__})."
        return result


def _compact_usd(value):
    if value is None:
        return "N/A"
    if abs(value) >= 1e9:
        return f"${value / 1e9:,.2f}B"
    if abs(value) >= 1e6:
        return f"${value / 1e6:,.1f}M"
    return f"${value:,.0f}"

# -----------------------------
# One-glance MARKET BIAS (24h heuristic, NOT a calibrated probability).
# Price momentum, modeled liquidation concentrations, account ratios,
# funding, and recent ETF flows are heterogeneous signals. A score measures
# their weighted directional lean; it does not forecast a success rate.
# -----------------------------
from directional_score_core import compute_directional_outlook


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("⚙️ Settings")
symbol = st.sidebar.selectbox("Asset", ["BTC-USD", "ETH-USD"], index=0)

# Asset labels
ASSET_TICKER = symbol.replace("-USD", "")
ASSET_NAME = "Bitcoin" if symbol == "BTC-USD" else ("Ethereum" if symbol == "ETH-USD" else ASSET_TICKER)
ETF_ASSET_LABEL = "Bitcoin" if symbol == "BTC-USD" else ("Ethereum" if symbol == "ETH-USD" else None)

period = st.sidebar.selectbox(
    "Time Range",
    ["1d", "7d", "1mo", "1y", "2y", "5y", "max"],
    index=3
)

predict_steps = st.sidebar.slider(
    "Prediction Steps (future candles)",
    min_value=10,
    max_value=200,
    value=60,
    step=5
)

# -----------------------------
# Sidebar: Display Controls
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🧩 Display Controls")

indicator_options = [
    "SMA Short",
    "SMA Medium",
    "SMA Long",
    "VWAP",
    "Bollinger Bands",
    "Support/Resistance",
]

show_indicators = st.sidebar.multiselect(
    "Indicators on price chart",
    indicator_options,
    default=indicator_options,
)

panel_options = ["Leverage & Liquidations", "Volume", "RSI", "OBV", "MACD", "AI Forecast"]
show_panels = st.sidebar.multiselect(
    "Panels to show",
    panel_options,
    default=panel_options,
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

st.sidebar.markdown("---")
st.sidebar.subheader(f"🏦 {ASSET_TICKER} ETF Flows")

# Free daily BTC ETF history, presented compactly in USD millions.
# Keep `flow` for the existing AI Action Hint calculations.
flow = None
if ASSET_TICKER == "BTC":
    etf_rows = fetch_bitcoin_etf_history()
    if etf_rows:
        latest_date, flow = etf_rows[-1]
        direction = "🔴 Outflow" if flow < 0 else ("🟢 Inflow" if flow > 0 else "🟡 Flat")
        st.sidebar.markdown(f"**Latest:** {flow:+,.1f}M · {direction}")
        st.sidebar.caption(f"As of {latest_date} · USD millions")

        if len(etf_rows) >= 7:
            current_sum = float(sum(value for _, value in etf_rows[-7:]))
            st.sidebar.markdown(f"**Last 7 sessions:** {current_sum:+,.1f}M")
            if len(etf_rows) >= 14:
                previous_sum = float(sum(value for _, value in etf_rows[-14:-7]))
                change = current_sum - previous_sum
                if abs(change) < 50:
                    trend = "🟡 Flat"
                elif change > 0:
                    trend = "🟢 Improving"
                else:
                    trend = "🔴 Weakening"
                st.sidebar.markdown(f"**Trend:** {trend} ({change:+,.1f}M)")
                st.sidebar.caption("Change vs previous 7 sessions")
        st.sidebar.caption("[Source: TFTC](https://www.tftc.io/bitcoin-etf-flows) · Daily, not live")
    else:
        st.sidebar.info("BTC ETF flows temporarily unavailable.")
        st.sidebar.caption("[Source: TFTC](https://www.tftc.io/bitcoin-etf-flows)")
else:
    # Same compact presentation as BTC, using ETH-specific dated daily flows.
    etf_rows = fetch_ethereum_etf_history()
    if etf_rows:
        latest_date, latest_flow = etf_rows[-1]
        direction = "🔴 Outflow" if latest_flow < 0 else ("🟢 Inflow" if latest_flow > 0 else "🟡 Flat")
        st.sidebar.markdown(f"**Latest:** {latest_flow:+,.1f}M · {direction}")
        st.sidebar.caption(f"As of {latest_date} · USD millions")

        if len(etf_rows) >= 7:
            current_sum = float(sum(value for _, value in etf_rows[-7:]))
            st.sidebar.markdown(f"**Last 7 sessions:** {current_sum:+,.1f}M")
            if len(etf_rows) >= 14:
                previous_sum = float(sum(value for _, value in etf_rows[-14:-7]))
                change = current_sum - previous_sum
                if abs(change) < 50:
                    trend = "🟡 Flat"
                elif change > 0:
                    trend = "🟢 Improving"
                else:
                    trend = "🔴 Weakening"
                st.sidebar.markdown(f"**Trend:** {trend} ({change:+,.1f}M)")
                st.sidebar.caption("Change vs previous 7 sessions")

        # ETH ETF information is secondary; only recent data should affect action hints.
        age_days = (pd.Timestamp.now(tz="UTC").date() - pd.Timestamp(latest_date).date()).days
        if age_days < 0 or age_days > 4:
            flow = None  # Display historical reading, but do not use a stale reading in action hints.
            st.sidebar.caption("⚠️ Older report · excluded from action hint")
        else:
            flow = latest_flow
        st.sidebar.caption("[Source: GrowYourEthereum](https://growyourethereum.com/data/ethereum-etf-flows) · Farside · Daily, not live")
    else:
        st.sidebar.info("ETH ETF flows temporarily unavailable.")
        st.sidebar.caption("[Source: GrowYourEthereum](https://growyourethereum.com/data/ethereum-etf-flows)")


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
# Header + Top Metrics (includes nearest S/R)
# -----------------------------
st.title(f"📊 {ASSET_NAME} Live Dashboard")
st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

metric_cols = st.columns(9)
col1, col2, col3, col4, col5, col6, col7, col8, col9 = metric_cols

regime = "Unknown"
dist = None

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

    nearest_support, nearest_resistance = nearest_sr_from_lists(price, supports, resistances)

    if nearest_support is not None:
        support_pct = (nearest_support - price) / price * 100.0
        col8.metric("Nearest Support", f"${nearest_support:,.0f}", f"{support_pct:.2f}%")
    else:
        col8.metric("Nearest Support", "N/A")

    if nearest_resistance is not None:
        resistance_pct = (nearest_resistance - price) / price * 100.0
        col9.metric("Nearest Resistance", f"${nearest_resistance:,.0f}", f"{resistance_pct:+.2f}%")
    else:
        col9.metric("Nearest Resistance", "N/A")
else:
    labels = [
        f"{ASSET_TICKER} Price", "SMA Short", "SMA Medium", "SMA Long",
        "Market Regime", "Dist vs SMA Long", "Strength Score",
        "Nearest Support", "Nearest Resistance",
    ]
    for column, label in zip(metric_cols, labels):
        column.metric(label, "Loading...")

# -----------------------------
# Price + Overlays (with indicator toggles)
# -----------------------------
st.subheader("📈 Price + Indicators")

if has_data(btc, ["Close"]):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(btc.index, btc["Close"], label="Price")

    if "SMA Short" in show_indicators and "SMA_short" in btc.columns:
        ax.plot(btc.index, btc["SMA_short"], label="SMA Short")
    if "SMA Medium" in show_indicators and "SMA_medium" in btc.columns:
        ax.plot(btc.index, btc["SMA_medium"], label="SMA Medium")
    if "SMA Long" in show_indicators and "SMA_long" in btc.columns:
        ax.plot(btc.index, btc["SMA_long"], label="SMA Long", linewidth=2.0)
    if "VWAP" in show_indicators and "VWAP" in btc.columns:
        ax.plot(btc.index, btc["VWAP"], label="VWAP", linestyle="--")
    if "Bollinger Bands" in show_indicators:
        if "BB_upper" in btc.columns:
            ax.plot(btc.index, btc["BB_upper"], label="BB Upper", linestyle="--", alpha=0.5)
        if "BB_lower" in btc.columns:
            ax.plot(btc.index, btc["BB_lower"], label="BB Lower", linestyle="--", alpha=0.5)

    if "Support/Resistance" in show_indicators:
        price_now = float(btc["Close"].iloc[-1])
        nearest_support, nearest_resistance = nearest_sr_from_lists(price_now, supports, resistances)

        support_below = sorted([s for s in supports if np.isfinite(s) and s < price_now])[-3:]
        if not support_below and nearest_support is not None:
            support_below = [nearest_support]

        for lvl in support_below:
            ax.axhline(lvl, linestyle="--", linewidth=1.3, alpha=0.85, color="green")
            ax.text(
                btc.index[-1], lvl, f"S: {lvl:,.0f}",
                va="center", ha="left", fontsize=9, color="green",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2),
            )

        if nearest_resistance is not None:
            ax.axhline(nearest_resistance, linestyle="--", linewidth=2.0, alpha=0.95, color="red")
            ax.text(
                btc.index[-1], nearest_resistance, f"R: {nearest_resistance:,.0f}",
                va="center", ha="left", fontsize=9, color="red",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2),
            )

    ax.set_ylabel("USD")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
else:
    st.info("Not enough data to plot price.")

# -----------------------------
# Leverage & Liquidation Analysis — compact, read-only, no trading.
# Coinalyze metrics refer to ONE exchange perpetual; liquidation zones
# are independent modeled estimates of positions from multiple venues.
# -----------------------------
if "Leverage & Liquidations" in show_panels:
    st.subheader(f"⚖️ {ASSET_TICKER} Leverage & Liquidation Analysis")
    with st.expander("📊 Market pressure · Liquidation zones", expanded=True):
        api_key = _coinalyze_key()
        if not api_key:
            st.info("Add COINALYZE_API_KEY to your development Streamlit Secrets to show live futures metrics.")
            st.markdown("[Get your free Coinalyze key](https://coinalyze.net/account/api-key/)")
        else:
            ctx = fetch_coinalyze_context(ASSET_TICKER, api_key)
            if not ctx["available"]:
                st.warning("Futures metrics are currently unavailable.")
                if ctx.get("fatal"):
                    with st.expander("Connection diagnostics"):
                        st.caption(ctx["fatal"])
            else:
                funding = ctx["funding_pct"]
                change = ctx["oi_change_pct"]
                longs = ctx["long_pct"]
                shorts = ctx["short_pct"]
                a, b, c = st.columns(3)
                with a:
                    st.metric("Funding", f"{funding:+.4f}%" if funding is not None else "N/A")
                    if funding is not None:
                        st.caption("Longs pay shorts" if funding > 0 else
                                   "Shorts pay longs" if funding < 0 else "Neutral")
                with b:
                    st.metric("Open interest", _compact_usd(ctx["oi_usd"]),
                              delta=f"{change:+.2f}% / 1h" if change is not None else None)
                with c:
                    if longs is not None and shorts is not None:
                        st.metric("Long / Short accounts", f"{longs:.1f}% / {shorts:.1f}%")
                        st.progress(min(max(longs / 100, 0), 1))
                    else:
                        st.metric("Long / Short accounts", "N/A")
                if funding is not None and longs is not None:
                    if funding > 0 and longs >= 60:
                        pressure = "Long-heavy positioning with positive funding — watch for downside squeeze risk."
                    elif funding < 0 and shorts is not None and shorts >= 60:
                        pressure = "Short-heavy positioning with negative funding — watch for upside squeeze risk."
                    else:
                        pressure = "Mixed leverage signals — no clear directional edge."
                    st.caption("**Market pressure:** " + pressure)
                st.caption(f"[Coinalyze](https://coinalyze.net/) · {ctx['exchange']} · "
                           f"{ctx['contract']} · cached 5 min")
            if ctx.get("notes"):
                with st.expander("Connection diagnostics", expanded=False):
                    for note in ctx["notes"]:
                        st.caption(note)

        # Compact 24-hour directional heuristic. Never describe this as
        # a probability or a verified trading edge. Uses fresh sources only.
        daily_for_outlook = load_data(symbol, "1mo")
        daily_closes = (daily_for_outlook["Close"]
                        if has_data(daily_for_outlook, ["Close"]) else [])
        # Ensure ctx exists if Coinalyze is not configured.
        context_for_outlook = ctx if api_key else {}
        # Zones are retrieved once below, but cached across reruns.
        zones = fetch_modeled_liquidation_zones(ASSET_TICKER)
        outlook = compute_directional_outlook(
            context_for_outlook, zones, daily_closes,
            etf_rows if "etf_rows" in locals() else [],
        )
        st.markdown("**🧭 24-hour directional outlook**")
        if outlook["ready"]:
            left, right = st.columns([2, 1])
            with left:
                if outlook["direction"] == "Bullish lean":
                    st.markdown(f"### 🟢 {outlook['icon']} {outlook['direction']}")
                elif outlook["direction"] == "Bearish lean":
                    st.markdown(f"### 🔴 {outlook['icon']} {outlook['direction']}")
                else:
                    st.markdown(f"### 🟡 {outlook['icon']} {outlook['direction']}")
                st.caption("Rule-based reading of market pressure; not a trade recommendation.")
            with right:
                st.metric("Upside directional score", f"{outlook['up_score']}/100")
                st.caption(f"Downside: {outlook['down_score']}/100 · "
                           "NOT % chance of a move")
            st.progress(outlook["up_score"] / 100)
            st.caption(f"Data coverage: {outlook['coverage']:.0%} of configured signal weights"
                       " · Some inputs update less frequently than others.")
        else:
            st.info(outlook["reason"])
            st.caption(f"Data coverage: {outlook['coverage']:.0%} · "
                       "No directional score shown when too much information is missing.")
        with st.expander("🔎 Why this direction?", expanded=False):
            for component in outlook["components"]:
                lean = ("up" if component["signal"] > .05 else
                        "down" if component["signal"] < -.05 else "neutral")
                st.markdown(f"**{component['name']}** — {lean}: {component['detail']}")
            if context_for_outlook.get("oi_change_pct") is not None:
                st.caption("Open interest changes the strength of other signals; "
                           "it does not independently select a direction.")
            st.caption("Fixed heuristic weights: price 35%, modeled liquidation zones 30%, "
                       "ETF flows 15%, funding 10%, account positioning 10%. "
                       "Missing signals reduce coverage. This score is not calibrated "
                       "or validated on future returns.")
        st.divider()

        st.markdown("**Estimated liquidation zones (24h model)**")
        if zones["valid"]:
            z1, z2 = st.columns(2)
            with z1:
                level = zones["long"]
                if level:
                    st.metric("▼ Longs at risk below", f"${level['price']:,.0f}",
                              delta=f"{level['distance_pct']:+.2f}% vs model price", delta_color="off")
                    st.caption(f"Estimated cluster: {_compact_usd(level['usd'])}")
                else:
                    st.metric("▼ Longs at risk below", "N/A")
            with z2:
                level = zones["short"]
                if level:
                    st.metric("▲ Shorts at risk above", f"${level['price']:,.0f}",
                              delta=f"{level['distance_pct']:+.2f}% vs model price", delta_color="off")
                    st.caption(f"Estimated cluster: {_compact_usd(level['usd'])}")
                else:
                    st.metric("▲ Shorts at risk above", "N/A")
            st.caption(f"[Swolecharts modeled liquidation levels]"
                       f"(https://swolecharts.com/liquidation-map/{ASSET_TICKER})"
                       f" · snapshot {zones['asof']} · reference ${zones['price']:,.0f}"
                       f" · model {zones['version']}")
        else:
            st.caption("Live modeled zones unavailable: " + zones["reason"])
            st.markdown(f"[View the free {ASSET_TICKER} liquidation map]"
                        f"(https://swolecharts.com/liquidation-map/{ASSET_TICKER})")
        st.caption("Clusters are estimates, not actual orders or predicted price targets. "
                   "Different providers cover different markets.")

        with st.expander("📖 How to read these indicators", expanded=False):
            st.markdown("**Funding:** A positive rate means longs pay shorts; a negative rate "
                        "means shorts pay longs. It reflects the perpetual futures market, "
                        "not the probability of an up/down move.")
            st.markdown("**Open interest:** Total outstanding futures exposure. The hourly "
                        "change shows whether exposure is growing or shrinking, not which side wins.")
            st.markdown("**Long/short accounts:** Fraction of tracked accounts net long or short. "
                        "This is not the proportion of money on each side.")
            st.markdown("**Liquidation zones:** Modeled concentrations below price represent "
                        "longs at risk if price falls; concentrations above represent shorts "
                        "at risk if price rises. These are estimates and change over time.")
            st.caption("Coinalyze measures a single exchange contract. Swolecharts estimates "
                       "multiple exchanges using different data and assumptions. These "
                       "numbers cannot be summed or treated as a calibrated forecast.")
        st.caption("Read-only analysis. No trades or changes to scheduled forecasting.")

# -----------------------------
# Volume
# -----------------------------
if "Volume" in show_panels:
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
if "RSI" in show_panels:
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
if "OBV" in show_panels:
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
if "MACD" in show_panels:
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
if "AI Forecast" in show_panels:
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
            c2.metric("Heuristic confidence", f"{conf_emoji} {conf_label}")
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
            st.caption("Live confidence is a heuristic score, not a validated probability. "
                       "The live forecast above uses different models from the historical "
                       "backtest below; backtest scores do not validate these live signals. "
                       "Educational only — not financial advice.")

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

        # Phase 2: feature-based daily return forecasting and walk-forward evaluation.
        # Independent of the selected chart timeframe and existing experimental live forecast.
        with st.expander("🧪 AI Forecast Performance & Backtesting — Phase 3", expanded=False):
            st.caption(
                "Daily BTC/ETH historical simulations (1, 7, 30, 60 days). "
                "At each forecast date, models train only on outcomes already known. "
                "Returns are predicted from lagged price, volatility, momentum and volume features. "
                "These are NOT archived live predictions or calibrated probabilities. "
                "The momentum benchmark projects the preceding 7-day trend at a daily compounded rate. "
                "The live forecast above uses separate models and is not validated by these tests."
            )
            from forecast_backtest import run_backtest

            @st.cache_data(ttl=3600, show_spinner=False)
            def get_daily_backtest_data(asset):
                return load_data(asset, "5y")

            if st.button("Run Phase 3 historical backtest", key=f"backtest_{symbol}"):
                with st.spinner("Testing historical BTC/ETH forecasts..."):
                    daily_data = get_daily_backtest_data(symbol)
                    if daily_data.empty:
                        st.error("Could not download daily history. Try again later.")
                    else:
                        scores, predictions = run_backtest(daily_data)
                        if scores.empty:
                            st.warning("Not enough daily history for this backtest.")
                        else:
                            st.session_state[f"backtest_scores_v3_{symbol}"] = scores
                            st.session_state[f"backtest_preds_v3_{symbol}"] = predictions

            scores = st.session_state.get(f"backtest_scores_v3_{symbol}")
            predictions = st.session_state.get(f"backtest_preds_v3_{symbol}")
            if scores is not None and not scores.empty and predictions is not None:
                # Display-only improvements: no model refitting when switching charts or horizons.
                def style_backtest_table(df):
                    def error_style(value):
                        try:
                            number = float(str(value).split()[0])
                        except (ValueError, TypeError, IndexError):
                            return ""
                        if number < -1e-9:
                            return "color: #138a36; font-weight: bold"
                        if number > 1e-9:
                            return "color: #d94841; font-weight: bold"
                        return ""

                    def skill_style(value):
                        try:
                            number = float(str(value).replace('%', ''))
                        except (ValueError, TypeError):
                            return ""
                        if number > 1e-9:
                            return "color: #138a36; font-weight: bold"
                        if number < -1e-9:
                            return "color: #d94841; font-weight: bold"
                        return ""

                    return (df.style
                            .map(error_style, subset=['Error vs baseline'])
                            .map(skill_style, subset=['Model Skill']))

                st.dataframe(style_backtest_table(scores), use_container_width=True, hide_index=True)
                st.caption("**Model Skill:** percentage reduction in mean price error versus no-change "
                           "(positive = better; negative = worse). **Error vs baseline:** difference "
                           "in mean price error in percentage points (negative = better).")
                horizon_counts = scores.groupby('Horizon')['Tests'].min().to_dict()
                low_count = [f"{h}: {int(n)} tests" for h, n in horizon_counts.items() if int(n) < 30]
                st.caption("Sample sizes: " + "; ".join(f"{h}: {int(n)} tests" for h, n in horizon_counts.items()))
                if low_count:
                    st.warning("Small historical sample: " + "; ".join(low_count) +
                               ". Interpret directional accuracy cautiously.")
                st.caption(
                    "Lower absolute price/return error is better. A negative error vs baseline is an improvement. "
                    "Direction accuracy measures "
                    "up/down predictions; the no-change baseline makes no directional call, "
                    "while 7-day momentum supplies a simple directional benchmark. "
                    "Targets are spaced by horizon to avoid overlapping evaluation periods."
                )
                # Phase 3: diagnostics reuse completed predictions; no extra model fitting.
                # Regimes depend only on the trailing 30-day return at forecast time.
                with st.expander("🔎 Phase 3: When do predictions work?", expanded=False):
                    st.caption(
                        "**Regime at prediction date:** Bull = previous 30-day return above +10%; "
                        "Bear = below −10%; otherwise Sideways / transition. "
                        "These thresholds are simple historical labels, not a forecast of the next regime."
                    )
                    diag_h = st.selectbox("Diagnostic forecast horizon", [1, 7, 30, 60],
                                          index=1, key=f"regime_horizon_{symbol}")
                    dsub = predictions[predictions['Horizon'] == diag_h].copy()
                    baseline_d = (dsub[dsub['Model'] == 'No-change baseline']
                                  .set_index('Forecast date')['Absolute error %'])
                    dsub['Baseline error %'] = dsub['Forecast date'].map(baseline_d)
                    regime_rows = []
                    for (regime_name, model_name), group in dsub.groupby(['Market regime', 'Model'], sort=True):
                        n = len(group)
                        mae = float(group['Absolute error %'].mean())
                        base_mae = float(group['Baseline error %'].mean())
                        skill = 100 * (1 - mae / base_mae) if base_mae > 0 else float('nan')
                        directional = group['Direction correct'].dropna()
                        regime_rows.append({
                            'Market regime': regime_name, 'Model': model_name, 'Tests': n,
                            'Price error': f'{mae:.2f}%',
                            'Model Skill': f'{skill:+.1f}%' if np.isfinite(skill) else 'N/A',
                            'Direction correct': (
                                f'{100 * directional.astype(float).mean():.1f}%'
                                if len(directional) else 'N/A'),
                        })
                    regime_df = pd.DataFrame(regime_rows)
                    if not regime_df.empty:
                        st.dataframe(regime_df, use_container_width=True, hide_index=True)
                        if (regime_df['Tests'] < 15).any():
                            st.warning('Some regimes have fewer than 15 forecasts. Their results are highly uncertain.')

                    st.markdown('**Historical stability: earlier tests vs more recent tests**')
                    st.caption(
                        "Splits the already-completed forecasts into two chronological halves. "
                        "This helps reveal whether a result depends on one market period."
                    )
                    stability = []
                    for model_name, group in dsub.groupby('Model', sort=True):
                        group = group.sort_values('Forecast date')
                        midpoint = len(group) // 2
                        for period_name, part in [('Earlier', group.iloc[:midpoint]),
                                                  ('Recent', group.iloc[midpoint:])]:
                            if part.empty:
                                continue
                            model_err = float(part['Absolute error %'].mean())
                            base_err = float(part['Baseline error %'].mean())
                            model_skill = (100 * (1 - model_err / base_err)
                                           if base_err > 0 else float('nan'))
                            stability.append({
                                'Model': model_name, 'Period': period_name,
                                'Tests': len(part), 'Price error': f'{model_err:.2f}%',
                                'Model Skill': (f'{model_skill:+.1f}%'
                                                if np.isfinite(model_skill) else 'N/A'),
                            })
                    if stability:
                        st.dataframe(pd.DataFrame(stability), use_container_width=True,
                                     hide_index=True)
                    st.caption(
                        "**Historical error sizes (not a future confidence interval):** "
                        "For each model, the table below gives the median and 80th-percentile "
                        "absolute return error across these completed tests. "
                        "Past performance does not guarantee future coverage."
                    )
                    error_rows = []
                    for model_name, group in dsub.groupby('Model', sort=True):
                        if model_name == 'No-change baseline':
                            continue
                        errors = group['Return error (pp)'].dropna()
                        if errors.empty:
                            continue
                        error_rows.append({
                            'Model': model_name, 'Tests': len(errors),
                            'Median absolute return error': f'{errors.quantile(0.5):.2f} pp',
                            '80th-percentile absolute error': f'{errors.quantile(0.8):.2f} pp',
                        })
                    if error_rows:
                        st.dataframe(pd.DataFrame(error_rows), use_container_width=True,
                                     hide_index=True)
                    st.caption('No new regressors are trained to generate these diagnostics.')

                chosen_h = st.selectbox("Inspect forecast horizon (days)", [1, 7, 30, 60],
                                        key=f"backtest_horizon_{symbol}")
                subset = predictions[predictions['Horizon'] == chosen_h].copy()
                if not subset.empty:
                    view = st.radio("Backtest chart", ["Predicted vs actual returns (%)",
                                                       "Predicted vs actual prices ($)"],
                                    horizontal=True, key=f"backtest_view_{symbol}")
                    if view == "Predicted vs actual returns (%)":
                        chart = subset.pivot(index='Target date', columns='Model',
                                             values='Predicted return %')
                        actual = subset.drop_duplicates('Target date').set_index('Target date')
                        chart = chart.join(actual[['Actual return %']]).sort_index()
                        st.line_chart(chart, use_container_width=True)
                        st.caption("Model returns are compared with actual BTC/ETH returns at "
                                   "the same target dates. A zero return is the no-change baseline.")
                    else:
                        chart = subset.pivot(index='Target date', columns='Model',
                                             values='Predicted price')
                        actual = subset.drop_duplicates('Target date').set_index('Target date')
                        chart = chart.join(actual[['Actual price']]).sort_index()
                        st.line_chart(chart, use_container_width=True)
                    st.download_button("Download Phase 3 predictions (CSV)",
                                       predictions.to_csv(index=False).encode('utf-8'),
                                       file_name=f"{ASSET_TICKER.lower()}_phase3_backtest.csv",
                                       mime='text/csv', key=f"backtest_download_{symbol}")
            else:
                st.info("Click Run Phase 3 historical backtest to evaluate the models.")


    else:
        st.info("Not enough data for AI predictions yet (need ~25+ data points).")


