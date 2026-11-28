"""Public market feeds and Coinalyze reads for the directional score recorder.
No trading endpoints. The API key is injected from GitHub Actions secrets.
Code follows the same source calls as the Streamlit dashboard.
"""
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
import re

ETF_BTC_FEED = "https://www.tftc.io/bitcoin-etf-flows/data.json"
class CoinalyzeDataError(Exception):
    pass

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

