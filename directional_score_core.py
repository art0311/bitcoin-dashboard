"""Shared 24-hour rule-based directional score. NOT a calibrated probability.
Exactly the same scoring function is called from Streamlit and the recorder.
"""
import numpy as np
import pandas as pd

def compute_directional_outlook(ctx, zones, daily_closes, etf_history, oi_scale=True):
    # Signed signal values are constrained to [-1, +1]. Zero is NOT used
    # for missing data. That distinction keeps coverage honest.
    components = []

    def add(name, weight, signal, detail):
        if signal is not None and np.isfinite(signal):
            components.append({"name": name, "weight": weight,
                               "signal": float(np.clip(signal, -1, 1)),
                               "detail": detail})

    # Latest daily data from Yahoo (including its potentially forming candle).
    series = pd.Series(daily_closes).dropna().astype(float)
    if len(series) >= 8 and series.iloc[-8] > 0 and series.iloc[-2] > 0:
        p = float(series.iloc[-1])
        d1 = p / float(series.iloc[-2]) - 1
        d7 = p / float(series.iloc[-8]) - 1
        # Scale: a 2% 1-day or 7% 7-day move saturates each subscore.
        momentum = .6 * np.clip(d1 / .02, -1, 1) + .4 * np.clip(d7 / .07, -1, 1)
        add("Price momentum", .35, momentum, f"1-day {d1:+.2%} · 7-day {d7:+.2%}")

    if zones.get("valid") and zones.get("long") and zones.get("short"):
        below, above = zones["long"], zones["short"]
        db, da = abs(below["distance_pct"]), abs(above["distance_pct"])
        if 0.15 <= db <= 25 and 0.15 <= da <= 25:
            # Proximity-weighted *estimated* cluster pressure, capped so one
            # far-away band cannot dominate the direction score.
            up = max(0, above["usd"]) / max(da, 1)
            down = max(0, below["usd"]) / max(db, 1)
            if up+down > 0:
                imbalance = (up-down)/(up+down)
                add("Liquidation zones", .30, imbalance,
                    f"Upper ${above['price']:,.0f} · lower ${below['price']:,.0f} (estimated)")

    if ctx and ctx.get("funding_pct") is not None:
        # Negative funding can signal shorts paying longs, but its
        # relationship with next-day price is uncertain and often contrarian.
        rate = float(ctx["funding_pct"])
        add("Funding", .10, -np.clip(rate/.03, -1, 1), f"{rate:+.4f}% · contrarian heuristic")

    if ctx and ctx.get("long_pct") is not None:
        long_pct = float(ctx["long_pct"])
        # Only strong account crowding contributes; the account count is not
        # the same as total leveraged exposure in dollars.
        bias = -np.clip((long_pct - 50)/35, -1, 1)
        add("Long/short accounts", .10, bias,
            f"{long_pct:.1f}% measured accounts long · contrarian heuristic")

    if etf_history and len(etf_history) >= 7:
        try:
            latest_date = pd.Timestamp(etf_history[-1][0]).date()
            age = (pd.Timestamp.now(tz='UTC').date() - latest_date).days
            if 0 <= age <= 4:
                seven = sum(float(v) for _, v in etf_history[-7:])
                # BTC/ETH ETFs use different USD flow magnitudes. Normalize
                # by each series' recent typical absolute daily flow instead
                # of assuming BTC-scale dollars apply to ETH.
                recent = [abs(float(v)) for _, v in etf_history[-30:]]
                typical = max(float(np.median(recent)), 10)
                raw = np.clip(seven / (7*typical), -1, 1)
                add("ETF flows", .15, raw,
                    f"7 sessions {seven:+,.1f}M · as of {latest_date}")
        except (ValueError, TypeError, OverflowError):
            pass

    coverage = sum(c["weight"] for c in components)
    if coverage < .65 or len(components) < 3:
        return {"ready": False, "coverage": coverage, "components": components,
                "reason": "Not enough current independent signals for a directional reading."}

    signed = sum(c["weight"] * c["signal"] for c in components) / coverage
    oi_delta = ctx.get("oi_change_pct") if ctx else None
    if oi_scale and oi_delta is not None and np.isfinite(oi_delta):
        # Higher OI can amplify existing pressure, but has NO direction itself.
        amp = 1 + .10 * float(np.clip(oi_delta / 2.0, -1, 1))
        signed *= amp
    signed = float(np.clip(signed, -1, 1))
    up_score = int(round(50 + 50*signed))
    if up_score >= 57:
        direction, icon = "Bullish lean", "↗"
    elif up_score <= 43:
        direction, icon = "Bearish lean", "↘"
    else:
        direction, icon = "Mixed / neutral", "↔"
    return {"ready": True, "coverage": coverage, "components": components,
            "up_score": up_score, "down_score": 100-up_score,
            "direction": direction, "icon": icon, "signed": signed,
            "oi_delta": oi_delta}
